# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from .FK import forward_kinematics
from zonopy.robots2.robot import ZonoArmRobot
from urchin import URDF
import numpy as np
import functools

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType


@functools.lru_cache
def __get_fk_so_links(urdf: URDF,
                      links: Tuple[str, ...] = None
                      ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    fk_links = None # the links we use for FK
    so_links = None # the links we actually generate the SO for
    if links is not None:
        fk_links = set()
        so_links = []
        for link_name in links:
            link = urdf.link_map[link_name]
            next_link_list = list(urdf._G.reverse()[link].values())
            if len(next_link_list) == 0:
                import warnings
                warnings.warn('Not generating SO for ' + link_name)
                continue
            for nl in next_link_list:
                fk_links.add(nl.name)
            fk_links.add(link_name)
            so_links.append(link_name)
        fk_links = list(fk_links)
    else:
        so_links = urdf.links.copy()
        so_links.remove(urdf.base_link)
        fk_links = [link.name for link in so_links]
        for el in urdf.end_links:
            so_links.remove(el)
        so_links = [link.name for link in so_links]
    return tuple(fk_links), tuple(so_links)


@functools.lru_cache
def __make_link_pairs(robot: ZonoArmRobot,
                      so_links: Tuple[str, ...]
                      ) -> OrderedDictType[str, Tuple[str,str]]:
    # Create the link occupancy pairs
    lso = OrderedDict()
    for link_name in so_links:
        pj = robot.link_parent_joint[link_name].name
        pairs = [(pj, cj.name) for cj in robot.link_child_joints[link_name]]
        lso[link_name] = pairs
    return lso


# Use forward occupancy or forward kinematics to get the joint occupancy
# For the true bohao approach, enable use_outer_bb
# ONLY USE WITH LINKS THAT ARE NOT BASE OR END LINKS
def sphere_occupancy(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                        List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                     robot: ZonoArmRobot,
                     zono_order: int = 20,
                     links: List[str] = None,
                     joint_radius_override: Dict[str, float] = {},
                     ) -> Tuple[OrderedDictType[str, Union[polyZonotope, batchPolyZonotope]],
                                OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                           Tuple[batchPolyZonotope, batchMatPolyZonotope]]]]:
    
    urdf = robot.urdf
    if len(urdf.end_links) > 1:
        import warnings
        warnings.warn('The robot appears to be branched! Branched robots may not function as expected.')
    
    # Get adjacent links as needed.
    # Remove end links
    # fk_links: the links we use for FK
    # so_links: the links we actually generate the SO for
    fk_links, so_links = __get_fk_so_links(urdf, tuple(links) if links is not None else None)

    # Get forward kinematics
    link_fk_dict = forward_kinematics(rotatotopes, robot, zono_order, links=fk_links)

    # Redirect to the respective joint name
    joint_fk_dict = OrderedDict()
    for name, val in link_fk_dict.items():
        joint = robot.link_parent_joint[name]
        joint_fk_dict[joint.name] = val
    
    # Process out the joint radii
    joints = list(joint_fk_dict.keys())
    joint_radii = {}
    for name in joints:
        if name in joint_radius_override:
            joint_radii[name] = joint_radius_override[name]
        else:
            # Make sure create_joint_occupany on the robot is set!
            joint_radii[name] = robot.joint_data[urdf._joint_map[name]].radius.max()

    # Create the joint occupancy
    jso = OrderedDict()

    # Get the max outer sphere
    for name, (pos, _) in joint_fk_dict.items():
        joint_center, joint_uncertainty = pos.split_dep_indep()

        uncertainty_rad = joint_uncertainty.to_interval().rad().max(-1)[0]*np.sqrt(3)
        link_radius = robot.joint_data[urdf._joint_map[name]].radius.max()
        radius = uncertainty_rad + link_radius

        jso[name] = (joint_center, radius)
    
    # Create the link occupancy pairs
    lso = __make_link_pairs(robot, so_links)
    
    return jso, lso, joint_fk_dict

import torch
def make_spheres(center_1: torch.Tensor,
                 center_2: torch.Tensor,
                 radius_1: torch.Tensor,
                 radius_2: torch.Tensor,
                 center_jac_1: torch.Tensor = None,
                 center_jac_2: torch.Tensor = None,
                 n_spheres: int = 5
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Generates centers and radii for spheres that overapproximate the taper between two spheres.
    
    Args:
        center_1: A tensor of shape (batch_dims, 3) representing the center(s) of the first sphere.
        center_2: A tensor of shape (batch_dims, 3) representing the center(s) of the second sphere.
        radius_1: A tensor of shape (batch_dims,) representing the radius/radii of the first sphere.
        radius_2: A tensor of shape (batch_dims,) representing the radius/radii of the second sphere.
        center_jac_1: A tensor of shape (batch_dims, 3, n_q) representing the Jacobian(s) to the n_q parameters that parameterize the center of the first sphere.
        center_jac_2: A tensor of shape (batch_dims, 3, n_q) representing the Jacobian(s) to the n_q parameters that parameterize the center of the second sphere.
        n_spheres: An integer representing the number of spheres to generate.
    
    Returns:
        A tuple of two tensors if center_jac_1 and center_jac_2 are None, otherwise a tuple of four tensors:
            centers: A tensor of shape (batch_dims, n_spheres, 3) representing the centers of the generated spheres.
            radii: A tensor of shape (batch_dims, n_spheres) representing the radii of the generated spheres.
            center_jac: A tensor of shape (batch_dims, n_spheres, 3, n_q) representing the Jacobians to the n_q parameters for the centers of the generated spheres.
            radii_jac: A tensor of shape (batch_dims, n_spheres, n_q) representing the Jacobians to the n_q parameters for the radii of the generated spheres.
    '''
    ## Compute the terms
    height = torch.linalg.vector_norm(center_2 - center_1, dim=-1, keepdim=True)
    direction = (center_2-center_1)/height
    sidelength = height/(2*n_spheres)
    m = (2*torch.arange(n_spheres, dtype=center_1.dtype, device=center_1.device) + 1)
    offsets = torch.einsum('s,...i->...si', m, sidelength) # n_spheres, 1
    radial_delta = (radius_1-radius_2)/(2*n_spheres)
    centers = center_1.unsqueeze(-2) + torch.einsum('...d,...si->...sd', direction, offsets) # batch_dims, n_spheres, 3
    radii = torch.sqrt(sidelength**2 + (radius_1.unsqueeze(-1) - torch.einsum('s,...->...s', m, radial_delta))**2) # batch_dims, n_spheres

    ## Compute the jacobians
    if center_jac_1 is not None and center_jac_2 is not None:
        djac = (center_jac_2-center_jac_1)
        jach = torch.einsum('...d,...dq->...q', direction, djac)
        jacu = (djac - torch.einsum('...d,...q->...dq',direction,jach))/height.unsqueeze(-1)
        jacd = torch.einsum('s,...q->...sq', m/(2*n_spheres), jach)
        jacc = center_jac_1.unsqueeze(-3) + torch.einsum('...si,...dq->...sdq', offsets, jacu) + torch.einsum('...d,...sq->...sdq', direction, jacd) # batch_dims, n_spheres, 3, n_q
        jacr = torch.einsum('...q,...s->...sq', jach, sidelength / (2*n_spheres*radii)) # batch_dims, n_spheres, n_q
        return centers, radii, jacc, jacr
    return centers, radii
