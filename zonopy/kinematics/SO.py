# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from .FK import forward_kinematics
from zonopy.robots2.robot import ZonoArmRobot
import numpy as np

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

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
    lso = OrderedDict()
    for link_name in so_links:
        pj = robot.link_parent_joint[link_name].name
        pairs = [(pj, cj.name) for cj in robot.link_child_joints[link_name]]
        lso[link_name] = pairs
    
    return jso, lso, joint_fk_dict

import torch
def make_spheres(p1, p2, r1, r2, jac1 = None, jac2 = None, n_spheres: int = 5):
    ## Compute the terms
    height = torch.linalg.vector_norm(p2 - p1, dim=-1, keepdim=True)
    direction = (p2-p1)/height
    sidelength = height/(2*n_spheres)
    m = (2*torch.arange(n_spheres, dtype=p1.dtype, device=p1.device) + 1)
    offsets = torch.einsum('s,...i->...si', m, sidelength) # n_spheres, 1
    radial_delta = (r1-r2)/(2*n_spheres)
    centers = p1.unsqueeze(-2) + torch.einsum('...d,...si->...sd', direction, offsets) # batch_dims, n_spheres, 3
    radii = torch.sqrt(sidelength**2 + (r1.unsqueeze(-1) - torch.einsum('s,...->...s', m, radial_delta))**2) # batch_dims, n_spheres

    ## Compute the jacobians
    if jac1 is not None and jac2 is not None:
        djac = (jac2-jac1)
        jach = torch.einsum('...d,...dq->...q', direction, djac)
        jacu = (djac - torch.einsum('...d,...q->...dq',direction,jach))/height.unsqueeze(-1)
        jacd = torch.einsum('s,...q->...sq', m/(2*n_spheres), jach)
        jacc = jac1.unsqueeze(-3) + torch.einsum('...si,...dq->...sdq', offsets, jacu) + torch.einsum('...d,...sq->...sdq', direction, jacd) # batch_dims, n_spheres, 3, n_q
        jacr = torch.einsum('...q,...s->...sq', jach, sidelength / (2*n_spheres*radii)) # batch_dims, n_spheres, n_q
        return centers, radii, jacc, jacr
    return centers, radii
