# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from urchin import URDF
import numpy as np
import torch

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

# Helper function to create a config dictionary from the rotatotopes if a list is provided
def make_rotato_cfg(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                       List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                    robot: URDF,
                    allow_incomplete: bool = False,
                    ) -> Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]]: 
    if isinstance(rotatotopes, dict):
        assert all(isinstance(x, str) for x in rotatotopes.keys()), "Keys for the rotato config are not strings!"
    elif isinstance(rotatotopes, (list, np.ndarray)):
        # Assume that this is for all actuated joints
        joint_names = [joint.name for joint in robot.actuated_joints]
        if allow_incomplete:
            joint_names = joint_names[:len(rotatotopes)]
        assert len(joint_names) == len(rotatotopes), "Unexpected number of rotatotopes!"
        rotatotopes = {name: rots for name, rots in zip(joint_names, rotatotopes)}
    else:
        raise TypeError
    return rotatotopes


# This is based on the Urchin's FK source
def forward_kinematics(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                          List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                       robot: URDF,
                       zono_order: int = 20,
                       links: List[str] = None,
                       ) -> OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                       Tuple[batchPolyZonotope, batchMatPolyZonotope]]]:
    # Create the rotato config dictionary
    cfg_map = make_rotato_cfg(rotatotopes, robot)

    # Get our output link set, assume it's all links if unspecified
    link_set = set()
    if links is not None:
        for lnk in links:
            link_set.add(robot._link_map[lnk])
    else:
        link_set = robot.links

    # Compute forward kinematics in reverse topological order, base to end
    fk = OrderedDict()
    for lnk in robot._reverse_topo:
        if lnk not in link_set:
            continue
        # Get the path back to the base and build with that
        path = robot._paths_to_base[lnk]
        pos = polyZonotope.zeros(3)
        rot = matPolyZonotope.eye(3)
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            joint = robot._G.get_edge_data(child, parent)["joint"]

            rotato_cfg = None
            if joint.mimic is not None:
                mimic_joint = robot._joint_map[joint.mimic.joint]
                if mimic_joint.name in cfg_map:
                    rotato_cfg = cfg_map[mimic_joint.name]
                    rotato_cfg = joint.mimic.multiplier * rotato_cfg + joint.mimic.offset
            elif joint.name in cfg_map:
                rotato_cfg = cfg_map[joint.name]
            else:
                rotato_cfg = torch.eye(3)
            
            # Get the transform for the joint
            # joint_rot = torch.as_tensor(joint.origin[0:3,0:3], dtype=torch.float64)
            # joint_pos = torch.as_tensor(joint.origin[0:3,3], dtype=torch.float64)
            joint_rot = joint.origin_tensor[0:3,0:3]
            joint_pos = joint.origin_tensor[0:3,3]
            joint_rot = joint_rot@rotato_cfg
            
            # We are moving from child to parent, so apply in reverse
            pos = joint_rot@pos + joint_pos
            rot = joint_rot@rot

            # Check existing FK to see if we can exit early
            if parent.name in fk:
                parent_pos, parent_rot = fk[parent.name]
                pos = parent_rot@pos + parent_pos
                rot = parent_rot@rot
                break

        # Save the values & reduce
        fk[lnk.name] = (pos.reduce_indep(zono_order), rot.reduce_indep(zono_order))

    return fk