# TODO CLEANUP IMPORTS & DOCUMENT

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from .FK import forward_kinematics
from zonopy.robots2.robot import ZonoArmRobot

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

# Use forward occupancy or forward kinematics to get the joint occupancy
# For the true bohao approach, enable use_outer_bb
def joint_occupancy(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                       List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                    robot: ZonoArmRobot,
                    zono_order: int = 20,
                    joints: List[str] = None,
                    joint_zono_override: Dict[str, polyZonotope] = {},
                    use_outer_bb: bool = False,
                    ) -> Tuple[OrderedDictType[str, Union[polyZonotope, batchPolyZonotope]],
                               OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                          Tuple[batchPolyZonotope, batchMatPolyZonotope]]]]:
    
    urdf = robot.urdf
    # Process out the joint bb's
    if joints is None:
        joints = [joint.name for joint in urdf.actuated_joints]
    
    # Identify the links we are going to care about
    links = [urdf._joint_map[name].child for name in joints]

    # Create the map of joint zonos
    joint_zonos = {}
    for name in joints:
        if name in joint_zono_override:
            joint_zonos[name] = joint_zono_override[name]
        elif use_outer_bb:
            joint_zonos[name] = robot.joint_data[urdf._joint_map[name]].outer_pz
        else:
            joint_zonos[name] = robot.joint_data[urdf._joint_map[name]].bounding_pz

    # Get forward kinematics
    link_fk_dict = forward_kinematics(rotatotopes, robot, zono_order, links=links)

    # Redirect to the respective joint name
    joint_fk_dict = OrderedDict()
    for name, val in link_fk_dict.items():
        joint_name = robot.link_parent_joint[name].name
        joint_fk_dict[joint_name] = val

    # Create the joint occupancy
    jo = OrderedDict()

    # If we are doing bohao's method, skip rotation
    if use_outer_bb:
        for name, (pos, rot) in joint_fk_dict.items():
            jo_joint = pos + joint_zonos[name]
            jo[name] = jo_joint.reduce_indep(zono_order)
    # Otherwise rotate
    else:
        for name, (pos, rot) in joint_fk_dict.items():
            jo_joint = pos + rot@joint_zonos[name]
            jo[name] = jo_joint.reduce_indep(zono_order)
    
    return jo, joint_fk_dict