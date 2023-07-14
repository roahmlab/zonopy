import torch
from zonopy import polyZonotope, matPolyZonotope

from collections import OrderedDict
import numpy as np

# def forward_kinematics(rotatos,robot_params,zono_order= 20):
#     '''
#     P: <list>
#     '''
#     n_joints = robot_params['n_joints']
#     P = robot_params['P']
#     R = robot_params['R']
#     P_motor = [polyZonotope(torch.zeros(3,dtype=torch.float32))]
#     R_motor = [matPolyZonotope(torch.eye(3,dtype=torch.float32))]
    
#     for i in range(n_joints):
#         P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
#         P_motor.append(P_motor_temp.reduce(zono_order))      
#         R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
#         R_motor.append(R_motor_temp.reduce(zono_order))
#     return R_motor[1:], P_motor[1:]

# This is based on the Urchin's FK source
def forward_kinematics(rotatotopes, robot, zono_order = 20, joint_names=None, links=None):
    # If we aren't given any joints, assume it's for all actuated joints
    if joint_names is None:
        joint_names = robot.actuated_joint_names
    
    # Make sure the number of joint names provided match up
    # to the number of joints in the rotatotopes
    assert len(joint_names) == len(rotatotopes)
    cfg_map = {name: rots for name, rots in zip(joint_names, rotatotopes)}

    # Get our output link set, assume it's all links if unspecified
    link_set = set()
    if links is not None:
        for lnk in links:
            link_set.add(robot.robot._link_map[lnk])
    else:
        link_set = robot.robot.links

    # Compute forward kinematics in reverse topological order, base to end
    fk = OrderedDict()
    for lnk in robot.robot._reverse_topo:
        if lnk not in link_set:
            continue
        # Get the path back to the base and build with that
        path = robot.robot._paths_to_base[lnk]
        pos = polyZonotope(torch.zeros(3, dtype=torch.float64).unsqueeze(0),compress=0,copy_Z=False)
        rot = matPolyZonotope(torch.eye(3, dtype=torch.float64).unsqueeze(0),compress=0,copy_Z=False)
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            joint = robot.robot._G.get_edge_data(child, parent)["joint"]

            rotato_cfg = None
            if joint.mimic is not None:
                mimic_joint = robot.robot._joint_map[joint.mimic.joint]
                if mimic_joint.name in cfg_map:
                    rotato_cfg = cfg_map[mimic_joint.name]
                    rotato_cfg = joint.mimic.multiplier * rotato_cfg + joint.mimic.offset
            elif joint.name in cfg_map:
                rotato_cfg = cfg_map[joint.name]
            else:
                rotato_cfg = torch.eye(3, dtype=torch.float64)
            
            # Get the transform for the joint
            joint_rot = torch.as_tensor(joint.origin[0:3,0:3], dtype=torch.float64)
            joint_pos = torch.as_tensor(joint.origin[0:3,3], dtype=torch.float64)
            joint_rot = joint_rot@rotato_cfg
            
            # We are moving from child to parent, so apply in reverse
            pos = joint_rot@pos + joint_pos
            rot = joint_rot@rot

            # Check existing FK to see if we can exit early
            if parent in fk:
                parent_pos, parent_rot = fk[parent]
                pos = parent_rot@pos + parent_pos
                rot = parent_rot@rot
                break

        # Save the values & reduce
        fk[lnk.name] = (pos.reduce(zono_order), rot.reduce(zono_order))

    return fk