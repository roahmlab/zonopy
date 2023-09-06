# TODO VALIDATE

import torch
import zonopy as zp
import numpy as np
from urchin import URDF, Link
from collections import OrderedDict
from zonopy.robots2.robot import ZonoArmRobot

from typing import Dict, Any, Union, List

# Helper function to create a config dictionary from the rotatotopes if a list is provided
def make_cfg_dict(configs: Union[Dict[str, Any], List[Any]],
                  robot: URDF,
                  allow_incomplete: bool = False,
                  ) -> Dict[str, Any]: 
    if isinstance(configs, dict):
        assert all(isinstance(x, str) for x in configs.keys()), "Keys for the config dict are not strings!"
    elif isinstance(configs, (list, np.ndarray)):
        # Assume that this is for all actuated joints
        joint_names = [joint.name for joint in robot.actuated_joints]
        if allow_incomplete:
            joint_names = joint_names[:len(configs)]
        assert len(joint_names) == len(configs), "Unexpected number of configs!"
        configs = {name: cfgs for name, cfgs in zip(joint_names, configs)}
    else:
        raise TypeError
    return configs

def pzrnea(rotatotopes,
           qd,
           qd_aux,
           qdd,
           robot: ZonoArmRobot,
           zono_order: int = 40,
           gravity: torch.Tensor = torch.tensor([0, 0, -9.81]),
           link_mass_override: Dict[str, Any] = None,
           link_center_mass_override: Dict[str, Any] = None,
           link_inertia_override: Dict[str, Any] = None):
    # Mass override
    # COM override
    # Intertia override
    
    # get mass
    # cet center of mass
    # get inertia wrt to com frame
    # get n_joints?
    # get n_actuated_joints
    # get transforms

    ## Inputs
    # joint_vel
    # joint_acc
    # joint_vel_aux

    # Prep
    # rot axis of base frame
    # TODO: torch or np?
    # z0 = np.array([0, 0, 1.])
    # irrelevant

    # position of each joint frame with respect to the prior frame
    # P = [joint.origin for joint in robot.joints]

    # Each frame's axis of rotation expressed in said frame
    # z = [joint.axis for joint in robot.joints]

    # orientation of frame I with respect to frame i-1
    # generate matPolyZonotopes for R from q
    # needed?

    ## RUN
    rot_map = make_cfg_dict(rotatotopes, robot.urdf)
    qd_map = make_cfg_dict(qd, robot.urdf)
    qd_aux_map = make_cfg_dict(qd_aux, robot.urdf)
    qdd_map = make_cfg_dict(qdd, robot.urdf)
    urdf = robot.urdf

    # Update all overrides
    link_mass = {link.name: robot.link_data[link].mass for link in urdf.links}
    if link_mass_override is not None:
        link_mass.update(link_mass_override)

    link_center_mass = {link.name: robot.link_data[link].center_mass for link in urdf.links}
    if link_center_mass_override is not None:
        link_center_mass.update(link_center_mass_override)

    link_inertia = {link.name: robot.link_data[link].inertia for link in urdf.links}
    if link_inertia_override is not None:
        link_inertia.update(link_inertia_override)
    
    # Set up initial conditions
    w_parent = torch.zeros(3, device=robot.device, dtype=robot.dtype)
    wdot_parent = torch.zeros(3, device=robot.device, dtype=robot.dtype)
    linear_acc_parent = torch.zeros(3, device=robot.device, dtype=robot.dtype)
    w_aux_parent = torch.zeros(3, device=robot.device, dtype=robot.dtype)
    if gravity is not None:
        linear_acc_parent = -torch.as_tensor(gravity, device=robot.device, dtype=robot.dtype)
    
    ## Forward recursion
    # for each joint

    joint_dict = OrderedDict()

    # Compute in reverse topological order, base to end
    # This is the slowest part (needs profiling)
    for lnk in urdf._reverse_topo:
        # If this is the base link, continue since there's no parent joint
        if lnk is urdf.base_link:
            continue
        # get the path to the base and use the relevant joint
        # consider only the first two joint
        path = urdf._paths_to_base[lnk]
        joint = urdf._G.get_edge_data(path[0], path[1])["joint"]
        parent_joint = None
        if path[1] is not urdf.base_link:
            parent_joint = urdf._G.get_edge_data(path[1], path[2])["joint"]

            w_parent = joint_dict[parent_joint]['w']
            w_aux_parent = joint_dict[parent_joint]['w_aux']
            wdot_parent = joint_dict[parent_joint]['wdot']
            linear_acc_parent = joint_dict[parent_joint]['linear_acc']

        # Get the relevant configs
        # TODO determine mimic behaviors
        name = joint.name
        multiplier = 1.
        offset = 0.
        if joint.mimic is not None:
            mimic_joint = urdf._joint_map[joint.mimic.joint]
            name = mimic_joint.name
            multiplier = joint.mimic.multiplier
            offset = joint.mimic.offset
            import warnings
            warnings.warn('Mimic joint may not work')

        rotato_cfg = rot_map.get(name, torch.eye(3, dtype=robot.dtype, device=robot.device))
        joint_vel = qd_map.get(name, 0)
        joint_vel_aux = qd_aux_map.get(name, 0)
        joint_acc = qdd_map.get(name, 0)
        # rotato_cfg = multiplier * rotato_cfg + offset
        # joint_vel = qd_map.get(name, 0) * multiplier
        # joint_vel_aux = qd_aux_map.get(name, 0) * multiplier
        # joint_acc = qdd_map.get(name, 0) * multiplier

        # Get the transform for the joint
        # joint_rot = joint.origin_tensor[0:3,0:3]
        joint_rot = robot.joint_data[joint].origin[0:3,0:3]
        joint_rot = joint_rot@rotato_cfg

        # joint_pos = joint.origin_tensor[0:3,3]
        joint_pos = robot.joint_data[joint].origin[0:3,3]

        joint_axis = robot.joint_data[joint].axis

        if joint.joint_type in ['revolute', 'continuous']:
            joint_tuple = (joint_vel, joint_vel_aux, joint_acc)
        else:
            joint_tuple = None

        out = pzrnea_forward_recursion(joint_rot,
                                       w_parent,
                                       w_aux_parent,
                                       wdot_parent,
                                       linear_acc_parent,
                                       joint_axis,
                                       joint_pos,
                                       link_mass[lnk.name],
                                       link_center_mass[lnk.name],
                                       link_inertia[lnk.name],
                                       joint_tuple,
                                       zono_order)
        
        out_w, out_w_aux, out_wdot, out_linear_acc, out_F, out_N = out
        joint_dict[joint] = {
            'w': out_w,
            'w_aux': out_w_aux,
            'wdot': out_wdot,
            'linear_acc': out_linear_acc,
            'F': out_F,
            'N': out_N,
            'joint_rot': joint_rot,
            'joint_pos': joint_pos,
            'child_link': lnk,
            'parent_joint': parent_joint
        }

        # If it's an end link, add an extra dictionary element
        if lnk in urdf.end_links:
            # Use the lnk as a dummy element. This denotes the initial conditions for each reverse
            joint_dict[lnk] = {
                'f': torch.zeros(3, dtype=robot.dtype, device=robot.device),
                'n': torch.zeros(3, dtype=robot.dtype, device=robot.device),
                'joint_rot': zp.matPolyZonotope.eye(3, dtype=robot.dtype, device=robot.device),
                'joint_pos': zp.polyZonotope.zeros(3, dtype=robot.dtype, device=robot.device),
                'parent_joint': joint
            }
    
    # Reverse recursion
    for props in reversed(joint_dict.values()):
        if props['parent_joint'] is None:
            continue
        parent_props = joint_dict[props['parent_joint']]

        out = pzrnea_reverse_recursion(props['joint_rot'],
                                       props['f'],
                                       parent_props['F'],
                                       props['n'],
                                       parent_props['N'],
                                       robot.link_data[parent_props['child_link']].center_mass,
                                       props['joint_pos'],
                                       zono_order)
        
        f = getattr(parent_props, 'f', 0) + out[0]
        n = getattr(parent_props, 'n', 0) + out[1]

        # Reduce here because it blows up otherwise
        parent_props['f'] = f.reduce(zono_order)
        parent_props['n'] = n.reduce(zono_order)

    # Calculate joint torques
    out_joint_dict = OrderedDict()
    for joint in joint_dict:
        if isinstance(joint, Link):
            continue
        out_joint_dict[joint] = {
            'force': joint_dict[joint]['f'],
            'moment': joint_dict[joint]['n']
        }
        if joint.joint_type in ['revolute', 'continuous']:
            joint_axis = robot.joint_data[joint].axis
            torque = joint_axis.unsqueeze(0) @ joint_dict[joint]['n'] # dot product
            out_joint_dict[joint]['torque'] = torque
    
    return out_joint_dict


def pzrnea_forward_recursion(joint_rot, w, w_aux, wdot, linear_acc, joint_axis, joint_pos, mass, com, I, joint_tuple = None, zono_order = 40):
    # handle if we have any joint state or not
    w_contrib = 0
    w_aux_contrib = 0
    wdot_contrib = 0
    if joint_tuple is not None:
        # tuple has vel, vel_aux, and acc
        joint_vel, joint_vel_aux, joint_acc = joint_tuple
        w_contrib = joint_vel * joint_axis
        w_aux_contrib = joint_vel_aux * joint_axis
        wdot_contrib = zp.cross(joint_rot.T @ w_aux, joint_vel * joint_axis) + joint_acc * joint_axis
    
    # 6.45 angular velocity (and auxillary)
    out_w = joint_rot.T @ w + w_contrib
    out_w = out_w.reduce(zono_order)
    out_w_aux = joint_rot.T @ w_aux + w_aux_contrib
    out_w_aux = out_w_aux.reduce(zono_order)

    # 6.46 angular acceleration
    out_wdot = joint_rot.T @ wdot + wdot_contrib
    out_wdot = out_wdot.reduce(zono_order)

    # 6.47 linear acceleration
    out_linear_acc = joint_rot.T @ (linear_acc + zp.cross(wdot, joint_pos) + zp.cross(w, zp.cross(w_aux, joint_pos)))
    out_linear_acc = out_linear_acc.reduce(zono_order) # reduce here because this blows up

    # 6.48 linear acceleration of COM aux
    out_linear_acc_com = out_linear_acc + zp.cross(out_wdot, com) + zp.cross(out_w, zp.cross(out_w_aux, com))
    # out_linear_acc_com.reduce_indep(zono_order)

    # 6.49 calculate forces
    out_F = mass * out_linear_acc_com
    out_F = out_F.reduce_indep(zono_order)

    # 6.50 calculate torques
    out_N = I @ out_wdot + zp.cross(out_w_aux, I @ out_w)
    out_N = out_N.reduce_indep(zono_order)
    return out_w, out_w_aux, out_wdot, out_linear_acc, out_F, out_N

def pzrnea_reverse_recursion(R, f, F, n, N, com, P, zono_order = 40):
    # this is going backwards!
    # 6.51 compute forces at each joint
    out_f = R @ f + F
    out_f.reduce_indep(zono_order)

    # 6.52 compute moments at each joint
    out_n = N + R @ n + zp.cross(com, F) + zp.cross(P, R @ f)
    out_n.reduce_indep(zono_order)
    return out_f, out_n

