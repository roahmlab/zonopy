# TODO VALIDATE

import torch
import zonopy as zp

def poly_zono_rnea(rotato,rotato_t, qd, q_aux_d, qdd, robot_params, use_gravity=True):
    
    zono_order = 40
    # number of active joints
    n_joints = robot_params['n_joints']
    mass = robot_params['mass']# link masses
    com = robot_params['com']# center of mass for each link
    I = robot_params['I']# inertia wrt to center of mass frame 
    P = robot_params['P']
    joint_axes = robot_params['joint_axes']

    joint_vel = qd
    joint_acc = qdd
    joint_vel_aux = q_aux_d

    
    # RNEA forward recursion
    w = [torch.zeros(3)]
    w_aux= [torch.zeros(3)]
    wdot = [torch.zeros(3)]
    linear_acc = [-robot_params['gravity']]
    F, N = [], []
    for i in range(n_joints):
        print(i)
        w_temp = rotato_t[i]@w[-1] + joint_vel[i]*joint_axes[i]
        w.append(w_temp.reduce(zono_order,'girard'))

        w_aux_temp = rotato_t[i]@w_aux[-1] + joint_vel_aux[i]*joint_axes[i]
        w_aux.append(w_aux_temp.reduce(zono_order,'girard'))

        prod1 = (rotato_t[i]@w_aux[-1]).reduce(zono_order,'girard')
        prod2 = (joint_vel[i]*joint_axes[i]).reduce(zono_order,'girard')
        wdot_temp = rotato_t[i]@wdot[-1] + zp.cross(prod1,prod2)+joint_acc[i]*joint_axes[i]
        wdot.append(wdot_temp.reduce(zono_order,'girard'))
        
        linear_acc_temp = rotato_t[i]@(linear_acc[-1]+zp.cross(wdot[-1],P[i])) + zp.cross(w[-1],zp.cross(w_aux[-1],P[i]))
        linear_acc.append(linear_acc_temp.reduce(zono_order,'girard'))

        prod1 = zp.cross(wdot[i],com[i])
        prod1 = prod1.reduce(zono_order,'girard')
        prod2 = zp.cross(w_aux[i],com[i])
        prod2 = prod2.reduce(zono_order,'girard')
        linear_acc_com_temp = linear_acc[i] + prod1 + zp.cross(w[i],prod2)
        linear_acc_com_temp = linear_acc_com_temp.reduce(zono_order,'girard')
        F_temp = (mass[i]@linear_acc_com_temp)
        F.append(F_temp.reduce(zono_order,'girard'))
        prod1 =(I[i]@wdot[i]).reduce(zono_order,'girard')
        prod2 = (I[i]@w[i]).reduce(zono_order,'girard')
        N_temp = (prod1 + zp.cross(w_aux[i],prod2)).reduce(zono_order,'girard')
        N.append(N_temp)
    
    #del w[0], w_aux[0], wdot[0], linear_acc[0]
    del w, w_aux, wdot, linear_acc
    del w_temp, w_aux_temp, wdot_temp, linear_acc_temp, linear_acc_com_temp 
    del F_temp, N_temp
    
    
    f, n, u = [torch.zeros(3)],[torch.zeros(3)],[]
    P.append(P[-1])
    rotato.append(rotato[-1])

    # RNEA reverse recursion
    for i in reversed(range(n_joints)):
        print(i)
        f_temp = rotato[i+1]@f[-1]+F[i]
        f.append(f_temp.reduce(zono_order,'girard'))

        prod1 = (rotato[i+1]@n[-1])
        prod1 = prod1.reduce(zono_order,'girard')
        prod2 = (rotato[i+1]@f[-2])
        prod2 = prod2.reduce(zono_order,'girard')
        n_temp = N[i] + prod1 + zp.cross(com[i],F[i]) + zp.cross(P[i+1],prod2)
        n.append(n_temp.reduce(zono_order,'girard'))
        u.append(zp.dot(joint_axes[i],n[-1]))
    f.reverse()
    n.reverse()
    u.reverse()
    return f[:-1],n[:-1],u

# Try 2
import numpy as np
from urchin import URDF, Link
from collections import OrderedDict
def pzrnea(robot: URDF, zono_order: int = 40, gravity: torch.Tensor = torch.tensor([0, 0, -9.81])):
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
    P = [joint.origin for joint in robot.joints]

    # Each frame's axis of rotation expressed in said frame
    z = [joint.axis for joint in robot.joints]

    # orientation of frame I with respect to frame i-1
    # generate matPolyZonotopes for R from q
    # needed?

    ## RUN
    # base link / frame
    w0 = torch.zeros(3)
    w0dot = torch.zeros(3)
    linear_acc0 = torch.zeros(3)
    w0_aux = torch.zeros(3)
    if gravity is not None:
        linear_acc0 = -gravity
    w_parent = w0
    w_aux_parent = w0_aux
    wdot_parent = w0dot
    linear_acc_parent = linear_acc0
    
    ## Forward recursion
    # for each joint

    joint_dict = OrderedDict()

    # Compute in reverse topological order, base to end
    for lnk in robot._reverse_topo:
        # If this is the base link, continue since there's no parent joint
        if lnk is robot.base_link:
            continue
        # get the path to the base and use the relevant joint
        # consider only the first two joint
        path = robot._paths_to_base[lnk]
        joint = robot._G.get_edge_data(path[0], path[1])["joint"]
        parent_joint = None
        if path[1] is not robot.base_link:
            parent_joint = robot._G.get_edge_data(path[1], path[2])["joint"]

            w_parent = joint_dict[parent_joint]['w']
            w_aux_parent = joint_dict[parent_joint]['w_aux']
            wdot_parent = joint_dict[parent_joint]['wdot']
            linear_acc_parent = joint_dict[parent_joint]['linear_acc']

        # Get the relevant rotato config (???)
        rotato_cfg = None
        if joint.mimic is not None:
            mimic_joint = robot._joint_map[joint.mimic.joint]
            if mimic_joint.name in cfg_map:
                rotato_cfg = cfg_map[mimic_joint.name]['R']
                rotato_cfg = joint.mimic.multiplier * rotato_cfg + joint.mimic.offset
                joint_vel = cfg_map[mimic_joint.name]['qd']
                joint_vel_aux = cfg_map[mimic_joint.name]['qd_aux']
                joint_acc = cfg_map[mimic_joint.name]['qdd']
        elif joint.name in cfg_map:
            rotato_cfg = cfg_map[joint.name]['R']
            joint_vel = cfg_map[joint.name]['qd']
            joint_vel_aux = cfg_map[joint.name]['qd_aux']
            joint_acc = cfg_map[joint.name]['qdd']
        else:
            rotato_cfg = torch.eye(3)

        # Get the transform for the joint
        joint_rot = joint.origin_tensor[0:3,0:3]
        joint_rot = joint_rot@rotato_cfg

        joint_pos = joint.origin_tensor[0:3,3]

        joint_axis = torch.as_tensor(joint.axis)

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
                                       torch.as_tensor(lnk.inertial.origin[0:3,3]),
                                       lnk.inertial.mass,
                                       torch.as_tensor(lnk.inertial.inertia),
                                       joint_tuple)
        
        out_w, out_w_aux, out_wdot, out_linear_acc, out_linear_acc_com, out_F, out_N = out
        joint_dict[joint] = {
            'w': out_w,
            'w_aux': out_w_aux,
            'wdot': out_wdot,
            'linear_acc': out_linear_acc,
            'linear_acc_com': out_linear_acc_com,
            'F': out_F,
            'N': out_N,
            'joint_rot': joint_rot,
            'joint_pos': joint_pos,
            'com': torch.as_tensor(lnk.inertial.origin[0:3,3]),
            'parent_joint': parent_joint
        }

        # If it's an end link, add an extra dictionary element
        if lnk in robot.end_links:
            # Use the lnk as a dummy element
            joint_dict[lnk] = {
                'f': torch.zeros(3),
                'n': torch.zeros(3),
                'joint_rot': zp.matPolyZonotope.eye(3),
                'joint_pos': zp.polyZonotope.zeros(3),
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
                                       parent_props['com'],
                                       parent_props['joint_pos'])
        
        f = getattr(parent_props, 'f', 0) + out[0]
        n = getattr(parent_props, 'n', 0) + out[1]
        parent_props['f'] = f
        parent_props['n'] = n

    # Calculate joint torques
    out_joint_dict = OrderedDict()
    for joint in joint_dict:
        if isinstance(joint, Link):
            continue
        out_joint_dict[joint] = {
            'force': joint_dict[joint]['f'],
            'moments': joint_dict[joint]['n']
        }
        if joint.joint_type in ['revolute', 'continuous']:
            joint_axis = torch.as_tensor(joint.axis)
            torque = joint_axis.T @ joint_dict[joint]['n']
            out_joint_dict[joint]['torque'] = torque
    
    return out_joint_dict


def pzrnea_forward_recursion(joint_rot, w, w_aux, wdot, linear_acc, joint_axis, joint_pos, com, mass, I, joint_tuple = None):
    # handle if we have any joint state or not
    w_contrib = 0
    w_aux_contrib = 0
    wdot_contrib = 0
    if joint_tuple is not None:
        # tuple has vel, vel_aux, and acc
        joint_vel, joint_vel_aux, joint_acc = joint_tuple
        w_contrib = joint_vel @ joint_axis
        w_aux_contrib = joint_vel_aux @ joint_axis
        wdot_contrib = zp.cross(joint_rot.T @ w_aux, joint_vel @ joint_axis) + joint_acc @ joint_axis
    
    # 6.45 angular velocity (and auxillary)
    out_w = joint_rot.T @ w + w_contrib
    out_w = out_w.reduce()
    out_w_aux = joint_rot.T @ w_aux + w_aux_contrib
    out_w_aux = out_w_aux.reduce()

    # 6.46 angular acceleration
    out_wdot = joint_rot.T @ wdot + wdot_contrib
    out_wdot = out_wdot.reduce()

    # 6.47 linear acceleration
    out_linear_acc = joint_rot.T @ (linear_acc + zp.cross(wdot, joint_pos) + zp.cross(w, zp.cross(w_aux, joint_pos)))
    out_linear_acc = out_linear_acc.reduce()

    # 6.48 linear acceleration of COM aux
    out_linear_acc_com = out_linear_acc + zp.cross(out_wdot, com) + zp.cross(out_w, zp.cross(out_w_aux, com))
    out_linear_acc_com.reduce()

    # 6.49 calculate forces
    out_F = mass @ out_linear_acc_com
    out_F = out_F.reduce()

    # 6.50 calculate torques
    out_N = I @ out_wdot + zp.cross(out_w_aux, I @ out_w)
    out_N = out_N.reduce()
    return out_w, out_w_aux, out_wdot, out_linear_acc, out_linear_acc_com, out_F, out_N

def pzrnea_reverse_recursion(R, f, F, n, N, com, P):
    # this is going backwards!
    # 6.51 compute forces at each joint
    out_f = R @ f + F
    out_f.reduce()

    # 6.52 compute moments at each joint
    out_n = N + R @ n + zp.cross(com, F) + zp.cross(P, R @ f)
    out_n.reduce()
    return out_f, out_n

def pzrnea_torques(z, n, joint_tuple = None):
    if joint_tuple is not None:
        return z.T @ n
    else:
        return None




if __name__ == '__main__':
    from zonopy.load_urdf.load_robot import load_poly_zono_params

    q_0 = torch.tensor([0,0.1,0.3,0.1,0,0.2])
    qd_0 = torch.tensor([0.01,0.02,0.03,0,0,0.1])
    uniform_bound = 1.2
    Kr = torch.eye(6)

    _,_, poly_zono_params,_,_ = load_poly_zono_params('fetch')
    poly_zono_rnea(q_0, qd_0, uniform_bound, Kr, poly_zono_params)