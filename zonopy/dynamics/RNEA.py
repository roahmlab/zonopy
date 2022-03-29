import torch
from torch import Tensor
import zonopy as zp
from zonopy.load_jrs.load_jrs import load_traj_JRS
from zonopy.kinematics.rotatotope import get_rotato_pair_from_jrs

import time

def poly_zono_rnea(q_0, qd_0, uniform_bound, Kr, robot_params, use_gravity=True):
    
    q, qd, q_aux_d, qdd = load_traj_JRS(q_0, qd_0, uniform_bound, Kr)
    max_key = max(q.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    
    zono_order = 40
    # number of active joints
    #n_joints = robot_params['n_joints']
    mass = robot_params['mass']# link masses
    com = robot_params['com']# center of mass for each link
    # NOTE: which frame?
    I = robot_params['I']# inertia wrt to center of mass frame 
    P = robot_params['P']
    joint_axes = robot_params['joint_axes']

    rotato, rotato_t = get_rotato_pair_from_jrs(q,joint_axes)

    joint_pos = q
    joint_vel = qd
    joint_acc = qdd
    joint_vel_aux = q_aux_d
    
    # rotation axis of base frame
    z0 = Tensor([0,0,1])

    
    w, w_aux, wdot, linear_acc = {},{},{},{}
    F, N = {}, {}
    n_time_steps = 3
    # RNEA forward recursion
    for t in range(n_time_steps):
        w[(-1,t)] = torch.zeros(3)
        w_aux[(-1,t)] = torch.zeros(3)
        wdot[(-1,t)] = torch.zeros(3)
        linear_acc[(-1,t)] = - robot_params['gravity']
        for i in range(n_joints):
            print(i,t)
            w_temp = rotato_t[(i,t)]@w[(i-1,t)] + joint_vel[(i,t)]*joint_axes[i]
            w_aux_temp = rotato_t[(i,t)]@w_aux[(i-1,t)] + joint_vel[(i,t)]*joint_axes[i]
            prod1 = (rotato_t[(i,t)]@w_aux[(i-1,t)]).reduce(zono_order,'girard')
            prod2 = (joint_vel[(i,t)]*joint_axes[i]).reduce(zono_order,'girard')
            wdot_temp = rotato_t[(i,t)]@wdot[(i-1,t)] + zp.cross(prod1,prod2)+joint_acc[(i,t)]*joint_axes[i]
            # NOTE double reduce wdot for i > 0 ... ?
            # wdot[(i,t)] = rotato_t[(i,t)]@wdot[(i-1,t)] + zp.cross(rotato_t[(i,t)]@w_aux[(i-1,t)],joint_vel[(i,t)]*joint_axes[i])+joint_acc[(i,t)]*joint_axes[i] 
            
            linear_acc_temp = rotato_t[(i,t)]@(linear_acc[(i-1,t)]+zp.cross(wdot[(i-1,t)],P[i])) + zp.cross(w[(i-1,t)],zp.cross(w[(i-1,t)],P[i]))
            w[(i,t)] = w_temp.reduce(zono_order,'girard')
            w_aux[(i,t)] = w_aux_temp.reduce(zono_order,'girard')
            wdot[(i,t)] = wdot_temp.reduce(zono_order,'girard')
            linear_acc[(i,t)] = linear_acc_temp.reduce(zono_order,'girard')
 

            prod1 = zp.cross(wdot[(i,t)],com[i]).reduce(zono_order,'girard')
            prod2 = zp.cross(w_aux[(i,t)],com[i]).reduce(zono_order,'girard')
            
            linear_acc_com_temp = linear_acc[(i,t)] + prod1 + zp.cross(w[(i,t)],prod2)
            
            linear_acc_com_temp = linear_acc_com_temp.reduce(zono_order,'girard')
            F[(i,t)] = (mass[i]@linear_acc_com_temp).reduce(zono_order,'girard')
            prod1 =(I[i]@wdot[(i,t)]).reduce(zono_order,'girard')
            prod2 = (I[i]@w[(i,t)]).reduce(zono_order,'girard')
            N[(i,t)] = (prod1 + zp.cross(w_aux[(i,t)],prod2)).reduce(zono_order,'girard')

        #del w[(-1,t)], w_aux[(-1,t)], wdot[(-1,t)], linear_acc[(-1,t)]
    del w_temp, w_aux_temp, wdot_temp, linear_acc_temp, linear_acc_com_temp 
    
    f, n = {}, {}
    u = {}
    P.append(P[-1])
    

    # RNEA reverse recursion
    for t in range(n_time_steps):
        rotato[(n_joints,t)] = rotato[(n_joints-1,t)]        
        f[(n_joints,t)], n[(n_joints,t)] = torch.zeros(3),torch.zeros(3)
        for i in reversed(range(n_joints)):
            print(i,t)
            f_temp = rotato[(i+1,t)]@f[(i+1,t)]+F[(i,t)]
            f[(i,t)] = f_temp.reduce(zono_order,'girard')

            prod1 = (rotato[(i+1,t)]@n[(i+1,t)]).reduce(zono_order,'girard')
            prod2 = (rotato[(i+1,t)]@f[(i+1,t)]).reduce(zono_order,'girard')
            n_temp = N[(i,t)] + prod1 + zp.cross(com[i],F[(i,t)]) + zp.cross(P[i+1],prod2)
            n[(i,t)] = n_temp.reduce(zono_order,'girard')
            u[(i,t)] = zp.dot(joint_axes[i],n[(i,t)])


        # NOTE: NOTE: zp.cross for tensor and poly
        # NOTE: NOTE: mul for pz, look something with joint axes zp.outer?
        # NOTE: NOTE: zp.dot DONE, but pz mul should be able to include this
    return u


if __name__ == '__main__':
    from zonopy.load_urdf.load_robot import load_poly_zono_params

    q_0 = torch.tensor([0,0.1,0.3,0.1,0,0.2])
    qd_0 = torch.tensor([0.01,0.02,0.03,0,0,0.1])
    uniform_bound = 1.2
    Kr = torch.eye(6)

    _,_, poly_zono_params,_,_ = load_poly_zono_params('fetch')
    poly_zono_rnea(q_0, qd_0, uniform_bound, Kr, poly_zono_params)