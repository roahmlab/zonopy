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


if __name__ == '__main__':
    from zonopy.load_urdf.load_robot import load_poly_zono_params

    q_0 = torch.tensor([0,0.1,0.3,0.1,0,0.2])
    qd_0 = torch.tensor([0.01,0.02,0.03,0,0,0.1])
    uniform_bound = 1.2
    Kr = torch.eye(6)

    _,_, poly_zono_params,_,_ = load_poly_zono_params('fetch')
    poly_zono_rnea(q_0, qd_0, uniform_bound, Kr, poly_zono_params)