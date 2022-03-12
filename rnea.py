import torch
from torch import Tensor
from interval import interval, matmul_interval, cross_interval
from get_params import get_robot_params, get_interval_params

def rx(th):
    # note: th should be a tensor
    c = torch.cos(th)
    s = torch.sin(th)

    R = torch.zeros((3,3)).to(th.device, th.dtype)
    R[0,0] = 1
    R[1,1] = c
    R[1,2] = -s
    R[2,1] = s
    R[2,2] = c

    return R


def ry(th):
    # note: th should be a tensor
    c = torch.cos(th)
    s = torch.sin(th)

    R = torch.zeros((3,3)).to(th.device, th.dtype)
    R[0,0] = c
    R[0,2] = s
    R[1,1] = 1
    R[2,0] = -s
    R[2,2] = c

    return R


def rz(th):
    # note: th should be a tensor
    c = torch.cos(th)
    s = torch.sin(th)

    R = torch.zeros((3,3)).to(th.device, th.dtype)
    R[0,0] = c
    R[0,1] = -s
    R[1,0] = s
    R[1,1] = c
    R[2,2] = 1

    return R


def cross(a,b):
    if isinstance(a, interval) or isinstance(b, interval):
        return cross_interval(a,b)
    else:
        return torch.cross(a,b)


def matmul(A,B):
    if isinstance(A, interval) or isinstance(B, interval):
        return matmul_interval(A, B)
    else:
        return torch.mm(A, B)


def rnea(q, qd, q_aux_d, qdd, use_gravity, robot_params):
    '''Robot parameters'''
    # use interval arithmetic
    use_interval = robot_params['use_interval']

    if not use_interval:
        # link masses
        mass = Tensor(robot_params['mass'])

        # center of mass for each link
        com = Tensor(robot_params['com'])

        # inertia wrt to center of mass frame
        I = Tensor(robot_params['I'])
    else:
        # link masses
        mass = robot_params['mass']

        # center of mass for each link
        com = robot_params['com']

        # inertia wrt to center of mass frame
        I = robot_params['I']
        for intv in I:
            print(intv)

    
    # number of active joints
    num_joints = robot_params['num_joints']
    
    # fixed transforms
    T0 = Tensor(robot_params['T0'])
    joint_axes = Tensor(robot_params['joint_axes'])


    ''' set inputs '''
    # TODO: why :
    # convert to tensor before proceeding?
    joint_pos = q
    joint_vel = qd
    joint_acc = qdd
    joint_vel_aux = q_aux_d


    ''' set up reference frames'''
    # rotation axis of base frame
    z0 = Tensor([0,0,1]).view(3,1)

    # orientation of frame i with respect to frame i-1
    # TODO: not quite sure
    R = torch.eye(3).repeat(num_joints+1,1,1).permute(1,2,0).contiguous()

    # position of the origin of frame i with respect to frame i-1
    P = torch.zeros(3, num_joints+1)

    # orientation of frame i-1 with respect to frame i
    R_t = torch.eye(3).repeat(num_joints,1,1).permute(1,2,0).contiguous()
    
    # Frame {i} axis of rotation expressed in Frame {i}
    z = torch.zeros(3, num_joints)

    # calculate frame-to-frame transformations based on DH table
    for i in range(num_joints):
        # orientation of Frame {i} with respect to Frame {i-1}
        dim = torch.where(joint_axes[:,i] !=0)

        if dim == (0,):
            R[:,:,i] = rx(joint_pos[i])
        elif dim == (1,):
            R[:,:,i] = ry(joint_pos[i])
        elif dim == (2,):
            R[:,:,i] = rz(joint_pos[i])
        else:
            assert False, "not matching any of the axis" 

        R_t[:,:,i] = R[:,:,i].t() # line 7
 
        # position of Frame {i} with respect to Frame {i-1}
        P[:,i] = T0[0:3, 3, i]
        
        # orientation of joint i axis of rotation with respect to Frame {i}
        z[:,i] = joint_axes[:,i]
    
 
    # get transform to end-effector
    if robot_params['num_bodies'] > robot_params['num_joints']:
        R[:, :, -1] = T0[0:3, 0:3, -1]
        P[:, -1] = T0[0:3, 3, i]
    
    
    ''' INITIALIZE '''
    # base link/frame
    w0 = torch.zeros(3,1)
    w0dot = torch.zeros(3,1)
    linear_acc0 = torch.zeros(3,1)
    w0_aux = torch.zeros(3,1) # auxilliary

    # set gravity
    if use_gravity:
        linear_acc0[:,0] = -Tensor(robot_params['gravity'])

    # angular velocity/acceleration
    w = torch.zeros(3, num_joints)
    wdot = torch.zeros(3, num_joints)
    w_aux = torch.zeros(3, num_joints)

    # linear acceleration of frame
    linear_acc = torch.zeros(3, num_joints)
    
    if not use_interval:
        # linear acceleration of com
        linear_acc_com = torch.zeros(3, num_joints)

        # link forces/torques
        F = torch.zeros(3, num_joints)
        N = torch.zeros(3, num_joints)
        
        # intialize f, n, u
        f = torch.zeros(3, num_joints + 1)
        n = torch.zeros(3, num_joints + 1)
        u = torch.zeros(num_joints,1)
    else:
        # linear acceleration of com
        linear_acc_com = interval(torch.zeros(3, num_joints), torch.zeros(3, num_joints))

        # link forces/torques
        F = interval(torch.zeros(3, num_joints), torch.zeros(3, num_joints))
        N = interval(torch.zeros(3, num_joints), torch.zeros(3, num_joints))
        
        # intialize f, n, u
        f = interval(torch.zeros(3, num_joints + 1), torch.zeros(3, num_joints + 1))
        n = interval(torch.zeros(3, num_joints + 1), torch.zeros(3, num_joints + 1))
        u = interval(torch.zeros(num_joints, 1), torch.zeros(num_joints,1))

    
    ''' RNEA forward recursion '''
    for i in range(num_joints):
        # PYTHON has zero indexing, soo...
        if i == 0:
            # (6.45) angular velocity
            w[:,i:i+1] = matmul(R_t[:,:,i], w0) + joint_vel[i] * z[:,i:i+1] # line 13

            # auxillary angular velocity
            w_aux[:,i:i+1] = matmul(R_t[:,:,i], w0_aux) + joint_vel_aux[i] * z[:,i:i+1] # line 13

            # (6.46) angular acceleration
            wdot[:,i:i+1] = matmul(R_t[:,:,i], w0dot) + cross(matmul(R_t[:,:,i], w0_aux), joint_vel[i] * z[:,i:i+1]) + joint_acc[i] * z[:,i:i+1] # line 15
                    
            # (6.47) linear acceleration       
            linear_acc[:,i:i+1] = matmul(R_t[:,:,i], linear_acc0) + cross(w0dot, P[:,i:i+1]) + cross(w0, cross(w0, P[:,i:i+1]))  # line 16 (TYPO IN PAPER)
                            
        else:
            # (6.45) angular velocity
            w[:,i:i+1] = matmul(R_t[:,:,i], w[:,i-1:i]) + joint_vel[i] * z[:,i:i+1] # line 13


            # auxillar angular velocity
            w_aux[:,i:i+1] = matmul(R_t[:,:,i], w_aux[:,i-1:i]) + joint_vel_aux[i] * z[:,i:i+1] # line 14


            # (6.46) angular acceleration
            wdot[:,i:i+1] = matmul(R_t[:,:,i], wdot[:,i-1:i]) + cross(matmul(R_t[:,:,i], w_aux[:,i-1:i]), joint_vel[i]*z[:,i:i+1]) + joint_acc[i] * z[:,i:i+1]  # line 15
        
                            
            # (6.47) linear acceleration
            linear_acc[:,i:i+1] = matmul(R_t[:,:,i], linear_acc[:,i-1:i]) + cross(wdot[:,i-1:i], P[:,i:i+1]) + cross(w[:,i-1:i], cross(w_aux[:,i-1:i],P[:,i:i+1])) # line 16 (TYPO IN PAPER)
 

        # (6.48) linear acceleration of CoM auxilliary
        linear_acc_com[:,i:i+1] = linear_acc[:,i:i+1] + cross(wdot[:,i:i+1],com[:,i:i+1]) + cross(w[:,i:i+1], cross(w_aux[:,i:i+1],com[:,i:i+1])) # line 23 (modified for standard RNEA)

        # (6.49) calculate forces
        # WARNING: difference between MATLAB and PYTHON here
        F[:,i:i+1] = mass[i] * linear_acc_com[:,i:i+1] # line 27

        # (6.50) calculate torques
        N[:,i:i+1] = matmul(I[i], wdot[:,i:i+1]) + cross(w_aux[:,i:i+1], matmul(I[i], w[:,i:i+1])) # calculated in line 29


    ''' RNEA reverse recursion '''
    for i in range(num_joints-1,-1,-1):
        # (6.51)
        f[:,i:i+1] = matmul(R[:,:,i+1], f[:,i+1:i+2]) + F[:,i:i+1] # line 28

        # (6.52)
        n[:,i:i+1] = N[:,i:i+1] + matmul(R[:,:,i+1], n[:,i+1:i+2]) + cross(com[:,i:i+1], F[:,i:i+1]) + cross(P[:,i+1:i+2], matmul(R[:,:,i+1], f[:,i+1:i+2])) # # line 29,  P(:,i) might not be right, line 29 (TYPO IN PAPER)

    # calculate joint torques
    for i in range(num_joints):
        # (6.53)
        u[i,0] = matmul(n[:,i:i+1].t(), z[:,i:i+1]) # line 31

    return u

if __name__ == '__main__':
    import json
    param_file = "fetch_arm_param.json"
    param = get_robot_params(param_file)
    param = get_interval_params(param_file)

    '''
    if param['use_interval']:
        I = Tensor([[0.00417,0.00443],[-0.00010,-0.00010],[0.00097,0.00103],
            [-0.00010,-0.00010], [0.00844,0.00896], [-0.00010,-0.00010],
            [0.00097,0.00103], [-0.00010,-0.00010], [0.00844,0.00896]])
        inf = I[:,0].reshape(3,3)
        sup = I[:,1].reshape(3,3)
        param['I'] = interval(inf,sup)
    '''

    q = torch.ones(6) * 0
    qd = torch.ones(6) * 0.2
    qdd = torch.ones(6) * 0.3
    q_aux_d = torch.ones(6) * 0.1
    import time
    start = time.time()
    print(rnea(q, qd, q_aux_d, qdd, True, param))
    end = time.time()
    print(f"took {end - start} seconds to compute")

    

