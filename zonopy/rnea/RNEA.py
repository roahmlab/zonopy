import torch

from zonopy.conSet import zonotope, matPolyZonotope

def poly_zonotope_rnea(q,q_d, q_aux_d, q_dd, robot_params, use_gravity=True):
    '''
    , return
    u: <n,1> dict of tortatotopes

    '''
    num_joints = robot_params['num_joints']
    mass = torch.tensor(robot_params['mass'])
    com = torch.tensor(robot_params['com'])
    I = torch.tensor(robot_params['I'])
    P = torch.tensor(robot_params['P'])
    z = torch.tensor(robot_params['joint_axes'])

    joint_pos, joint_vel, joint_acc, joint_vel_aux = q, q_d, q_dd, q_aux_d
    
    z0 = torch.zeros(3,1)

    JRS_poly = load_JRS(joint_pos,joint_vel)
    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    rotato,rotato_T = get_rotato_from_jrs(JRS_poly,z)

    w0 = torch.zeros(3,1)
    w0_dot = torch.zeros(3,1)
    linear_acc0 = torch.zeros(3,1)
    w0_aux = torch.zeros(3,1)

    if use_gravity:
        # NOTE: shape
        linear_acc0 = - torch.tensor(robot_params['gravity'])
    
    
    # RNEA foward recursion
    w, w_aux, w_dot, linear_acc = {},{},{},{}
    for i in range(num_joints):
        w[i] = rotato_T @



    # RNEA backward recursion


    return u


