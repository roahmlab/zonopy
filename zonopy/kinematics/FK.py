import torch
from zonopy import polyZonotope, matPolyZonotope

def forward_kinematics(rotatos,robot_params):
    '''
    P: <list>
    '''
    n_joints = robot_params['n_joints']
    P = robot_params['P']

    P_motor = [polyZonotope(torch.zeros(3,dtype=torch.float32))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=torch.float32))]
    
    for i in range(n_joints):
        P_motor.append(R_motor[-1]@P[i] + P_motor[-1])
        R_motor.append(R_motor[-1]@rotatos[i])
    return R_motor[1:], P_motor[1:]
