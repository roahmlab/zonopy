import torch
from zonopy import polyZonotope, matPolyZonotope

def forward_kinematics(rotatos,robot_params):
    '''
    P: <list>
    '''
    zono_order= 40
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    P_motor = [polyZonotope(torch.zeros(3,dtype=torch.float32))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=torch.float32))]
    
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce(zono_order))      
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        R_motor.append(R_motor_temp.reduce(zono_order))
    return R_motor[1:], P_motor[1:]
