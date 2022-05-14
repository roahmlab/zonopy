import torch
from zonopy import polyZonotope, matPolyZonotope

def forward_occupancy(rotatos,link_zonos,robot_params):
    '''
    P: <list>
    '''
    zono_order = 40
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    P_motor = [polyZonotope(torch.zeros(3,dtype=torch.float32).unsqueeze(0))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=torch.float32).unsqueeze(0))]
    FO_link = []
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce_indep(zono_order))
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        #print(f'{i}-th joint n gens before reduce: {R_motor_temp.n_generators}')
        #print(f'{i}-th joint n gens after reduce: {R_motor_temp.reduce_indep(zono_order).n_generators}')
        R_motor.append(R_motor_temp.reduce_indep(zono_order))

        FO_link_temp = R_motor[-1]@link_zonos[i] + P_motor[-1]
        FO_link.append(FO_link_temp.reduce_indep(zono_order))
        #print(FO_link_temp.n_dep_gens)
    #import pdb;pdb.set_trace()
    return FO_link, R_motor[1:], P_motor[1:]
