import torch
from zonopy import polyZonotope, matPolyZonotope

def batch_forward_occupancy(H_jrs,link_zonos,robot_params):

    zono_order = 20
    n_joints = robot_params['n_joints']
    H = robot_params['H']
    H_motor = [matPolyZonotope(torch.eye(4,dtype=torch.float32).unsqueeze(0))]
    FO_link = []
    for i in range(n_joints):
        H_motor_temp = H_motor[-1]@H[i]@H_jrs[i]
        H_motor.append(H_motor_temp.reduce_dep(zono_order))
        FO_link_temp = H_motor_temp@link_zonos[i]
        FO_link.append(FO_link_temp.reduce_dep(zono_order))
    return FO_link, H_motor[1:]

    return 


def forward_occupancy(rotatos,link_zonos,robot_params):
    '''
    P: <list>
    '''
    zono_order = 30
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    P_motor = [polyZonotope(torch.zeros(3,dtype=torch.float32).unsqueeze(0))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=torch.float32).unsqueeze(0))]
    FO_link = []
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce_dep(zono_order))
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        R_motor.append(R_motor_temp.reduce_dep(zono_order))
        FO_link_temp = R_motor[-1]@link_zonos[i] + P_motor[-1]
        FO_link.append(FO_link_temp.reduce_dep(zono_order))
        #print(FO_link_temp.n_dep_gens)
        #import pdb;pdb.set_trace()
    return FO_link, R_motor[1:], P_motor[1:]



'''
def forward_occupancy_fancy(qpos,qvel,joint_axes,P,link_zonos):
    # P: <list>
 
    assert len(qpos) == len(qvel) and len(qpos) == len(joint_axes) and len(qpos) == len(P)
    JRS_poly = load_JRS(qpos,qvel)
    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    rotato = get_rotato_from_jrs(JRS_poly,joint_axes)

    P_motor = {}
    R_motor = {}
    FO_link = {}
    for t in range(n_time_steps):
        P_motor[(-1,t)] = polyZonotope(torch.zeros(3,dtype=torch.float32))
        R_motor[(-1,t)] = matPolyZonotope(torch.eye(3,dtype=torch.float32))
        for i in range(n_joints):
            P_motor[(i,t)] = R_motor[(i-1,t)]@P[i] + P_motor[(i-1,t)]
            R_motor[(i,t)] = R_motor[(i-1,t)]@rotato[(i,t)]
            FO_link[(i,t)] = R_motor[(i,t)]@link_zonos[i] + P_motor[(i,t)]
        del P_motor[(-1,t)], R_motor[(-1,t)]
    return FO_link, R_motor, P_motor
'''

