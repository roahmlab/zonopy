import torch
from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.conSet.polynomial_zonotope.quat_poly_zono import quatPolyZonotope
from zonopy.load_jrs.load_jrs import load_JRS, load_qJRS
from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs, get_quatato_from_jrs

dim = 3

def forward_occupancy(qpos,qvel,joint_axes,P,link_zonos):
    '''
    P: <list>
    '''
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
        for i in range(n_joints):
            P_motor[(i,t)] = polyZonotope(torch.zeros(3,dtype=torch.float32))
            R_motor[(i,t)] = matPolyZonotope(torch.eye(3,dtype=torch.float32))
            for j in range(i+1):
                P_motor[(i,t)] = R_motor[(i,t)]@P[j]+P_motor[(i,t)]
                R_motor[(i,t)] = R_motor[(i,t)]@rotato[(j,t)]
            FO_link[(i,t)] = R_motor[(i,t)]@link_zonos[i] + P_motor[(i,t)]
    return FO_link, R_motor, P_motor

def forward_occupancy_fancy(qpos,qvel,joint_axes,P,link_zonos):
    '''
    P: <list>
    '''
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

def forward_occupancy_quat(qpos,qvel,joint_axes,P,link_zonos):
    '''
    P: <list>
    '''
    assert len(qpos) == len(qvel) and len(qpos) == len(joint_axes) and len(qpos) == len(P)
    JRS_poly = load_qJRS(qpos,qvel)
    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    quatato = get_quatato_from_jrs(JRS_poly,joint_axes)

    P_motor = {}
    R_motor = {}
    FO_link = {}
    for t in range(n_time_steps):
        for i in range(n_joints):
            P_motor[(i,t)] = polyZonotope(torch.zeros(3,dtype=torch.float32))
            R_motor[(i,t)] = quatPolyZonotope(torch.tensor([1,0,0,0],dtype=torch.float32))
            for j in range(i+1):
                P_motor[(i,t)] = R_motor[(i,t)]@P[j]+P_motor[(i,t)]
                R_motor[(i,t)] = R_motor[(i,t)]*quatato[(j,t)]
            FO_link[(i,t)] = R_motor[(i,t)]@link_zonos[i] + P_motor[(i,t)]
    return FO_link, R_motor, P_motor