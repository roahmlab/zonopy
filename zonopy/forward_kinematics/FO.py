import torch
from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.load_jrs.load_jrs import load_JRS
from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs

dim = 3

def forward_occupancy(qpos,qvel,joint_axes,P,link_zonos):
    '''
    P: <list>
    '''
    assert len(qpos) == len(qvel) and len(qpos) == len(joint_axes) and len(qpos) == len(P) and len(qpos) == len(link_zonos)
    JRS_poly = load_JRS(qpos,qvel)
    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    rotato = get_rotato_from_jrs(JRS_poly,joint_axes)

    link_FO = {}
    for t in range(n_time_steps):
        for i in range(n_joints):
            link_FO[(i,t)] = rotato[(i,t)]@link_zonos[i]
            for j in reversed(range(i)):
                link_FO[(i,t)] =rotato[(j,t)]@(link_FO[(i,t)]+P[j]) 

    return link_FO


