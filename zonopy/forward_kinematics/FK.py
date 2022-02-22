import torch
from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.load_jrs.load_jrs import load_JRS
from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs

dim = 3

def forward_kinematics(qpos,qvel,joint_axes,P):
    '''
    P: <list>
    '''
    assert len(qpos) == len(qvel) and len(qpos) == len(joint_axes) and len(qpos) == len(P)
    JRS_poly = load_JRS(qpos,qvel)
    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    rotato = get_rotato_from_jrs(JRS_poly,joint_axes)

    EE = {}
    for t in range(n_time_steps):
        for i in range(n_joints):
            EE[(i,t)] = torch.zeros(3,1)
            for j in reversed(range(i+1)):
                EE[(i,t)] = rotato(j,t)@(EE[(i,t)]+P[j])

    return EE


