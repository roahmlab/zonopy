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

    P_ee = {}
    R_ee = {}
    for t in range(n_time_steps):
        for i in range(n_joints):
            P_ee[(i,t)] = polyZonotope(torch.zeros(3,dtype=torch.float32))
            R_ee[(i,t)] = matPolyZonotope(torch.eye(3,dtype=torch.float32))
            for j in reversed(range(i+1)):
                #print(i)
                #print(R_ee[(i,t)])
                #f i == 1 and j==1:
                    #import pdb; pdb.set_trace()
                P_ee[(i,t)] = R_ee[(i,t)]@P[j]+P_ee[(i,t)]
                R_ee[(i,t)] = R_ee[(i,t)]@rotato[(j,t)]

                #P_ee[(i,t)] = rotato[(j,t)]@(P_ee[(i,t)]+P[j])
    return R_ee, P_ee


