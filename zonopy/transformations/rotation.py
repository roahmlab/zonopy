"""
Define rotatotope
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""

import torch
cos_dim = 0
sin_dim = 1
vel_dim = 2
acc_dim = 3
k_dim = 3

from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.utils.math import cos, sin
from zonopy.conSet import DEFAULT_OPTS


def get_rotato_pair_from_jrs_trig(PZ_JRS_trig,joint_axes,R0=None):
    '''
    PZ_JRS_trig
    PZ_JRS_trig[i]
    joint_axes = <list>
    R0
    R0[i]
    '''
    n_joints = len(PZ_JRS_trig)
    assert len(joint_axes) == n_joints, f'The number of rotational axes ({len(joint_axes)}) should be the same as the number of joints ({n_joints}).'
    assert R0 is None or len(R0) == n_joints
    rotato, rotato_t = [],[]
    for i in range(n_joints):
        R = gen_rotatotope_from_jrs_trig(PZ_JRS_trig[i],joint_axes[i])
        if R0 is None:
            rotato.append(R)
            rotato_t.append(R.T)        
        else:
            rotato.append(R@R0[i])
            rotato_t.append(R0[i].T@R.T)
    return rotato, rotato_t

def get_rotato_from_jrs_trig(PZ_JRS_trig,joint_axes,R0=None):
    '''
    PZ_JRS_trig
    joint_axes = <list>
    '''
    n_joints = len(PZ_JRS_trig)
    assert len(joint_axes) == n_joints, f'The number of rotational axes ({len(joint_axes)}) should be the same as the number of joints ({n_joints}).'
    assert R0 is None or len(R0) == n_joints
    rotato = []
    for i in range(n_joints):
        R = gen_rotatotope_from_jrs_trig(PZ_JRS_trig[i],joint_axes[i])
        if R0 is None:
            rotato.append(R)       
        else:
            rotato.append(R@R0[i])
    return rotato

def gen_rotatotope_from_jrs_trig(polyZono,rot_axis):
    '''
    polyZono: <polyZonotope>
    rot_axis: <torch.Tensor>
    '''
    rot_axis = rot_axis.to(dtype=torch.float32)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],device=polyZono.device)

    cosq = polyZono.c[cos_dim]
    sinq = polyZono.c[sin_dim]
    # Rodrigues' rotation formula
    C = torch.eye(3,device=polyZono.device) + sinq*w_hat + (1-cosq)*w_hat@w_hat

    cosq = polyZono.G[cos_dim]
    sinq = polyZono.G[sin_dim]
    n_dgens = len(cosq) 
    G = sinq*w_hat.repeat(n_dgens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_dgens,1,1).permute(1,2,0)

    cosq = polyZono.Grest[cos_dim]
    sinq = polyZono.Grest[sin_dim]
    n_igens = len(cosq) 
    Grest = sinq*w_hat.repeat(n_igens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_igens,1,1).permute(1,2,0)
    # NOTE: delete zero colums?
    return matPolyZonotope(C,G,Grest,polyZono.expMat,polyZono.id,polyZono.dtype,polyZono.itype,polyZono.device)


def gen_rotatotope_from_jrs(q, rot_axis, deg=6, R0=None):
    cos_q = cos(q,deg)
    sin_q = sin(q,deg)
    cos_sin_q = cos_q.exactCartProd(sin_q)

    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],device=q.device)

    cosq = cos_sin_q.c[cos_dim]
    sinq = cos_sin_q.c[sin_dim]
    # Rodrigues' rotation formula
    C = torch.eye(3,device=q.device) + sinq*w_hat + (1-cosq)*w_hat@w_hat

    
    cosq = cos_sin_q.G[cos_dim]
    sinq = cos_sin_q.G[sin_dim]
    n_dgens = len(cosq) 
    G = sinq*w_hat.repeat(n_dgens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_dgens,1,1).permute(1,2,0)
    
    cosq = cos_sin_q.Grest[cos_dim]
    sinq = cos_sin_q.Grest[sin_dim]
    n_igens = len(cosq) 
    Grest = sinq*w_hat.repeat(n_igens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_igens,1,1).permute(1,2,0)
    return matPolyZonotope(C,G,Grest,cos_sin_q.expMat,cos_sin_q.id,cos_sin_q.dtype,cos_sin_q.itype,cos_sin_q.device)


def gen_rot_from_q(q,rot_axis):
    if isinstance(q,(int,float)):
        q = torch.tensor(q,dtype=torch.float)
        device= DEFAULT_OPTS.DEVICE
    else:
        device = q.device

    cosq = torch.cos(q)
    sinq = torch.sin(q)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],device=device)
    # Rodrigues' rotation formula
    Rot = torch.eye(3,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    return Rot

if __name__ == '__main__':
    import zonopy as zp
    pz = zp.polyZonotope([0],[[0,torch.pi]],[[torch.pi/2,0]])
    R = gen_rotatotope_from_jrs(pz)