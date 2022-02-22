import torch
cos_dim = 0
sin_dim = 1
vel_dim = 2
acc_dim = 3
k_dim = 3

from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
def get_rotato_from_jrs(JRS_poly,joint_axes):
    '''
    JRS_poly
    joint_axes = <list>
    '''
    assert type(joint_axes) == list

    max_key = max(JRS_poly.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    assert len(joint_axes) == n_joints, f'The number of rotational axes ({len(joint_axes)}) should be the same as the number of joints ({n_joints}).'

    rotato = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            rotato[(i,t)] = gen_rotatotope(JRS_poly[(i,t)],joint_axes[i])

    return rotato

def gen_rotatotope(polyZono,rot_axis):
    '''
    polyZono: <polyZonotope>
    rot_axis: <torch.Tensor>
    '''
    rot_axis = rot_axis.to(dtype=torch.float32)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

    cosq = polyZono.c[cos_dim]
    sinq = polyZono.c[sin_dim]
    # Rodrigues' rotation formula
    C = torch.eye(3) + sinq*w_hat + (1-cosq)*w_hat@w_hat

    cosq = polyZono.G[cos_dim]
    sinq = polyZono.G[sin_dim]
    n_dgens = len(cosq) 
    G = sinq*w_hat.repeat(n_dgens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_dgens,1,1).permute(1,2,0)

    cosq = polyZono.Grest[cos_dim]
    sinq = polyZono.Grest[sin_dim]
    n_igens = len(cosq) 
    Grest = sinq*w_hat.repeat(n_igens,1,1).permute(1,2,0)-cosq*(w_hat@w_hat).repeat(n_igens,1,1).permute(1,2,0)


    # NOTE: delete zero colums?
    return matPolyZonotope(C,G,Grest,polyZono.expMat,polyZono.id)
