import torch
from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope

cos_dim = 0
sin_dim = 1
vel_dim = 2
acc_dim = 3
k_dim = 3

def gen_batch_H_from_jrs_trig(bPZ,rot_axis):
    rot_axis = rot_axis.to(dtype=torch.float32)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1],0],[w[2],0,-w[0],0],[-w[1],w[0],0,0],[0,0,0,0]]) # one at the last
    cosq = bPZ.c[bPZ.batch_idx_all+(slice(cos_dim,cos_dim+1),)].unsqueeze(-1)
    sinq = bPZ.c[bPZ.batch_idx_all+(slice(sin_dim,sin_dim+1),)].unsqueeze(-1)
    C = torch.eye(4) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    cosq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(cos_dim,cos_dim+1))].unsqueeze(-1)
    sinq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(sin_dim,sin_dim+1))].unsqueeze(-1)
    G = sinq*w_hat - cosq*(w_hat@w_hat)
    return batchMatPolyZonotope(torch.cat((C.unsqueeze(-3),G),-3),bPZ.n_dep_gens,bPZ.expMat,bPZ.id)