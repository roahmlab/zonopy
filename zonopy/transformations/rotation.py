"""
Define rotatotope
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""

from __future__ import annotations
import torch
cos_dim = 0
sin_dim = 1
vel_dim = 2
acc_dim = 3
k_dim = 3

from zonopy.conSet.interval.interval import interval
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope
from zonopy.utils.math import cos as zpcos, sin as zpsin
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from typing import Union
    from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope as PZType
    from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType

def gen_batch_rotatotope_from_jrs_trig(bPZ,rot_axis):
    dtype, device = bPZ.dtype, bPZ.device
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)
    cosq = bPZ.c[bPZ.batch_idx_all+(slice(cos_dim,cos_dim+1),)].unsqueeze(-1)
    sinq = bPZ.c[bPZ.batch_idx_all+(slice(sin_dim,sin_dim+1),)].unsqueeze(-1)
    C = torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    cosq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(cos_dim,cos_dim+1))].unsqueeze(-1)
    sinq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(sin_dim,sin_dim+1))].unsqueeze(-1)
    G = sinq*w_hat - cosq*(w_hat@w_hat)
    return batchMatPolyZonotope(torch.cat((C.unsqueeze(-3),G),-3),bPZ.n_dep_gens,bPZ.expMat,bPZ.id,compress=0)

def gen_rotatotope_from_jrs_trig(polyZono,rot_axis):
    '''
    polyZono: <polyZonotope>
    rot_axis: <torch.Tensor>
    '''
    dtype, device = polyZono.dtype, polyZono.device
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)

    cosq = polyZono.c[cos_dim]
    sinq = polyZono.c[sin_dim]
    # Rodrigues' rotation formula
    C = (torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat).unsqueeze(0)
    cosq = polyZono.Z[1:,cos_dim:cos_dim+1].unsqueeze(-1)
    sinq = polyZono.Z[1:,sin_dim:sin_dim+1].unsqueeze(-1)
    G = sinq*w_hat - cosq*(w_hat@w_hat)
    return matPolyZonotope(torch.vstack((C,G)),polyZono.n_dep_gens,polyZono.expMat,polyZono.id,compress=0)


SIGN_COS = (-1, -1, 1, 1)
SIGN_SIN = (1, -1, -1, 1)
def _cos_sin(pz: Union[PZType, BPZType], order: int = 6) -> Union[PZType, BPZType]:
    # Do both cos and sin at the same time
    # cos_q = zp.cos(pz, order=order)
    # sin_q = zp.sin(pz, order=order)
    # return cos_q.exactCartProd(sin_q)

    cs_cf = torch.cos(pz.c)
    sn_cf = torch.sin(pz.c)
    out_cos = cs_cf
    out_sin = sn_cf

    factor = 1
    T_factor = 1
    pz_neighbor = pz - pz.c
    for i in range(order):
        factor = factor * (i + 1)
        T_factor = T_factor * pz_neighbor
        if i % 2:
            out_cos = out_cos + (SIGN_COS[i%4] * cs_cf / factor) * T_factor
            out_sin = out_sin + (SIGN_SIN[i%4] * sn_cf / factor) * T_factor
        else:
            out_cos = out_cos + (SIGN_COS[i%4] * sn_cf / factor) * T_factor
            out_sin = out_sin + (SIGN_SIN[i%4] * cs_cf / factor) * T_factor
    
    # add lagrange remainder interval to Grest
    rem = pz_neighbor.to_interval()
    rem_pow = (T_factor * pz_neighbor).to_interval()
    if order % 2:
        Jcos = zpcos(pz.c + interval([0], [1]) * rem)
        Jsin = zpsin(pz.c + interval([0], [1]) * rem)
    else:
        Jcos = zpsin(pz.c + interval([0], [1]) * rem)
        Jsin = zpcos(pz.c + interval([0], [1]) * rem)
    if order % 4 == 0 or order % 4 == 1:
        Jcos = -Jcos
    if order % 4 == 1 or order % 4 == 2:
        Jsin = -Jsin
    remainder_sin = 1. / (factor * (order + 1)) * rem_pow * Jsin
    remainder_cos = 1. / (factor * (order + 1)) * rem_pow * Jcos

    # Assumes a 1D pz
    c = out_cos.c + remainder_cos.center()
    G = out_cos.G
    Grest = torch.sum(out_cos.Grest, dim=-2) + remainder_cos.rad()
    Zcos = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
    c = out_sin.c + remainder_sin.center()
    G = out_sin.G
    Grest = torch.sum(out_sin.Grest, dim=-2) + remainder_sin.rad()
    Zsin = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
    if isinstance(pz, polyZonotope):
        out_cos = polyZonotope(Zcos, out_cos.n_dep_gens, out_cos.expMat, out_cos.id, compress=0, copy_Z=False)
        out_sin = polyZonotope(Zsin, out_sin.n_dep_gens, out_sin.expMat, out_sin.id, compress=0, copy_Z=False)
    else:
        out_cos = batchPolyZonotope(Zcos, out_cos.n_dep_gens, out_cos.expMat, out_cos.id, compress=0, copy_Z=False)
        out_sin = batchPolyZonotope(Zsin, out_sin.n_dep_gens, out_sin.expMat, out_sin.id, compress=0, copy_Z=False)

    return out_cos.exactCartProd(out_sin)

def get_pz_rotations_from_q(
        q: Union[PZType, BPZType],
        rotation_axis: np.ndarray,
        taylor_deg: int = 6
        ) -> Union[MPZType, BMPZType]:
    cos_sin_q = _cos_sin(q, order=taylor_deg)
    # Skew Symmetric Rotation Matrix for cross prod 
    e = rotation_axis/np.linalg.norm(rotation_axis)
    U = torch.tensor(
        [[0, -e[2], e[1]],
        [e[2], 0, -e[0]],
        [-e[1], e[0], 0]], 
        dtype=torch.get_default_dtype()
        )

    # Create rotation matrices from cos and sin dimensions
    cq = cos_sin_q.c[...,0]
    cq = cq.reshape(*cq.shape, 1, 1)
    sq = cos_sin_q.c[...,1]
    sq = sq.reshape(*sq.shape, 1, 1)
    C = torch.eye(3) + sq*U + (1-cq)*U@U
    C = C.unsqueeze(-3)

    tmp_Grest = torch.empty(0)
    if cos_sin_q.n_indep_gens > 0:
        cq = cos_sin_q.Grest[...,0]
        cq = cq.reshape(*cq.shape, 1, 1)
        sq = cos_sin_q.Grest[...,1]
        sq = sq.reshape(*sq.shape, 1, 1)
        tmp_Grest = sq*U + -cq*U@U

    tmp_G = torch.empty(0)
    if cos_sin_q.n_dep_gens > 0:
        cq = cos_sin_q.G[...,0]
        cq = cq.reshape(*cq.shape, 1, 1)
        sq = cos_sin_q.G[...,1]
        sq = sq.reshape(*sq.shape, 1, 1)
        tmp_G = sq*U + -cq*U@U
    
    Z = torch.concat((C, tmp_G, tmp_Grest), dim=-3)
    if len(Z.shape) == 3:
        out = matPolyZonotope(Z, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, compress=0,copy_Z=False)
    else:
        out = batchMatPolyZonotope(Z, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, compress=2)
    return out


def gen_rot_from_q(q,rot_axis):
    if isinstance(q,(int,float)):
        q = torch.tensor(q,dtype=torch.float)
    dtype, device = q.dtype, q.device
    cosq = torch.cos(q,dtype=dtype,device=device)
    sinq = torch.sin(q,dtype=dtype,device=device)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)
    # Rodrigues' rotation formula
    Rot = torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    return Rot


def gen_rotatotope_from_jrs(*args):
    raise NotImplementedError

if __name__ == '__main__':
    import zonopy as zp
    pz = zp.polyZonotope([0],[[0,torch.pi]],[[torch.pi/2,0]])
    R = gen_rotatotope_from_jrs(pz)