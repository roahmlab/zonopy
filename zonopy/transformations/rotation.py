"""
Define rotatotope
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import numpy as np

from zonopy.conSet.interval.interval import interval
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope
from zonopy.utils.math import cos as zpcos, sin as zpsin

if TYPE_CHECKING:
    from typing import Union
    from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope as PZType
    from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType

# Make sure to embed this into the comments!
# cos_dim = 0
# sin_dim = 1
# vel_dim = 2
# acc_dim = 3
# k_dim = 3
# TODO: Document
def gen_batch_rotatotope_from_jrs_trig(
        bPZ: BPZType,
        rot_axis: torch.Tensor
        ) -> BPZType:

    # Adapt common function
    Z = _Ztrig_to_rot(bPZ.Z, rot_axis)
    return batchMatPolyZonotope(Z, bPZ.n_dep_gens, bPZ.expMat, bPZ.id, copy_Z=False)


# TODO: Document Check if compress is needed
def gen_rotatotope_from_jrs_trig(
        polyZono: PZType,
        rot_axis: torch.Tensor
        ) -> PZType:
    '''
    polyZono: <polyZonotope>
    rot_axis: <torch.Tensor>
    '''

    # Adapt common function
    Z = _Ztrig_to_rot(polyZono.Z, rot_axis)
    return matPolyZonotope(Z, polyZono.n_dep_gens, polyZono.expMat, polyZono.id, copy_Z=False).compress(2)


# TODO: DOCUMENT
def get_pz_rotations_from_q(
        q: Union[PZType, BPZType],
        rotation_axis: Union[np.ndarray, torch.Tensor],
        taylor_deg: int = 6
        ) -> Union[MPZType, BMPZType]:
    cos_sin_q = cos_sin_cartProd(q, order=taylor_deg)
    rot_axis_tensor = torch.as_tensor(rotation_axis, device=cos_sin_q.Z.device)
    Z = _Ztrig_to_rot(cos_sin_q.Z, rot_axis_tensor)
    if len(Z.shape) == 3:
        out = matPolyZonotope(Z, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, copy_Z=False).compress(2)
    else:
        out = batchMatPolyZonotope(Z, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, copy_Z=False).compress(2)
    return out


# TODO: CHECK NECESSITY
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


# TODO: REMOVE
def gen_rotatotope_from_jrs(*args):
    raise NotImplementedError


# TODO: DOCUMENT
SIGN_COS = (-1, -1, 1, 1)
SIGN_SIN = (1, -1, -1, 1)
def cos_sin_cartProd(
        pz: Union[PZType, BPZType],
        order: int = 6
        ) -> Union[PZType, BPZType]:
    # Do both cos and sin at the same time

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
    Grest = torch.sum(torch.abs(out_cos.Grest), dim=-2) + remainder_cos.rad()
    Zcos = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
    c = out_sin.c + remainder_sin.center()
    G = out_sin.G
    Grest = torch.sum(torch.abs(out_sin.Grest), dim=-2) + remainder_sin.rad()
    Zsin = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
    if isinstance(pz, polyZonotope):
        out_cos = polyZonotope(Zcos, out_cos.n_dep_gens, out_cos.expMat, out_cos.id, copy_Z=False)
        out_sin = polyZonotope(Zsin, out_sin.n_dep_gens, out_sin.expMat, out_sin.id, copy_Z=False)
    else:
        out_cos = batchPolyZonotope(Zcos, out_cos.n_dep_gens, out_cos.expMat, out_cos.id, copy_Z=False)
        out_sin = batchPolyZonotope(Zsin, out_sin.n_dep_gens, out_sin.expMat, out_sin.id, copy_Z=False)

    return out_cos.exactCartProd(out_sin)


# TODO: Evaluate scripting performance on GPU
# note, performance difference is negligible on CPU
# @torch.jit.script
def _Ztrig_to_rot(
        pz_trig_Z: torch.Tensor,
        rot_axis: torch.Tensor
        ) -> torch.Tensor:
    # Get the skew symmetric matrix for the cross product
    e = rot_axis/torch.norm(rot_axis)
    U = torch.tensor(
        [[0, -e[2], e[1]],
        [e[2], 0, -e[0]],
        [-e[1], e[0], 0]],
        dtype=pz_trig_Z.dtype,
        device=pz_trig_Z.device)
    # Swap with bottom for scripted version
    # U = torch.zeros((3,3), dtype=pz_trig_Z.dtype, device=pz_trig_Z.device)
    # U[0, 1] = -e[2]
    # U[0, 2] = e[1]
    # U[1, 0] = e[2]
    # U[1, 2] = -e[0]
    # U[2, 0] = -e[1]
    # U[2, 1] = e[0]
    
    # Preallocate
    Z = torch.empty(pz_trig_Z.shape[:-1] + (3,3,), dtype=pz_trig_Z.dtype, device=pz_trig_Z.device)

    # Compute for C and use broadcasting
    cq = pz_trig_Z[..., 0, 0, None, None]
    sq = pz_trig_Z[..., 0, 1, None, None]
    Z[...,0,:,:] = torch.eye(3) + sq*U + (1-cq)*U@U

    # Compute for G & use broadcasting
    cq = pz_trig_Z[..., 1:, 0, None, None]
    sq = pz_trig_Z[..., 1:, 1, None, None]
    Z[...,1:,:,:] = sq*U + -cq*U@U
    return Z

if __name__ == '__main__':
    import zonopy as zp
    pz = zp.polyZonotope([0],[[0,torch.pi]],[[torch.pi/2,0]])
    R = gen_rotatotope_from_jrs(pz)