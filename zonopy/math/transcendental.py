"""Transcendental functions for zonopy.

This module contains transcendental functions for zonopy. Dispatching is done
based on the type of the input. For example, if the input is a zonotope, then
the output will be a zonotope. If the input is a polynomial zonotope, then the
output will be a polynomial zonotope. These functions currently don't consider
precision errors into account.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
import zonopy.internal as zpi
from zonopy import (
    interval,
    zonotope,
    matZonotope,
    polyZonotope,
    matPolyZonotope,
    batchPolyZonotope,
    batchMatPolyZonotope,
    batchMatZonotope,
)

if TYPE_CHECKING:
    from typing import Union
    from zonopy import interval as IType
    from zonopy import zonotope as ZType
    from zonopy import matZonotope as MZType
    from zonopy import polyZonotope as PZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchMatPolyZonotope as BMPZType


SIGN_COS = (-1, -1, 1, 1)
SIGN_SIN = (1, -1, -1, 1)


def sin(zpset: Union[IType, PZType, BPZType],
        order: int = 6,
        ) -> Union[IType, PZType, BPZType]:
    """Sine function

    This function computes the sine function for zonopy continuous set types.
    The order of the Taylor expansion can be specified. The default is 6.

    Args:
        zpset: The set to compute the sine function for.
        order: The order of the Taylor expansion.
    
    Returns:
        The result of the sine function for the set.
    """
    if isinstance(zpset,interval):
        half_pi = torch.pi / 2
        res_inf, res_sup = _int_cos_script(zpset.inf - half_pi, zpset.sup - half_pi)
        return interval(res_inf,res_sup,zpset.dtype,zpset.device)

    elif isinstance(zpset,(polyZonotope,batchPolyZonotope)):
        pz = zpset
        # Make sure we're only using 1D pz's
        assert pz.dimension == 1, "Operation only valid for a 1D PZ"
        pz_c = torch.sin(pz.c)

        out = pz_c

        cs_cf = torch.cos(pz.c)
        sn_cf = pz_c

        factor = 1
        T_factor = 1
        pz_neighbor = pz - pz.c

        for i in range(order):
            factor = factor * (i + 1)
            T_factor = T_factor * pz_neighbor
            if i % 2 == 0:
                out = out + (SIGN_SIN[i%4] * cs_cf / factor) * T_factor
            else:
                out = out + (SIGN_SIN[i%4] * sn_cf / factor) * T_factor

        # add lagrange remainder interval to Grest
        rem = pz_neighbor.to_interval()
        rem_pow = (T_factor * pz_neighbor).to_interval()

        if order % 2 == 1:
            J = sin(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        else:
            J = cos(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        
        if order % 4 == 1 or order % 4 == 2:
            J = -J

        remainder = 1. / (factor * (order + 1)) * rem_pow * J

        # Assumes a 1D pz
        c = out.c + remainder.center()
        G = out.G
        Grest = torch.sum(out.Grest, dim=-2) + remainder.rad()
        Z = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
        if len(Z.shape) > 2:
            out = batchPolyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        else:
            out = polyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        return out
    
        # TODO: test this implementation
        # Not validated, but something like this
        # c_shape = pz_c.shape[:-1]
        # rounded_order = (order + 1) % 2
        # factors = torch.empty(order + rounded_order + 1)
        # factors[0] = 0
        # factors[1:] = torch.arange(1,order+rounded_order+1).cumprod(0)
        # factors = factors.reshape((-1,2,)+(1,)*len(c_shape)).expand((-1,-1,)+c_shape)
        # factors[1::2] *= -1
        # factors[1:,0] = sn_cf/factors[1:,0]
        # factors[:factors.shape[0]-round_order,1] = cs_cf/factors[:,1]
        # factors = factors.flatten(0,1)[1:]
        # for i in range(order):
        #     T_factor = T_factor * pz_neighbor
        #     out += factors[i] * T_factor

    return NotImplementedError


def cos(zpset: Union[IType, PZType, BPZType],
        order: int = 6,
        ) -> Union[IType, PZType, BPZType]:
    """Cosine function

    This function computes the cosine function for zonopy continuous set types.
    The order of the Taylor expansion can be specified. The default is 6.
    
    Args:
        zpset: The set to compute the cosine function for.
        order: The order of the Taylor expansion.
    
    Returns:
        The result of the cosine function for the set.
    """
    if isinstance(zpset,interval):
        res_inf, res_sup = _int_cos_script(zpset.inf, zpset.sup)
        return interval(res_inf,res_sup,zpset.dtype,zpset.device)

    elif isinstance(zpset,(polyZonotope,batchPolyZonotope)):
        pz = zpset
        # Make sure we're only using 1D pz's
        assert pz.dimension == 1, "Operation only valid for a 1D PZ"
        pz_c = torch.cos(pz.c)

        out = pz_c

        cs_cf = pz_c
        sn_cf = torch.sin(pz.c)
            
        factor = 1
        T_factor = 1
        pz_neighbor = pz - pz.c

        for i in range(order):
            factor = factor * (i + 1)
            T_factor = T_factor * pz_neighbor
            if i % 2:
                out = out + (SIGN_COS[i%4] * cs_cf / factor) * T_factor
            else:
                out = out + (SIGN_COS[i%4] * sn_cf / factor) * T_factor

        # add lagrange remainder interval to Grest
        rem = pz_neighbor.to_interval()
        rem_pow = (T_factor * pz_neighbor).to_interval()

        if order % 2 == 0:
            J = sin(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        else:
            J = cos(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        
        if order % 4 == 0 or order % 4 == 1:
            J = -J

        remainder = 1. / (factor * (order + 1)) * rem_pow * J

        # Assumes a 1D pz
        c = out.c + remainder.center()
        G = out.G
        Grest = torch.sum(out.Grest, dim=-2) + remainder.rad()
        Z = torch.cat([c.unsqueeze(-2), G, Grest.unsqueeze(-2)], axis=-2)
        if len(Z.shape) > 2:
            out = batchPolyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        else:
            out = polyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        return out
    
        # TODO: test this implementation
        # Not validated, but something like this
        # c_shape = pz_c.shape[:-1]
        # round_order = order % 2
        # factors = torch.arange(1,order+round_order+1).cumprod(0)
        # factors = factors.reshape((-1,2,)+(1,)*len(c_shape)).expand((-1,-1,)+c_shape)
        # factors[0::2] *= -1
        # factors[:,0] = sn_cf/factors[:,0]
        # factors[:factors.shape[0]-round_order,1] = cs_cf/factors[:,1]
        # factors = factors.flatten(0,1)[:order]
        # for i in range(order):
        #     T_factor = T_factor * pz_neighbor
        #     out += factors[i] * T_factor
    
    return NotImplementedError


def cos_sin_cartprod(
        pz: Union[PZType, BPZType],
        order: int = 6
        ) -> Union[PZType, BPZType]:
    """Cosine and sine cartesian product
    
    This function computes the cartesian product of the cosine and sine
    functions for zonopy continuous set types. The order of the Taylor
    expansion can be specified. The default is 6. Currently only works for
    1D polyZonotopes.
    
    Args:
        pz: The set to compute the cosine and sine cartesian product for.
        order: The order of the Taylor expansion.
        
    Returns:
        The result of the cosine and sine cartesian product for the set.
    """
    # Do both cos and sin at the same time

    cs_cf = torch.cos(pz.c)
    sn_cf = torch.sin(pz.c)
    out_cos = cs_cf
    out_sin = sn_cf

    factor = 1
    T_factor = 1
    pz_neighbor = pz + (-pz.c)
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
        Jcos = cos(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        Jsin = sin(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
    else:
        Jcos = sin(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
        Jsin = cos(pz.c + interval([0], [1], dtype=pz.dtype, device=pz.device) * rem)
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


######################
# INTERNAL FUNCTIONS #
######################

@torch.jit.script
def _int_cos_script(inf, sup):
    """Interval cos function.
    
    This function is used to compute the interval cos function. It is used
    internally by the cos and sin functions.

    Args:
        inf: Lower bound of the interval.
        sup: Upper bound of the interval.

    Returns:
        The lower and upper bounds of the interval cos function.
    """
    # Expand out the interval cos function then jit it.
    # End reduction seems to be slightly faster than inline reduction
    pi_twice = torch.pi * 2
    n = torch.floor(inf / pi_twice)
    lower = inf - n * pi_twice
    upper = sup - n * pi_twice

    # Allocate for full check
    out_low = torch.zeros(((6,) + inf.shape), dtype=inf.dtype, device=inf.device)
    out_high = torch.zeros(((6,) + inf.shape), dtype=inf.dtype, device=inf.device)

    # full period
    not_full_period = upper - lower < pi_twice
    out_high[0] = (~not_full_period).long()
    out_low[0] = -out_high[0]

    # 180 rotated
    rot_180 = lower > torch.pi
    nom = torch.logical_and(not_full_period, ~rot_180)
    nom_180 = torch.logical_and(not_full_period, rot_180)

    # Region 1, upper < 180
    reg1 = upper < torch.pi
    reg1_nom = torch.logical_and(reg1, nom)
    reg1_nom_num = reg1_nom.long()
    out_low[1] = reg1_nom_num * torch.cos(upper)
    out_high[1] = reg1_nom_num * torch.cos(lower)

    # Region 2, 180 <= upper < 360
    reg2 = torch.logical_and(upper < pi_twice, ~reg1)
    reg2_nom = torch.logical_and(reg2, nom)
    reg2_nom_num = reg2_nom.long()
    out_low[3] = -reg2_nom_num
    out_high[3] = reg2_nom_num * torch.cos(torch.minimum(pi_twice-upper, lower))

    # Flip the 180 (flip upper&lower and negate)
    # 180 - 360 (shifted by 180)
    reg2_180 = torch.logical_and(reg2, nom_180)
    reg2_180_num = reg2_180.long()
    out_low[2] = reg2_180_num * torch.cos(lower)
    out_high[2] = reg2_180_num * torch.cos(upper)

    # Flip the 180 (flip upper&lower and negate)
    # we know that nom_180 requires a shift of pi, so reg3_180
    # is 360 - 540 (shifted by 180)
    reg3 = torch.logical_and(upper < pi_twice + torch.pi, ~reg2)
    reg3_180 = torch.logical_and(reg3, nom_180)
    reg3_180_num = reg3_180.long()
    out_low[4] = reg3_180_num * torch.cos(torch.minimum(pi_twice-upper, lower))
    out_high[4] = reg3_180_num
    
    # Region 4, 360 < upper, lower < 180
    reg4 = torch.logical_or(reg1, reg2)
    reg4 = ~torch.logical_or(reg4, reg3_180)
    reg4_num = reg4.long()
    out_low[5] = -reg4_num
    out_high[5] = reg4_num
    
    # Reduce and return
    return out_low.sum(0), out_high.sum(0)