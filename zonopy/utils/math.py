import torch
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
from zonopy.utils.utils import compare_permuted_gen, compare_permuted_dep_gen

SIGN_COS = (-1, -1, 1, 1)
SIGN_SIN = (1, -1, -1, 1)

# TODO: CHECK
def close(zono1,zono2,eps = 1e-6,match_id=False):
    assert isinstance(zono1, type(zono2)) 
    if isinstance(zono1, zonotope):
        assert zono1.dimension == zono2.dimension
        eps = zono1.dimension**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    elif isinstance(zono1, matZonotope):
        assert zono1.n_rows == zono2.n_rows and zono1.n_cols == zono2.n_cols
        eps = (zono1.n_rows*zono1.n_cols)**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    elif isinstance(zono1,polyZonotope):
        assert zono1.dimension == zono2.dimension
        eps = zono1.dimension**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if match_id:
            if torch.any(torch.sort(zono1.id).values != torch.sort(zono2.id).values):
                return False
        if zono1.n_dep_gens != zono2.n_dep_gens or zono1.n_indep_gens != zono2.n_indep_gens or torch.norm(zono1.c-zono2.c) > eps:
            return False
        if not compare_permuted_gen(zono1.Grest,zono2.Grest,eps):
            return False
        return compare_permuted_dep_gen(zono1.expMat[:,torch.argsort(zono1.id)],zono2.expMat[:,torch.argsort(zono2.id)],zono1.G,zono2.G,eps)
    elif isinstance(zono1,matPolyZonotope):
        assert zono1.n_rows == zono2.n_rows and zono1.n_cols == zono2.n_cols
        eps = (zono1.n_rows*zono1.n_cols)**(0.5)*eps
        zono1, zono2 = zono1.deleteZerosGenerators(eps), zono2.deleteZerosGenerators(eps)
        if match_id:
            if torch.any(torch.sort(zono1.id).values != torch.sort(zono2.id).values):
                return False
        if zono1.n_dep_gens != zono2.n_dep_gens or zono1.n_indep_gens != zono2.n_indep_gens or torch.norm(zono1.c-zono2.c) > eps:
            return False
        if not compare_permuted_gen(zono1.Grest,zono2.Grest,eps):
            return False
        return compare_permuted_dep_gen(zono1.expMat[:,torch.argsort(zono1.id)],zono2.expMat[:,torch.argsort(zono2.id)],zono1.G,zono2.G,eps)
    else:
        print('Other types are not implemented yet.')
    
# TODO: FIXME
def cross(zono1,zono2):
    '''
    
    '''
    if isinstance(zono2,torch.Tensor):
        assert len(zono2.shape) == 1 and zono2.shape[0] == 3
        if isinstance(zono1,torch.Tensor):
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            return torch.cross(zono1,zono2)
        elif isinstance(zono1, (polyZonotope, batchPolyZonotope)):
            assert zono1.dimension ==3
            return cross(-zono2,zono1)

    elif isinstance(zono2, (polyZonotope, batchPolyZonotope)):
        assert zono2.dimension == 3
        if type(zono1) == torch.Tensor:
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            zono1_skew_sym = torch.tensor([[0,-zono1[2],zono1[1]],
                                           [zono1[2],0,-zono1[0]],
                                           [-zono1[1],zono1[0],0]], dtype=zono1.dtype, device=zono1.device)

        elif isinstance(zono1, (polyZonotope, batchPolyZonotope)):
            assert zono1.dimension ==3

            Z = zono1.Z
            Z_skew = torch.zeros(Z.shape + Z.shape[-1:], dtype=Z.dtype, device=Z.device)
            Z_skew[..., 0, 1] = -Z[...,2]
            Z_skew[..., 0, 2] =  Z[...,1]
            Z_skew[..., 1, 0] =  Z[...,2]
            Z_skew[..., 1, 2] = -Z[...,0]
            Z_skew[..., 2, 0] = -Z[...,1]
            Z_skew[..., 2, 1] =  Z[...,0]

            if len(Z_skew.shape) > 3:
                zono1_skew_sym = batchMatPolyZonotope(Z_skew, zono1.n_dep_gens, zono1.expMat, zono1.id, copy_Z=False)
            else:
                zono1_skew_sym = matPolyZonotope(Z_skew, zono1.n_dep_gens, zono1.expMat, zono1.id, copy_Z=False)

        return zono1_skew_sym@zono2    
    

# TODO: FIXME
def dot(zono1,zono2):
    if isinstance(zono1,torch.Tensor):
        if isinstance(zono2,polyZonotope):
            assert len(zono1.shape) == 1 and zono1.shape[0] == zono2.dimension
            zono1 = zono1.to(dtype=zono2.dtype)

            c = (zono1@zono2.c).reshape(1)
            G = (zono1@zono2.G).reshape(1,-1)
            Grest = (zono1@zono2.Grest).reshape(1,-1)
            return polyZonotope(c,G,Grest,zono2.expMat,zono2.id,zono2.dtype,zono2.itype,zono2.device).compress(2)


# TODO: DOCUMENT
@torch.jit.script
def _int_cos_script(inf, sup):
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


# TODO: DOCUMENT
def sin(Set,order=6):
    if isinstance(Set,interval):
        half_pi = torch.pi / 2
        res_inf, res_sup = _int_cos_script(Set.inf - half_pi, Set.sup - half_pi)
        return interval(res_inf,res_sup,Set.dtype,Set.device)

    elif isinstance(Set,(polyZonotope,batchPolyZonotope)):
        pz = Set
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
        if isinstance(pz, polyZonotope):
            out = polyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        else:
            out = batchPolyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        return out

    return NotImplementedError
    
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

# TODO: DOCUMENT
def cos(Set,order = 6):
    if isinstance(Set,interval):
        res_inf, res_sup = _int_cos_script(Set.inf, Set.sup)
        return interval(res_inf,res_sup,Set.dtype,Set.device)

    elif isinstance(Set,(polyZonotope,batchPolyZonotope)):
        pz = Set
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
        if isinstance(pz, polyZonotope):
            out = polyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        else:
            out = batchPolyZonotope(Z, out.n_dep_gens, out.expMat, out.id).compress(2)
        return out
    
    return NotImplementedError

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
        


