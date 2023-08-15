# TODO DOCUMENT

from __future__ import annotations
from .polynomial_zonotope.utils import mergeExpMatrix
import torch
import numpy as np
from typing import TYPE_CHECKING
import zonopy.internal as zpi
import zonopy as zp

if TYPE_CHECKING:
    from typing import Union, Tuple
    from .polynomial_zonotope.poly_zono import polyZonotope as PZType
    from .polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from .polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from .polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType
    from .zonotope.zono import zonotope as ZonoType
    from .zonotope.batch_zono import batchZonotope as BZonoType
    from .zonotope.mat_zono import matZonotope as MZonoType
    from .zonotope.batch_mat_zono import batchMatZonotope as BMZonoType


# exact Plus
def _add_genzono_impl(
        zono1: Union[ZonoType, BZonoType],
        zono2: Union[ZonoType, BZonoType],
        batch_shape: Tuple = ()
        ) -> torch.Tensor:
    assert zono1.dimension == zono2.dimension, f'zonotope dimension does not match: {zono1.dimension} and {zono2.dimension}.'

    expand_shape = batch_shape+(-1, -1)
    Zlist = (
        (zono1.center+zono2.center).unsqueeze(-2),
        zono1.generators.expand(expand_shape),
        zono2.generators.expand(expand_shape),
    )
    Z = torch.cat(Zlist, dim=-2)
    return Z


def _add_genzono_num_impl(
        zono: Union[ZonoType, BZonoType, PZType, BPZType],
        num: Union[torch.Tensor, float, int]
        ) -> torch.Tensor:
    assert isinstance(num, (float,int)) or len(num.shape) == 0 \
        or num.shape[-1] == zono.dimension or num.shape[-1] == 1, \
        f'dimension does not match: should be {zono.dimension} or 1, not {num.shape[-1]}.'
    Z = torch.clone(zono.Z)
    Z[...,0,:] += num
    return Z

# TODO discard, exists for compat
def _add_genpz_num_impl(
        pz: Union[PZType, BPZType],
        num: Union[torch.Tensor, float, int]
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    Z = _add_genzono_num_impl(pz, num)
    return Z, pz.n_dep_gens, pz.expMat, pz.id


def _mul_genzono_num_impl(
        zono: Union[ZonoType, BZonoType, PZType, BPZType],
        num: Union[torch.Tensor, float, int],
        batch_shape: Tuple = None
        ) -> torch.Tensor:
    assert isinstance(num, (float,int)) or len(num.shape) == 0 \
        or zono.dimension == num.shape[0] or zono.dimension == 1 \
        or len(num.shape) == 1, 'Invalid dimension.'
    if batch_shape is not None \
            and isinstance(num, torch.Tensor) \
            and num.shape[:len(batch_shape)] == batch_shape:
        num = num.unsqueeze(1)
    Z = zono.Z * num
    return Z

# TODO discard, exists for compat
def _mul_genpz_num_impl(
        pz: Union[PZType, BPZType],
        num: Union[torch.Tensor, float, int],
        batch_shape: Tuple = None
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    Z = _mul_genzono_num_impl(pz, num, batch_shape=batch_shape)
    return Z, pz.n_dep_gens, pz.expMat, pz.id


def _matmul_genmzono_impl(
        mzono1: Union[MZonoType, BMZonoType],
        mzono2: Union[MZonoType, BMZonoType]
        ) -> torch.Tensor:
    assert mzono1.n_cols == mzono2.n_rows

    # Generate new Z matrix
    Z = mzono1.Z.unsqueeze(-3)@mzono2.Z.unsqueeze(-4)
    return Z.flatten(-4,-3)

# exact Plus
def _add_genpz_impl(
        pz1: Union[PZType, BPZType],
        pz2: Union[PZType, BPZType],
        batch_shape: Tuple = ()
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    id, expMat1, expMat2 = mergeExpMatrix(pz1.id, pz2.id, pz1.expMat, pz2.expMat)
    expMat = torch.vstack((expMat1,expMat2))
    n_dep_gens = pz1.n_dep_gens + pz2.n_dep_gens

    expand_shape = batch_shape+(-1, -1)
    Zlist = (
        (pz1.c+pz2.c).unsqueeze(-2),
        pz1.G.expand(expand_shape),
        pz2.G.expand(expand_shape),
        pz1.Grest.expand(expand_shape),
        pz2.Grest.expand(expand_shape)
        )
    Z = torch.cat(Zlist, dim=-2)
    return Z, n_dep_gens, expMat, id


def _add_genpz_zono_impl(
        pz: Union[PZType, BPZType],
        zono: Union[ZonoType, BZonoType],
        batch_shape: Tuple = ()
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    expand_shape = batch_shape+(-1, -1)
    Zlist = (
        (pz.c+zono.center).unsqueeze(-2),
        pz.G.expand(expand_shape),
        pz.Grest.expand(expand_shape),
        zono.generators.expand(expand_shape)
        )
    Z = torch.cat(Zlist, dim=-2)
    return Z, pz.n_dep_gens, pz.expMat, pz.id


@torch.jit.script
def __mul_Z_tensormerge(Z1: torch.Tensor, Z2: torch.Tensor, z1_ndep: int, z2_ndep: int) -> torch.Tensor:
    # _Z = Z1.unsqueeze(-2)*Z2.unsqueeze(-3)
    _Z = torch.einsum("...id, ...jd->...ijd",Z1,Z2)
    z1 = _Z[..., :z1_ndep+1, 0, :]
    z2 = _Z[..., :z1_ndep+1, 1:z2_ndep+1, :].flatten(-3,-2) # COPIES
    z3 = _Z[..., z1_ndep+1:, :, :].flatten(-3,-2) # COPIES
    z4 = _Z[..., :z1_ndep+1, z2_ndep+1:, :].flatten(-3,-2) # COPIES
    # One way to improve this is to create the output tensor and use views to save to these components directly
    # In that case, the very slight torchscript benefit here probably wouldn't matter
    Z = torch.cat((z1,z2,z3,z4),dim=-2)
    return Z


def _mul_genpz_impl(
        pz1: Union[PZType, BPZType],
        pz2: Union[PZType, BPZType]
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    assert pz1.dimension == pz2.dimension, 'Both polynomial zonotope must have same dimension!'

    # Generate the expMat for the overlapping parts
    id, expMat1, expMat2 = mergeExpMatrix(pz1.id, pz2.id, pz1.expMat, pz2.expMat)
    first = expMat2.expand((pz1.n_dep_gens,-1,-1)).reshape(pz1.n_dep_gens*expMat2.shape[0],expMat2.shape[1])
    second = expMat1.expand((pz2.n_dep_gens,-1,-1)).transpose(0,1).reshape(pz2.n_dep_gens*expMat1.shape[0],expMat1.shape[1])
    expMat = torch.vstack((expMat1,expMat2,first + second))
    n_dep_gens = (pz1.n_dep_gens+1) * (pz2.n_dep_gens+1)-1 
    
    # If batch_dim are not equal, this fails!
    # Generate new Z matrix
    Z = __mul_Z_tensormerge(pz1.Z, pz2.Z, pz1.n_dep_gens, pz2.n_dep_gens)
    return Z, n_dep_gens, expMat, id


# NOTE: this is 'OVERAPPROXIMATED' multiplication for keeping 'fully-k-sliceables'
# The actual multiplication should take
# dep. gnes.: C_G, G_c, G_G, Grest_Grest, G_Grest, Grest_G
# indep. gens.: C_Grest, Grest_c
#
# But, the sliceable multiplication takes
# dep. gnes.: C_G, G_c, G_G (fully-k-sliceable)
# indep. gnes.: C_Grest, Grest_c, Grest_Grest
#               G_Grest, Grest_G (partially-k-sliceable)


@torch.jit.script
def __matmul_Z_tensormerge(Z1: torch.Tensor, Z2: torch.Tensor, z1_ndep: int, z2_ndep: int) -> torch.Tensor:
    _Z = Z1.unsqueeze(-3) @ Z2.unsqueeze(-4)
    z1 = _Z[..., :z1_ndep+1, 0, :, :]
    z2 = _Z[..., :z1_ndep+1, 1:z2_ndep+1, :, :].flatten(-4,-3) # COPIES
    z3 = _Z[..., z1_ndep+1:, :, :, :].flatten(-4,-3) # COPIES
    z4 = _Z[..., :z1_ndep+1, z2_ndep+1:, :, :].flatten(-4,-3) # COPIES
    # One way to improve this is to create the output tensor and use views to save to these components directly
    # In that case, the very slight torchscript benefit here probably wouldn't matter
    Z = torch.cat((z1,z2,z3,z4),dim=-3)
    return Z

# NOTE: DUE TO THE OVERAPPROXIMATED NATURE, BOTH INPUTS MUST BE COMPRESSED ALREADY
def _matmul_genmpz_impl(
        mpz1: Union[MPZType, BMPZType],
        mpz2: Union[MPZType, BMPZType]
        ) -> Tuple[torch.Tensor, int, torch.Tensor, np.ndarray]:
    assert mpz1.n_cols == mpz2.n_rows

    # Generate the expMat for the overlapping parts
    id, expMat1, expMat2 = mergeExpMatrix(mpz1.id,mpz2.id,mpz1.expMat,mpz2.expMat)
    first = expMat2.expand((mpz1.n_dep_gens,)+expMat2.shape).reshape(mpz1.n_dep_gens*expMat2.shape[0],expMat2.shape[1])
    second = expMat1.expand((mpz2.n_dep_gens,)+expMat1.shape).transpose(0,1).reshape(mpz2.n_dep_gens*expMat1.shape[0],expMat1.shape[1])
    expMat = torch.vstack((expMat1,expMat2,first + second))
    n_dep_gens = (mpz1.n_dep_gens+1) * (mpz2.n_dep_gens+1)-1 

    # Generate new Z matrix
    Z = __matmul_Z_tensormerge(mpz1.Z, mpz2.Z, mpz1.n_dep_gens, mpz2.n_dep_gens)

    # Generate the expMat for the overlapping parts
    return Z, n_dep_gens, expMat, id