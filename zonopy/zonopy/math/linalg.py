
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
from zonopy.math.utils import compare_permuted_gen, compare_permuted_dep_gen


if TYPE_CHECKING:
    from typing import Union
    from zonopy import interval as IType
    from zonopy import zonotope as ZType
    from zonopy import matZonotope as MZType
    from zonopy import polyZonotope as PZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchMatPolyZonotope as BMPZType


def cross(
        zono1: Union[torch.Tensor, np.ndarray, PZType, BPZType],
        zono2: Union[torch.Tensor, np.ndarray, PZType, BPZType],
        ) -> Union[torch.Tensor, PZType, BPZType]:
    """Returns the cross product of two zonotopes.
    
    Args:
        zono1 (Union[torch.Tensor, np.ndarray, PZType, BPZType]): The first zonotope.
        zono2 (Union[torch.Tensor, np.ndarray, PZType, BPZType]): The second zonotope.
        
    Returns:
        Union[torch.Tensor, MPZType, BMPZType]: The cross product.
    """
    # Handle flipped case as well as torch/np passthrough
    if isinstance(zono2, (torch.Tensor, np.ndarray)):
        assert len(zono2.shape) == 1 and zono2.shape[0] == 3
        if isinstance(zono1, (torch.Tensor, np.ndarray)):
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            return torch.cross(torch.as_tensor(zono1), torch.as_tensor(zono2))
        elif isinstance(zono1, (polyZonotope, batchPolyZonotope)):
            assert zono1.dimension == 3
            return cross(-zono2, zono1)

    # Handle PZ cases
    elif isinstance(zono2, (polyZonotope, batchPolyZonotope)):
        assert zono2.dimension == 3
        if isinstance(zono1, (torch.Tensor, np.ndarray)):
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            zono1_skew_sym = torch.tensor([[0,-zono1[2],zono1[1]],
                                           [zono1[2],0,-zono1[0]],
                                           [-zono1[1],zono1[0],0]], dtype=zono2.dtype, device=zono2.device)
        elif isinstance(zono1, (polyZonotope, batchPolyZonotope)):
            assert zono1.dimension == 3
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
    
    return NotImplementedError


########################
# UNVERIFIED FUNCTIONS #
########################

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


# FIXME: Doesn't work
def dot(zono1,zono2):
    if isinstance(zono1,torch.Tensor):
        if isinstance(zono2,polyZonotope):
            assert len(zono1.shape) == 1 and zono1.shape[0] == zono2.dimension
            zono1 = zono1.to(dtype=zono2.dtype)

            c = (zono1@zono2.c).reshape(1)
            G = (zono1@zono2.G).reshape(1,-1)
            Grest = (zono1@zono2.Grest).reshape(1,-1)
            return polyZonotope(c,G,Grest,zono2.expMat,zono2.id,zono2.dtype,zono2.itype,zono2.device).compress(2)