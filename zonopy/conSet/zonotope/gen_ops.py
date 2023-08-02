from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING
import zonopy.internal as zpi
import zonopy as zp

if TYPE_CHECKING:
    from typing import Union, Tuple
    from ..polynomial_zonotope.poly_zono import polyZonotope as PZType
    from ..polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from ..polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from ..polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType
    from ..zonotope.zono import zonotope as ZonoType
    from ..zonotope.batch_zono import batchZonotope as BZonoType
    from ..zonotope.mat_zono import matZonotope as MZonoType
    from ..zonotope.batch_mat_zono import batchMatZonotope as BMZonoType

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
        zono: Union[ZonoType, BZonoType],
        num: Union[torch.Tensor, float, int]
        ) -> torch.Tensor:
    assert isinstance(num, (float,int)) or len(num.shape) == 0 \
        or num.shape[-1] == zono.dimension or num.shape[-1] == 1, \
        f'dimension does not match: should be {zono.dimension} or 1, not {num.shape[-1]}.'
    Z = torch.clone(zono.Z)
    Z[...,0,:] += num
    return Z


def _matmul_genmzono_impl(
        mzono1: Union[MZonoType, BMZonoType],
        mzono2: Union[MZonoType, BMZonoType]
        ) -> torch.Tensor:
    assert mzono1.n_cols == mzono2.n_rows

    # Generate new Z matrix
    Z = mzono1.Z.unsqueeze(-3)@mzono2.Z.unsqueeze(-4)
    return Z.flatten(-4,-3)