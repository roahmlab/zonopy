"""This module contains functions for computing rotations of zonopy types.

Currently, only polyZonotopes are supported.
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
from .transcendental import cos_sin_cartprod

if TYPE_CHECKING:
    from typing import Union
    from zonopy import interval as IType
    from zonopy import zonotope as ZType
    from zonopy import matZonotope as MZType
    from zonopy import polyZonotope as PZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchMatPolyZonotope as BMPZType


def rot_mat(
        input: Union[PZType, BPZType],
        rot_axis: Union[np.ndarray, torch.Tensor],
        taylor_deg: int = 6
        ) -> Union[MPZType, BMPZType]:
    """Returns the rotation matrix from a polyZonotope holding angles or the trigonometric form.

    Args:
        input (Union[PZType, BPZType]): The polyZonotope holding angles or the trigonometric form.
        rot_axis (Union[np.ndarray, torch.Tensor]): The rotation axis.
        taylor_deg (int, optional): The degree of the Taylor approximation if angles. Defaults to 6.
    
    Returns:
        Union[MPZType, BMPZType]: The rotation matrix.
    """
    if input.dimension == 1:
        input = cos_sin_cartprod(input, order=taylor_deg)
    rot_axis_tensor = torch.as_tensor(rot_axis, device=input.Z.device)
    Z = _Ztrig_to_rot(input.Z, rot_axis_tensor)
    if len(Z.shape) == 3:
        return matPolyZonotope(Z, input.n_dep_gens, input.expMat, input.id, copy_Z=False)
    else:
        return batchMatPolyZonotope(Z, input.n_dep_gens, input.expMat, input.id, copy_Z=False)


######################
# INTERNAL FUNCTIONS #
######################

# TODO: Evaluate scripting performance on GPU
# note, performance difference is negligible on CPU
# @torch.jit.script
def _Ztrig_to_rot(
        pz_trig_Z: torch.Tensor,
        rot_axis: torch.Tensor
        ) -> torch.Tensor:
    """Converts a polyZonotope Z matrix in trigonometric form to a rotation matrix.
    
    Args:
        pz_trig_Z (torch.Tensor): The polyZonotope Z matrix in trigonometric form.
        rot_axis (torch.Tensor): The rotation axis.
    
    Returns:
        torch.Tensor: The rotation matrix.
    """
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
    Z[...,0,:,:] = torch.eye(3, dtype=pz_trig_Z.dtype, device=pz_trig_Z.device) + sq*U + (1-cq)*U@U

    # Compute for G & use broadcasting
    cq = pz_trig_Z[..., 1:, 0, None, None]
    sq = pz_trig_Z[..., 1:, 1, None, None]
    Z[...,1:,:,:] = sq*U + -cq*U@U
    return Z


##############
# DEPRECATED #
##############

def gen_batch_rotatotope_from_jrs_trig(
        bPZ: BPZType,
        rot_axis: torch.Tensor
        ) -> BPZType:
    """Deprecated, see :func:`rot_mat`"""
    if not zpi.__hide_deprecation__:
        import warnings
        warnings.warn(
            "gen_batch_rotatotope_from_jrs_trig is deprecated, use rot_mat instead",
            DeprecationWarning)
    return rot_mat(bPZ, rot_axis)


def gen_rotatotope_from_jrs_trig(
        polyZono: PZType,
        rot_axis: torch.Tensor
        ) -> PZType:
    """Deprecated, see :func:`rot_mat`"""
    if not zpi.__hide_deprecation__:
        import warnings
        warnings.warn(
            "gen_rotatotope_from_jrs_trig is deprecated, use rot_mat instead",
            DeprecationWarning)
    return rot_mat(polyZono, rot_axis)


def get_pz_rotations_from_q(
        q: Union[PZType, BPZType],
        rotation_axis: Union[np.ndarray, torch.Tensor],
        taylor_deg: int = 6
        ) -> Union[MPZType, BMPZType]:
    """Deprecated, see :func:`rot_mat`"""
    if not zpi.__hide_deprecation__:
        import warnings
        warnings.warn(
            "get_pz_rotations_from_q is deprecated, use rot_mat instead",
            DeprecationWarning)
    return rot_mat(q, rotation_axis, taylor_deg)