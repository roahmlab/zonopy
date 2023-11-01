from .linalg import (
    cross,
    # FIXME: Below don't work / are unverified
    dot,
    close,
)
from .transcendental import (
    cos,
    sin,
    cos_sin_cartprod,
)
from .rotation import (
    rot_mat,
    # DEPRECATED
    gen_batch_rotatotope_from_jrs_trig, 
    gen_rotatotope_from_jrs_trig, 
    get_pz_rotations_from_q,
)