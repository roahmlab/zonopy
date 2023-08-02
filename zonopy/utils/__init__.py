from zonopy.utils.math import close, cross, dot, sin, cos
from zonopy.utils.plot import *
from zonopy.conSet import PROPERTY_ID
from zonopy.utils.collision import config_safety_check, traj_safety_check, obstacle_collison_free_check
def reset(n_ids = 0, deivce='cpu'):
    PROPERTY_ID._reset(deivce)
    if n_ids > 0:
        PROPERTY_ID.update(n_ids)


