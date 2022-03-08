import json
from interval import interval
from torch import Tensor
    
def get_robot_params(param_file):
    with open(param_file) as f:
        param = json.load(f)
    return param

def get_interval_params(param_file):
    with open(param_file) as f:
        param = json.load(f)
    mass = Tensor(param['mass'])
    com = Tensor(param['com'])

    lo_mass_scale = 0.97;
    hi_mass_scale = 1.03;

    param['mass'] = interval(lo_mass_scale * mass, hi_mass_scale * mass)
    param['com'] = interval(com.clone(), com.clone())
    param['use_interval'] = True
    return param