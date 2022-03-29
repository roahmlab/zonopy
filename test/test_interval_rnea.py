import torch
from torch import Tensor
from interval import interval, matmul_interval, cross_interval
from get_params import get_robot_params, get_interval_params
from rnea import rnea
import json

param_file = "fetch_arm_param.json"
#param = get_robot_params(param_file)
param = get_interval_params(param_file)

with open('sample_input.json','r') as f:
    data = json.load(f)
    # print(data)

outputs = []
for d in data:
    q = Tensor(d['q'])
    qd = Tensor(d['qd'])
    qdd = Tensor(d['qdd'])
    q_aux_d = Tensor(d['q_aux_d'])

    u = rnea(q, qd, q_aux_d, qdd, True, param)
    inf = u.inf.view(-1).tolist()
    sup = u.sup.view(-1).tolist()

    d = {'inf':inf, 'sup':sup}

    outputs.append(d)

with open('sample_output_python.json', 'w') as f:
    json.dump(outputs, f)


