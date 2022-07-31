

import torch
import zonopy as zp
from zonopy.kinematics import forward_kinematics

qpos =  torch.Tensor([0,0])
qvel =  torch.Tensor([1.2,1.2])
params = {'joint_axes':[torch.Tensor([0,0,1])]*2, 
        'R': [torch.eye(3)]*2,
        'P': [torch.Tensor([0,0,0]), torch.Tensor([1,0,0])],
        'n_joints':2}
_, R_trig = zp.load_JRS_trig(qpos,qvel)
_, R =zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=1,make_gens_independent =True)
n_time_steps = len(R_trig)

P_motor_trig, P_motor = [], []

for t in range(n_time_steps):
    _,P_motor_temp = forward_kinematics(R_trig[t],params)
    P_motor_trig.append(P_motor_temp)
    _,P_motor_temp = forward_kinematics(R[t],params)
    P_motor.append(P_motor_temp)

ax = zp.plot_polyzonos(P_motor_trig,plot_freq=1,edgecolor='red',hold_on=True)

zp.plot_polyzonos(P_motor,plot_freq=1,ax=ax)

