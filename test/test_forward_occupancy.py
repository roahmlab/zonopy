import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy

qpos =  torch.Tensor([0,0])
qvel =  torch.Tensor([torch.pi,torch.pi/2])
params = {'joint_axes':[torch.Tensor([0,0,1])]*2, 
        'P': [torch.Tensor([0,0,0]), torch.Tensor([1,0,0])],
        'n_joints':2}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0],[0,0,0.01],[0,0,0]])).to_polyZonotope()]*2
_, R_trig = zp.load_JRS_trig(qpos,qvel)
_, _, _, _, _, _, _, R_des, _, R, _=zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=3,make_gens_independent =True)
n_time_steps = len(R_trig)


FO_link_trig, FO_link = [], []
for t in range(n_time_steps):
    FO_link_temp,_,_ = forward_occupancy(R_trig[t],link_zonos,params)
    FO_link_trig.append(FO_link_temp)
    FO_link_temp,_,_ = forward_occupancy(R_des[t],link_zonos,params)
    FO_link.append(FO_link_temp)


ax = zp.plot_polyzonos(FO_link_trig,plot_freq=1,edgecolor='blue',hold_on=True)

zp.plot_polyzonos(FO_link,plot_freq=1,ax=ax)
