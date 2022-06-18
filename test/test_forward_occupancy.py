import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy

import time
#if torch.cuda.is_available():
#    zp.conSet.DEFAULT_OPTS.set(device='cuda:0')
zp.setup_cuda()
N_joints = 7
qpos =  torch.tensor([0.0]*N_joints)
qvel =  torch.tensor([torch.pi/2]*N_joints)
params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints

t_start = time.time()
_, R_trig = zp.load_JRS_trig(qpos,qvel)
t =  time.time()
print(t-t_start)
t_start = t
#_, R =zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=1,make_gens_independent =True)
n_time_steps = len(R_trig)
t =  time.time()
print(t-t_start)
t_start = t

FO_link_trig, FO_link = [], []

for t in range(n_time_steps):
    FO_link_temp,_,_ = forward_occupancy(R_trig[t],link_zonos,params)
    FO_link_trig.append(FO_link_temp)
    #FO_link_temp,_,_ = forward_occupancy(R[t],link_zonos,params)
    #FO_link.append(FO_link_temp)
t =  time.time()
print(t-t_start)
t_start = t
ax = zp.plot_polyzonos(FO_link_trig,plot_freq=1,edgecolor='blue')#,hold_on=True)

#zp.plot_polyzonos(FO_link,plot_freq=1,ax=ax)
