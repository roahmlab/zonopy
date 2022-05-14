import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import time
zp.setup_cuda()

N_joints = 7
qpos =  torch.tensor([0.0]*N_joints)
qvel =  torch.tensor([torch.pi/2]*N_joints)
params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints # [1,0,0]

t_start = time.time()

J1, R_trig1 = zp.load_batch_JRS_trig(qpos,qvel)

t =  time.time()
print(t-t_start)
t_start = t

FO_link1,r_trig1, p_trig1 = forward_occupancy(R_trig1,link_zonos,params)

t =  time.time()
print(t-t_start)
t_start = t

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca()
for i in range(N_joints):
    for t in range(100):
        FO_link1[i][t].to_zonotope().plot(ax)


'''
for i in range(N_joints):
  
    FO_link_temp[i].to_batchZonotope().plot(ax,edgecolor='blue')

plt.autoscale()
plt.show()
import pdb;pdb.set_trace()
'''

t_start = time.time()

J2, R_trig2 = zp.load_JRS_trig(qpos,qvel)

t =  time.time()
print(t-t_start)
t_start = t
#_, R =zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=1,make_gens_independent =True)
n_time_steps = len(R_trig2)
t =  time.time()
print(t-t_start)
t_start = t

FO_link2, r_trig2, p_trig2 = [], [], []

for t in range(1,n_time_steps):
    FO_link_temp,r_temp,p_temp = forward_occupancy(R_trig2[t],link_zonos,params)
    FO_link2.append(FO_link_temp)
    r_trig2.append(r_temp)
    p_trig2.append(p_temp)

t =  time.time()
print(t-t_start)
t_start = t
ax = zp.plot_polyzonos(FO_link2,plot_freq=1,edgecolor='blue',ax=ax)#,hold_on=True)
import pdb;pdb.set_trace()





#'''


