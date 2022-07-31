import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import time
zp.setup_cuda()
batch_size = 1
N_joints = 7


params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints # [

qpos = torch.rand(batch_size,N_joints)*2*torch.pi-torch.pi
qvel = torch.rand(batch_size,N_joints)*2*torch.pi-torch.pi

t_start = time.time()
_, R_trig1 = zp.load_batch_JRS_trig_ic(qpos,qvel)
FO_link1,_, _ = forward_occupancy(R_trig1,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t


FO_link2 = []
t_start = time.time()
for i in range(batch_size):
    _, R_trig2 = zp.load_batch_JRS_trig(qpos[i],qvel[i])
    FO_link_temp,_, _ = forward_occupancy(R_trig2,link_zonos,params)
    FO_link2.append(FO_link_temp)

t =  time.time()
print(t-t_start)
t_start = t

FO_link3 = [[] for _ in range(batch_size)]
t_start = time.time()
for i in range(batch_size):
    _, R_trig3 = zp.load_JRS_trig(qpos[i],qvel[i])
    for t in range(100):
        FO_link_temp,_, _ = forward_occupancy(R_trig3[t],link_zonos,params)
        FO_link3[i].append(FO_link_temp)

t =  time.time()
print(t-t_start)
t_start = t

plot_on = True
if plot_on:
    import matplotlib.pyplot as plt
    for i in range(batch_size):
        fig = plt.figure()
        ax = fig.gca()
        for j in range(N_joints):
            for t in range(100):
                if t%1==0:
                    FO_link1[j][i,t].to_zonotope().plot(ax)
                    FO_link2[i][j][t].to_zonotope().plot(ax,edgecolor='red')
                    FO_link3[i][t][j].to_zonotope().plot(ax,edgecolor='blue')
        plt.autoscale()
        plt.show()


import pdb; pdb.set_trace()

