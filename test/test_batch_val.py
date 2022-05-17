import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import time
zp.setup_cuda()
torch.set_default_dtype(torch.float64)
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

qpos = torch.tensor([[-1.1637,  0.4806,  3.0557, -2.2889,  1.3313,  0.5660, -2.7517]])
qvel = torch.tensor([[-2.5318, -2.7337, -0.6787,  1.9214, -0.4102, -1.9387, -1.6142]])

t_start = time.time()
J1, R_trig1 = zp.load_batch_JRS_trig_ic(qpos,qvel)
FO_link1,r_trig1, p_trig1 = forward_occupancy(R_trig1,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t



FO_link2 = []
r_trig2 = []
p_trig2 = []
t_start = time.time()
for i in range(batch_size):
    J2, R_trig2 = zp.load_batch_JRS_trig(qpos[i],qvel[i])
    FO_link_temp,r_trig_temp, p_trig_temp = forward_occupancy(R_trig2,link_zonos,params)
    FO_link2.append(FO_link_temp)
    r_trig2.append(r_trig_temp)
    p_trig2.append(p_trig_temp)
t =  time.time()
print(t-t_start)
t_start = t



FO_link3 = [[] for _ in range(batch_size)]
r_trig3 = [[] for _ in range(batch_size)]
p_trig3 = [[] for _ in range(batch_size)]

t_start = time.time()
for i in range(batch_size):
    J3, R_trig3 = zp.load_JRS_trig(qpos[i],qvel[i])
    for t in range(100):
        FO_link_temp,r_trig_temp, p_trig_temp = forward_occupancy(R_trig3[t],link_zonos,params)
        FO_link3[i].append(FO_link_temp)
        r_trig3[i].append(r_trig_temp)
        p_trig3[i].append(p_trig_temp)
t =  time.time()
print(t-t_start)
t_start = t

#import pdb;pdb.set_trace()

for i in range(batch_size):
    for j in range(N_joints):
        diff13 = torch.zeros(100)
        diff_p13 = torch.zeros(100)
        diff_r13 = torch.zeros(100)
        diff12 = torch.zeros(100)
        diff_p12 = torch.zeros(100)
        diff_r12 = torch.zeros(100)
        for t in range(100):
            if t != 0 and t!=50:
                #import pdb;pdb.set_trace()
                if not zp.close(FO_link1[j][i,t],FO_link2[i][j][t]):
                    import pdb;pdb.set_trace()
                    print('oh1')
                if not zp.close(FO_link1[j][i,t],FO_link3[i][t][j]):
                    print('oh2')
                ind =(FO_link1[j][i,t].Z).sum(1)!=0
                Z1 = FO_link1[j][i,t].Z[ind]
                ind =(FO_link2[i][j][t].Z).sum(1)!=0
                Z2 = FO_link2[i][j][t].Z[ind]
                ind = (FO_link3[i][t][j].Z).sum(1)!=0
                Z3 = FO_link3[i][t][j].Z[ind]
                diff13[t] = torch.max(abs(Z1 - Z3))
                diff12[t] = torch.max(abs(Z1 - Z2))
                if j !=0:
                    ind =(p_trig1[j][i,t].Z).sum(1)!=0
                    Z1 = p_trig1[j][i,t].Z[ind]
                    ind =(p_trig2[i][j][t].Z).sum(1)!=0
                    Z2 = p_trig2[i][j][t].Z[ind]                    
                    ind = (p_trig3[i][t][j].Z).sum(1)!=0
                    Z3 = p_trig3[i][t][j].Z[ind]
                    diff_p13[t] = torch.max(abs(Z1 - Z3))
                    diff_p12[t] = torch.max(abs(Z1 - Z2))
                ind =(r_trig1[j][i,t].Z).sum((1,2))!=0
                Z1 = r_trig1[j][i,t].Z[ind]
                ind =(r_trig2[i][j][t].Z).sum((1,2))!=0
                Z2 = r_trig2[i][j][t].Z[ind]    
                ind = (r_trig3[i][t][j].Z).sum((1,2))!=0
                Z3 = r_trig3[i][t][j].Z[ind]
                diff_r13[t] = torch.max(abs(Z1 - Z3))
                diff_r12[t] = torch.max(abs(Z1 - Z2))
        if j == 4:
            import pdb; pdb.set_trace()
        print(f'1-3 {i+1}-th batch, {j+1}-th joint P :{diff_p13.max()}')
        print(f'1-3 {i+1}-th batch, {j+1}-th joint R :{diff_r13.max()}')        
        print(f'1-3 {i+1}-th batch, {j+1}-th joint FO :{diff13.max()}')
        print(f'1-2 {i+1}-th batch, {j+1}-th joint P :{diff_p12.max()}')
        print(f'1-2 {i+1}-th batch, {j+1}-th joint R :{diff_r12.max()}')        
        print(f'1-2 {i+1}-th batch, {j+1}-th joint FO :{diff12.max()}')

import pdb;pdb.set_trace()





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
