import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import time
zp.setup_cuda()
batch_size = 1
N_joints = 4


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
FO_link1,r_trig1, p_trig1 = forward_occupancy(R_trig1,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t


FO_link3 = [[] for _ in range(batch_size)]
r_trig3 = [[] for _ in range(batch_size)]
p_trig3 = [[] for _ in range(batch_size)]

t_start = time.time()
for i in range(batch_size):
    _, R_trig3 = zp.load_JRS_trig(qpos[i],qvel[i])
    for t in range(100):
        FO_link_temp,r_trig3_temp, p_trig3_temp = forward_occupancy(R_trig3[t],link_zonos,params)
        FO_link3[i].append(FO_link_temp)
        r_trig3[i].append(r_trig3_temp)
        p_trig3[i].append(p_trig3_temp)
t =  time.time()
print(t-t_start)
t_start = t
for i in range(batch_size):
    for j in range(N_joints):
        diff = torch.zeros(100)
        diff_p = torch.zeros(100)
        diff_r = torch.zeros(100)
        for t in range(100):
            if t != 0 and t!=50:
                #import pdb;pdb.set_trace()
                ind =(FO_link1[j][i,t].Z).sum(1)!=0
                Z1 = FO_link1[j][i,t].Z[ind]
                ind = (FO_link3[i][t][j].Z).sum(1)!=0
                Z2 = FO_link3[i][t][j].Z[ind]
                diff[t] = torch.max(abs(Z1 - Z2))
                if j !=0:
                    ind =(p_trig1[j][i,t].Z).sum(1)!=0
                    Z1 = p_trig1[j][i,t].Z[ind]
                    ind = (p_trig3[i][t][j].Z).sum(1)!=0
                    Z2 = p_trig3[i][t][j].Z[ind]
                    diff_p[t] = torch.max(abs(Z1 - Z2))
                ind =(r_trig1[j][i,t].Z).sum((1,2))!=0
                Z1 = r_trig1[j][i,t].Z[ind]
                ind = (r_trig3[i][t][j].Z).sum((1,2))!=0
                Z2 = r_trig3[i][t][j].Z[ind]
                diff_r[t] = torch.max(abs(Z1 - Z2))
        print(f'{i+1}-th batch, {j+1}-th joint P :{diff_p.max()}')
        print(f'{i+1}-th batch, {j+1}-th joint R :{diff_r.max()}')        
        print(f'{i+1}-th batch, {j+1}-th joint FO :{diff.max()}')

