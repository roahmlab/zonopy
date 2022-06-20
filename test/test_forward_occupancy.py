import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import time


# Set cuda if available
zp.setup_cuda()

# Set parameter for forward occupancy compuataion
N_joints = 2
qpos =  torch.tensor([0.0]*N_joints)
qvel =  torch.tensor([torch.pi/2]*N_joints)
params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints

time_step_to_test = 1

t_start = time.time()
_, R_trig = zp.load_JRS_trig(qpos,qvel)
n_time_steps = len(R_trig)
t =  time.time()
print(f'Elasped time for naive load JRS: {t-t_start} sec.')
t_start = t



FO_link_trig, FO_link = [], []
for t in range(time_step_to_test):
    FO_link_temp,_,_ = forward_occupancy(R_trig[t],link_zonos,params)
    FO_link_trig.append(FO_link_temp)
t =  time.time()
print(f'Elasped time for naive FO: {t-t_start} sec.')
t_start = t

'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca()
for i in range(N_joints):
    for t in range(time_step_to_test):
        FO_link_trig[t][i].to_zonotope().plot(ax)


plt.autoscale()
plt.show()
'''