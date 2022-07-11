import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

def rot(q,joint_axes):
    #joint_axes
    w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
    w = (w@joint_axes.T).transpose(0,-1)
    q = q.reshape(q.shape+(1,1))
    return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

##### Set parameters #####
time_freq = 20
n_joints = 2
qpos =  torch.tensor([0.0]*n_joints)
qvel =  torch.tensor([torch.pi/2]*n_joints)
joint_axes = [torch.tensor([0.0,0.0,1.0]),torch.tensor([0.0,1.0,0.0])]
params = {'joint_axes':joint_axes, 
        'R': [torch.eye(3)]*n_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(n_joints-1),
        'n_joints':n_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0,0],[0.5,0,0],[0,0.01,0],[0,0,0.3]])).to_polyZonotope()]*n_joints 
joint_axes = torch.vstack(params['joint_axes'])


fig = plt.figure()
ax = a3.Axes3D(fig)

##### Compute forward kinematics for current configuration #####
lamb = torch.zeros(n_joints)

T_PLAN, T_FULL = 0.5, 1.0
T_len = 100
t_traj = torch.linspace(0,T_FULL,T_len+1)
t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len):] - T_PLAN

qpos_to_peak = wrap_to_pi(qpos + torch.outer(t_to_peak,qvel) + .5*torch.outer(t_to_peak**2,torch.pi/24*lamb))
qvel_to_peak = qvel + torch.outer(t_to_peak,torch.pi/24*lamb)
bracking_accel = (0 - qvel_to_peak[-1])/(T_FULL - T_PLAN)
qpos_to_brake = wrap_to_pi(qpos_to_peak[-1] + torch.outer(t_to_brake,qvel_to_peak[-1]) + .5*torch.outer(t_to_brake**2,bracking_accel))
qvel_to_brake = qvel_to_peak[-1] + torch.outer(t_to_brake,bracking_accel)

R_q = rot(torch.vstack((qpos_to_peak[:-1],qpos_to_brake[:-1])),joint_axes)
link_patches = []
for t in range(100): 
    R, P = torch.eye(3), torch.zeros(3)
    if t%time_freq == 0:
        for j in range(n_joints):
            P = R@params['P'][j] + P
            R = R@params['R'][j]@R_q[t,j]
            link_patch = (R@link_zonos[j]+P).to_zonotope().polyhedron_patch()
            link_patches.extend(link_patch)            
link_patches = Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.2)
ax.add_collection(link_patches)


##### Compute forward occupancy #####
_, R_trig = zp.load_batch_JRS_trig(qpos,qvel,joint_axes)
FO_link,_, _ = forward_occupancy(R_trig,link_zonos,params)

print('FO')

FO_patches = []
for j in range(n_joints): 
    FO_link_temp = FO_link[j].to_batchZonotope().reduce(4)     
    for t in range(100):        
        if t%time_freq == 0:
            FO_patch = FO_link_temp[t].polyhedron_patch()
            FO_patches.extend(FO_patch)
FO_patches = Poly3DCollection(FO_patches,alpha=0.03,edgecolor='green',facecolor='green',linewidths=0.5)
ax.add_collection3d(FO_patches)  

FO_patches = []
for j in range(n_joints): 
    for t in range(100): 
        FO_link_slc = FO_link[j].slice_all_dep(lamb.unsqueeze(0).repeat(100,1)).reduce(4)        
        if t%time_freq == 0:
            FO_patch = FO_link_slc[t].polyhedron_patch()
            FO_patches.extend(FO_patch)
FO_patches = Poly3DCollection(FO_patches,alpha=0.03,edgecolor='red',facecolor='red',linewidths=0.2)
ax.add_collection3d(FO_patches)  

ax.set_xlim([-1.1*n_joints,1.1*n_joints])
ax.set_ylim([-1.1*n_joints,1.1*n_joints])
ax.set_zlim([-1.1*n_joints,1.1*n_joints])
plt.show()

import pdb;pdb.set_trace()

