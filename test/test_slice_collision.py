import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
import time
zp.setup_cuda()

N_joints = 2

qpos =  torch.tensor([0.0]*N_joints)
qvel =  torch.tensor([torch.pi/2]*N_joints)

params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints # [1,0,0]

t_start = time.time()
J, R_trig = zp.load_batch_JRS_trig(qpos,qvel)
FO_link,_, _ = forward_occupancy(R_trig,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t

ka = (1*torch.ones(N_joints)).reshape(1,N_joints).repeat(100,1)
#FO_link[0].center_slice_all_dep


# obstacle
obs = zp.zonotope([[-0.5,1.4],[0.2,0],[0,0.2]])

patches = [obs.polygon_patch(edgecolor='blue',facecolor='blue')]
for j in range(N_joints):
    Z = FO_link[j].to_batchZonotope()
    Z_slc = FO_link[j].slice_all_dep(ka)    

    buff1 = Z_slc.project()-obs
    A1,b1 = buff1.polytope() # A: n_timesteps,*,dimension 
    safety1 = torch.max(A1@torch.zeros(2)-b1,dim=-1).values<1e-6
    center = obs.center.reshape(1,1,2).repeat(100,1,1)
    g_obs = obs.generators.unsqueeze(0).repeat(100,1,1)
    obs_Z = obs.Z.unsqueeze(0).repeat(100,1,1)
    A2, b2 = zp.batchZonotope(torch.cat((obs_Z,FO_link[j].Grest[:,:,:2]),-2)).polytope()
    c_k = FO_link[j].center_slice_all_dep(ka)[:,:2]
    safety2 = torch.max((A2@c_k.unsqueeze(-1)).squeeze(-1) - b2,-1).values<1e-6 #

    import pdb;pdb.set_trace() # (safety1 == safety2).all()
    for t in range(100):
        if safety2[t]:
            color = 'red'
            f_color = 'none'
        else:
            color = 'green'
            f_color = 'none'
        patches.append(Z[t].polygon_patch(edgecolor='gray'))
        patches.append(Z_slc[t].polygon_patch(facecolor=f_color, edgecolor=color))


fig = plt.figure()
ax = fig.gca()
ax.add_collection(PatchCollection(patches, match_original=True))
plt.title('slice collision detection')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2.2, 2.2, -2.2, 2.2])
plt.show()

