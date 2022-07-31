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

_, R_trig = zp.load_batch_JRS_trig(qpos,qvel)
FO_link,_, _ = forward_occupancy(R_trig,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t


# obstacle
obs = zp.zonotope([[1.5,0.9],[0.2,0],[0,0.2]])

patches = [obs.polygon_patch(edgecolor='blue',facecolor='blue')]
for j in range(N_joints):
    Z = FO_link[j].to_batchZonotope()
    buff = Z.project()-obs
    A,b = buff.polytope() # A: n_timesteps,*,dimension 
    safety = torch.max(A@torch.zeros(2)-b,dim=-1).values<1e-6
    for t in range(100):
        if safety[t]:
            color = 'red'
            f_color = 'none'
        else:
            f_color = 'none'
            color = 'green'
        patches.append(Z[t].polygon_patch(facecolor=f_color, edgecolor=color))

fig = plt.figure()
ax = fig.gca()
ax.add_collection(PatchCollection(patches, match_original=True))
plt.title('Collision detection')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2.2, 2.2, -2.2, 2.2])
plt.show()