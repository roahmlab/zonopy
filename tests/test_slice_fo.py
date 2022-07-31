import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
import time
zp.setup_cuda()

N_joints = 2
PI = torch.tensor(torch.pi)
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

g_ka = torch.minimum(torch.maximum(PI/24,abs(qvel/3)),PI/3)
lambda_ka = (2*torch.rand(N_joints)-1)
ka = g_ka*lambda_ka

#FO_link[0].center_slice_all_dep


patches = []
for j in range(N_joints):
    Z = FO_link[j].to_batchZonotope()
    Z_slc = FO_link[j].slice_all_dep(lambda_ka.reshape(1,N_joints).repeat(100,1))
    for t in range(100):
        patches.append(Z[t].polygon_patch())
        patches.append(Z_slc[t].polygon_patch(edgecolor='red'))



T_len = 100
t_traj = torch.linspace(0,1,T_len+1)
t_to_peak = t_traj[:int(0.5*T_len)+1]
t_to_brake = t_traj[int(0.5*T_len)+1:] - 0.5
qpos_to_peak = qpos + torch.outer(t_to_peak,qvel) + .5*torch.outer(t_to_peak**2,ka)
qvel_to_peak = qvel + torch.outer(t_to_peak,ka)
qpos_peak = qpos_to_peak[-1]
qvel_peak = qvel_to_peak[-1]
#to stop
bracking_accel = (0 - qvel)/0.5
qpos_to_brake = qpos_peak + torch.outer(t_to_brake,qvel_peak) + .5*torch.outer(t_to_brake**2,bracking_accel)
qvel_to_brake = qvel_peak + torch.outer(t_to_brake,bracking_accel)

q = torch.vstack((qpos_to_peak,qpos_to_brake))
w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
w = (w@torch.tensor([[0.0,0.0,1.0]]*N_joints).T).transpose(0,-1)
q = q.reshape(q.shape+(1,1))
R_q =  torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

for t in range(T_len):
    R, P = torch.eye(3), torch.zeros(3)
    for j in range(N_joints):                
        P = R@params['P'][j] + P
        R = R@params['R'][j]@R_q[t,j]
        link_patch = (R@link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='blue',facecolor='blue')
        patches.append(link_patch)            

fig = plt.figure()
ax = fig.gca()
ax.add_collection(PatchCollection(patches, match_original=True))
plt.title('FO slice')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2.2, 2.2, -2.2, 2.2])
plt.show()

#import pdb;pdb.set_trace()

'''
eps = 1e-6

diff = eps*torch.eye(N_joints)


for j in range(N_joints):
    grad_num = torch.zeros(100,3,N_joints)
    c = FO_link[j].center_slice_all_dep(ka) # n_timesteps, dimension 
    for i in range(N_joints):
        grad_num[:,:,i] = (FO_link[j].center_slice_all_dep(ka+diff[i]) - c)/eps
    grad = FO_link[j].grad_center_slice_all_dep(ka) # c_k: n_timesteps, dimension, n_joints 
    D = (grad - grad_num)
    import pdb; pdb.set_trace()
'''