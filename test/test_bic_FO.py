import torch
import zonopy as zp
from zonopy.kinematics.FO import batch_forward_occupancy, forward_occupancy
import time
zp.setup_cuda()
batch_size = 32
N_joints = 7


params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
        'n_joints':N_joints}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints # [


qpos = torch.zeros(batch_size,N_joints)
qvel = torch.pi/2*torch.ones(batch_size,N_joints)

t_start = time.time()
J1, R_trig1 = zp.load_batch_JRS_trig_ic(qpos,qvel)
t =  time.time()
print(t-t_start)
t_start = t


FO_link1,r_trig1, p_trig1 = batch_forward_occupancy(R_trig1,link_zonos,params)
t =  time.time()
print(t-t_start)
t_start = t
import pdb; pdb.set_trace()



