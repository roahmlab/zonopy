import torch
import os
# from mat4py import savemat
from scipy.io import savemat

# 2D multi-link arm

dirname = os.path.dirname(__file__)
config_path = os.path.join(dirname,'random_config/')
try:
    os.mkdir(config_path)
except:
    print('Directory already exist.') 

n_JRS = 401
c_kvi = torch.linspace(-torch.pi,torch.pi,n_JRS)

# key 
config_key = {}
N_test = 50
n_joints = 2
config_key['N_test'] = N_test
config_key['n_joints'] = n_joints
savemat(config_path+f'config_key.mat', config_key)

# random config with a same lenght of links
for i in range(N_test):
    data_config = {}
    data_config['qpos'] = (2*torch.pi*torch.rand(n_joints)-torch.pi).tolist()
    qvel = 2*torch.pi*torch.rand(n_joints)-torch.pi 
    data_config['qvel'] = qvel.tolist()
    j = torch.argmin(abs(c_kvi-qvel.reshape(n_joints,1)),dim=-1)
    delta_kai = torch.maximum(torch.tensor(torch.pi/24),abs(c_kvi[j]/3))
    data_config['qacc'] = (delta_kai*(2*torch.rand(n_joints)-1)).tolist()
    data_config['joint_axes'] = [[0,0,1]]*2
    data_config['P'] = [[0,0,0]] + [[1,0,0]]*(n_joints-1)
    data_config['link_zonos'] = [[[0.5,0.5,0],[0,0,0.01],[0,0,0]]]*2
    savemat(config_path+f'data_config_{i}.mat', data_config)