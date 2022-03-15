import torch
import os
from scipy.io import loadmat,savemat
import zonopy
import matplotlib.pyplot as plt

def process_dict_pz(dict_pz,n_joints,n_time_steps):
    processed_dict = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            processed_dict[str(i)+','+str(t)] = dict_pz[i,t].to_zonotope().Z.tolist()
    
    return processed_dict


dirname = os.path.dirname(__file__)
config_path = os.path.join(dirname,'random_config/')
save_path = os.path.join(dirname,'saved_py/')
try:
    os.mkdir(save_path)
except:
    print('Directory already exist.') 

key = loadmat(config_path+'config_key.mat')

n_joints = key['n_joints'].item()
N_test = key['N_test'].item()


for i in range(N_test):
    data_config = loadmat(config_path+f'data_config_{i}.mat')
    qpos = torch.tensor(data_config['qpos']).reshape(n_joints)
    qvel = torch.tensor(data_config['qvel']).reshape(n_joints)
    JRS_poly = zonopy.load_JRS(qpos,qvel)

    max_key = max(JRS_poly.keys())
    n_time_steps = max_key[1]+1
    JRS_poly = process_dict_pz(JRS_poly,n_joints,n_time_steps)
    savemat(save_path+f'jrs_py_{i}.mat',JRS_poly)
    #import pdb; pdb.set_trace()