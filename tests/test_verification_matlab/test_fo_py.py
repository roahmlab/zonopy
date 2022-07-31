import torch
import os
from scipy.io import loadmat,savemat

import zonopy
from zonopy.forward_kinematics.FO import forward_occupancy_fancy as forward_occupancy
import matplotlib.pyplot as plt

def parse_list_torch(data):
    return [torch.tensor(data[i],dtype=torch.float32) for i in range(len(data))]

def parse_list_PZ(data):
    return [zonopy.zonotope(torch.tensor(data[i],dtype=torch.float32)).to_polyZonotope() for i in range(len(data))]

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
    qacc = torch.tensor(data_config['qacc']).reshape(n_joints)
    joint_axes = parse_list_torch(data_config['joint_axes'])
    P = parse_list_torch(data_config['P'])
    link_zonos = parse_list_PZ(data_config['link_zonos'])
    

    FO_link,_,P_motor = forward_occupancy(qpos,qvel,joint_axes,P,link_zonos)

    max_key = max(FO_link.keys())
    n_time_steps = max_key[1]+1
    
    FO_link = process_dict_pz(FO_link,n_joints,n_time_steps)
    P_motor = process_dict_pz(P_motor,n_joints,n_time_steps)
    #import pdb; pdb.set_trace()
    savemat(save_path+f'fo_py_{i}.mat',FO_link)
    savemat(save_path+f'pm_py_{i}.mat',P_motor)
    #import pdb; pdb.set_trace()

key['n_time_steps'] = n_time_steps
savemat(save_path+'test_key.mat',key)


