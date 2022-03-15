import torch
import os
from scipy.io import loadmat,savemat
import zonopy
from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs
import matplotlib.pyplot as plt

def parse_list_torch(data):
    return [torch.tensor(data[i],dtype=torch.float32) for i in range(len(data))]

def process_dict_mat_pz(dict_mat_pz,n_joints,n_time_steps):
    processed_dict = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            processed_dict[str(i)+','+str(t)] = dict_mat_pz[i,t].to_matZonotope().Z.tolist()
    
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
    qvel = torch.tensor(data_config['qvel']).reshape(n_joints)
    joint_axes = parse_list_torch(data_config['joint_axes'])

    JRS_poly = zonopy.load_JRS(qpos,qvel)
    rotato = get_rotato_from_jrs(JRS_poly,joint_axes)
    max_key = max(JRS_poly.keys())
    n_time_steps = max_key[1]+1
    rotato = process_dict_mat_pz(rotato,n_joints,n_time_steps)
    savemat(save_path+f'rotato_py_{i}.mat',rotato)
    #import pdb; pdb.set_trace()