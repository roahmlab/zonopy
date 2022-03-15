import torch
import os
from scipy.io import loadmat,savemat
import zonopy
from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs
import matplotlib.pyplot as plt

def parse_list_PZ(data):
    return [zonopy.zonotope(torch.tensor(data[i],dtype=torch.float32)).to_polyZonotope() for i in range(len(data))]

def parse_list_torch(data):
    return [torch.tensor(data[i],dtype=torch.float32) for i in range(len(data))]

def process_dict_mat_pz(dict_mat_pz,n_joints,n_time_steps):
    processed_dict = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            processed_dict[str(i)+','+str(t)] = dict_mat_pz[i,t].to_matZonotope().Z.tolist()
    
    return processed_dict

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
    qvel = torch.tensor(data_config['qvel']).reshape(n_joints)
    joint_axes = parse_list_torch(data_config['joint_axes'])
    P = parse_list_torch(data_config['P'])
    link_zonos = parse_list_PZ(data_config['link_zonos'])

    JRS_poly = zonopy.load_JRS(qpos,qvel)
    rotato = get_rotato_from_jrs(JRS_poly,joint_axes)
    max_key = max(JRS_poly.keys())
    n_time_steps = max_key[1]+1

    r_r_l,r_P, r_l, r_r = {},{},{},{}
    for t in range(n_time_steps):
        r_r_l[(0,t)] = rotato[(0,t)]@(rotato[(1,t)]@link_zonos[1])
        for j in range(n_joints):
            r_P[(j,t)] = rotato[(j,t)]@P[j]
            r_l[(j,t)] =rotato[(j,t)]@link_zonos[j]
            r_r[(j,t)] = rotato[(j,t)]@ rotato[(0,t)]

    r_r_l = process_dict_pz(r_r_l,1,n_time_steps)
    r_P = process_dict_pz(r_P,n_joints,n_time_steps)
    r_l = process_dict_pz(r_l,n_joints,n_time_steps)
    r_r = process_dict_mat_pz(r_r,n_joints,n_time_steps)
    
    savemat(save_path+f'r_r_l_py_{i}.mat',r_r_l)
    savemat(save_path+f'r_P_py_{i}.mat',r_P)
    savemat(save_path+f'r_l_py_{i}.mat',r_l)
    savemat(save_path+f'r_r_py_{i}.mat',r_r)
    #import pdb; pdb.set_trace()