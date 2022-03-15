import torch
import os
import zonopy as zp
from scipy.io import loadmat,savemat


def parse_list_matZ_py(data,n_joints,n_time_steps):
    z = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            z[(i,t)] = zp.matZonotope(torch.tensor(data[str(i)+','+str(t)]))
    return z

def parse_list_matZ_m(data,n_joints,n_time_steps):
    z = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            z[(i,t)] = zp.matZonotope(torch.tensor(data[i,0][t,0]))
    return z


dirname = os.path.dirname(__file__)
py_path = os.path.join(dirname,'saved_py/')
m_path = os.path.join(dirname,'saved_m/')

key = loadmat(py_path+'test_key.mat')

n_joints = key['n_joints'].item()
N_test = key['N_test'].item()
n_time_steps = key['n_time_steps'].item()



for i in range(N_test):
    print(f'{i+1}-th test')
    rotato_py = loadmat(py_path+f'rotato_py_{i}.mat')
    rotato_py = parse_list_matZ_py(rotato_py,n_joints,n_time_steps)

    rotato_m = loadmat(m_path+f'rotato_m_{i}.mat')
    rotato_m = rotato_m['rotato_mat']
    rotato_m = parse_list_matZ_m(rotato_m,n_joints,n_time_steps)
    
    flag = True
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(rotato_py[j,t],rotato_m[j,t]):
                flag = False            
    if flag:
        print('matched!')
    else:
        print('unmatched!')
