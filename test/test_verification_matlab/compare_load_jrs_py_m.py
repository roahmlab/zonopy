import torch
import os
import zonopy as zp
from scipy.io import loadmat,savemat


def parse_list_Z_py(data,n_joints,n_time_steps):
    z = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            z[(i,t)] = zp.zonotope(torch.tensor(data[str(i)+','+str(t)]))
    return z

def parse_list_Z_m(data,n_joints,n_time_steps):
    z = {}
    for i in range(n_joints):
        for t in range(n_time_steps):
            z[(i,t)] = zp.zonotope(torch.tensor(data[i,0][t,0]))
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
    JRS_py = loadmat(py_path+f'jrs_py_{i}.mat')
    JRS_py = parse_list_Z_py(JRS_py,n_joints,n_time_steps)
    #ax = zp.plot_dict_zono(JRS_py,plot_freq=10,hold_on=True)

    JRS_m = loadmat(m_path+f'jrs_m_{i}.mat')
    JRS_m = JRS_m['jrs_mat']
    JRS_m = parse_list_Z_m(JRS_m,n_joints,n_time_steps)

    #zp.plot_dict_zono(JRS_m,plot_freq=10,edgecolor='red',ax=ax)
    flag = True
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(JRS_py[j,t],JRS_m[j,t]):
                flag = False
    if flag:
        print('matched!')
    else:
        print('unmatched!')


