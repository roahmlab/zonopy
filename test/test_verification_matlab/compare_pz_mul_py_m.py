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

    r_r_l_py = loadmat(py_path+f'r_r_l_py_{i}.mat')
    r_r_l_py = parse_list_Z_py(r_r_l_py,n_joints-1,n_time_steps)
    r_P_py = loadmat(py_path+f'r_P_py_{i}.mat')
    r_P_py = parse_list_Z_py(r_P_py,n_joints,n_time_steps)
    r_l_py = loadmat(py_path+f'r_l_py_{i}.mat')
    r_l_py = parse_list_Z_py(r_l_py,n_joints,n_time_steps)
    #ax = zp.plot_dict_zono(r_l_py,plot_freq=10,hold_on=True)


    r_r_l_m = loadmat(m_path+f'r_r_l_m_{i}.mat')
    r_r_l_m = r_r_l_m['r_r_l_mat']
    r_r_l_m = parse_list_Z_m(r_r_l_m,n_joints-1,n_time_steps)
    r_P_m = loadmat(m_path+f'r_P_m_{i}.mat')
    r_P_m = r_P_m['r_P_mat']
    r_P_m = parse_list_Z_m(r_P_m,n_joints,n_time_steps)
    r_l_m = loadmat(m_path+f'r_l_m_{i}.mat')
    r_l_m = r_l_m['r_l_mat']
    r_l_m = parse_list_Z_m(r_l_m,n_joints,n_time_steps)
    #zp.plot_dict_zono(r_l_m,plot_freq=10,edgecolor='red',ax=ax)

    import pdb; pdb.set_trace()
    flag = True
    for j in range(n_joints-1):
        for t in range(n_time_steps):
            if not zp.close(r_r_l_py[j,t],r_r_l_m[j,t]):
                flag = False            
    if flag:
        print('r_r_l: matched!')
    else:
        print('r_r_l: unmatched!')

    flag = True
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(r_P_py[j,t],r_P_m[j,t]):
                flag = False            
    if flag:
        print('r_P: matched!')
    else:
        print('r_P: unmatched!')
    
    flag = True 
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(r_l_py[j,t],r_l_m[j,t]):
                flag = False
    if flag:
        print('r_l: matched!')
    else: 
        print('r_l: unmatched!')