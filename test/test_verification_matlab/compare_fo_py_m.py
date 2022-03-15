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
    FO_link_py = loadmat(py_path+f'fo_py_{i}.mat')
    FO_link_py = parse_list_Z_py(FO_link_py,n_joints,n_time_steps)
    P_motor_py = loadmat(py_path+f'pm_py_{i}.mat')
    P_motor_py = parse_list_Z_py(P_motor_py,n_joints,n_time_steps)
    ax = zp.plot_dict_zono(FO_link_py,plot_freq=10,hold_on=True)

    FO_link_m = loadmat(m_path+f'fo_m_{i}.mat')
    FO_link_m = FO_link_m['FO_link_m']
    FO_link_m = parse_list_Z_m(FO_link_m,n_joints,n_time_steps)
    P_motor_m = loadmat(m_path+f'pm_m_{i}.mat')
    P_motor_m = P_motor_m['P_motor_m']
    P_motor_m = parse_list_Z_m(P_motor_m,n_joints,n_time_steps)
    zp.utils.plot_dict_zono(FO_link_m,plot_freq=10,edgecolor='red',ax=ax)
    
    flag = True 
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(P_motor_py[j,t],P_motor_m[j,t]):
                flag = False
    if flag:
        print('P_motor: matched!')
    else: 
        print('P_motor: unmatched!')


    flag = True 
    for j in range(n_joints):
        for t in range(n_time_steps):
            if not zp.close(FO_link_py[j,t],FO_link_m[j,t]):
                flag = False
    if flag:
        print('FO_link: matched!')
    else: 
        print('FO_link: unmatched!')

    import pdb; pdb.set_trace()


