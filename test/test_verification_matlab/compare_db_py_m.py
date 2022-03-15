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

    r0_r1_l_py = loadmat(py_path+f'r0_r1_l_py_{i}.mat')
    r0_r1_l_py = parse_list_Z_py(r0_r1_l_py,1,n_time_steps)
    pm1_py = loadmat(py_path+f'pm1_py_{i}.mat')
    pm1_py = parse_list_Z_py(pm1_py,1,n_time_steps)
    fo1_py = loadmat(py_path+f'fo1_py_{i}.mat')
    fo1_py = parse_list_Z_py(fo1_py,1,n_time_steps)
    ax = zp.plot_dict_zono(fo1_py,plot_freq=10,hold_on=True)


    r0_r1_l_m = loadmat(m_path+f'r0_r1_l_m_{i}.mat')
    r0_r1_l_m = r0_r1_l_m['r0_r1_l_mat']
    r0_r1_l_m = parse_list_Z_m(r0_r1_l_m,1,n_time_steps)
    pm1_m = loadmat(m_path+f'pm1_m_{i}.mat')
    pm1_m = pm1_m['pm1_mat']
    pm1_m = parse_list_Z_m(pm1_m,1,n_time_steps)
    fo1_m = loadmat(m_path+f'fo1_m_{i}.mat')
    fo1_m = fo1_m['fo1_mat']
    fo1_m = parse_list_Z_m(fo1_m,1,n_time_steps)
    zp.utils.plot_dict_zono(fo1_m,plot_freq=10,edgecolor='red',ax=ax)


    import pdb; pdb.set_trace()
    '''
    flag = True
    for j in range(1):
        for t in range(n_time_steps):
            if not zp.close(r0_r1_l_py[j,t],r0_r1_l_m[j,t]):
                flag = False            
    if flag:
        print('r0_r1_l: matched!')
    else:
        print('r0_r1_l: unmatched!')

    flag = True
    for j in range(1):
        for t in range(n_time_steps):
            if not zp.close(pm1_py[j,t],pm1_m[j,t]):
                flag = False            
    if flag:
        print('pm1: matched!')
    else:
        print('pm1: unmatched!')
    
    flag = True 
    for j in range(1):
        for t in range(n_time_steps):
            if not zp.close(fo1_py[j,t],fo1_m[j,t]):
                flag = False
    if flag:
        print('fo1: matched!')
    else: 
        print('fo1: unmatched!')
    '''
    FO_link_py = loadmat(py_path+f'fo_py_{i}.mat')
    FO_link_py = parse_list_Z_py(FO_link_py,n_joints,n_time_steps)    


    FO_link_m = loadmat(m_path+f'fo_m_{i}.mat')
    FO_link_m = FO_link_m['FO_link_m']
    FO_link_m = parse_list_Z_m(FO_link_m,n_joints,n_time_steps)

    flag = True 
    for j in range(1):
        for t in range(n_time_steps):
            if not zp.close(fo1_py[j,t],FO_link_py[1,t]):
                flag = False
    if flag:
        print('fo1_py: matched!')
    else: 
        print('fo1_py: unmatched!')

    flag = True 
    for j in range(1):
        for t in range(n_time_steps):
            if not zp.close(FO_link_m[1,t],fo1_m[j,t]):
                flag = False
    if flag:
        print('fo1_m: matched!')
    else: 
        print('fo1_m: unmatched!')