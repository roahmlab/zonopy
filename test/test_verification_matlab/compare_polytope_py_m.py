'''
ERROR LESS THAN 1e-5
'''

import torch
import os
import zonopy as zp
from scipy.io import loadmat


dirname = os.path.dirname(__file__)
py_path = os.path.join(dirname,'saved_py/')
m_path = os.path.join(dirname,'saved_m/')

N_test = 50

for i in range(N_test):
    print(f'{i+1}-th test')
    P_py = loadmat(py_path+f'polytope_py_{i}.mat')
    
    #ax = zp.plot_dict_zono(JRS_py,plot_freq=10,hold_on=True)

    P_m = loadmat(m_path+f'polytope_m_{i}.mat')
    
    if torch.linalg.norm(torch.Tensor(P_py['a']-P_m['polytopes']['a'][0,0])) < 1e-5:
        print(f'2d A correct!')
    if torch.linalg.norm(torch.Tensor(P_py['b']-P_m['polytopes']['b'][0,0].T)) < 1e-5:
        print(f'2d B correct!')
    if torch.linalg.norm(torch.Tensor(P_py['c']-P_m['polytopes']['c'][0,0])) < 1e-5:
        print(f'2d C correct!')
    if torch.linalg.norm(torch.Tensor(P_py['A']-P_m['polytopes']['A'][0,0])) < 1e-5:
        print(f'3d A correct!')
    if torch.linalg.norm(torch.Tensor(P_py['B']-P_m['polytopes']['B'][0,0].T)) < 1e-5:
        print(f'3d B correct!')
    if torch.linalg.norm(torch.Tensor(P_py['C']-P_m['polytopes']['C'][0,0])) < 1e-5:
        print(f'3d C correct!')


    import pdb;pdb.set_trace()