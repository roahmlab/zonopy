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

to_generate_ID = zp.polyZonotope(torch.ones(10),torch.eye(10))
N_test = 50
# 32
for i in range(N_test):
    print(f'{i+1}-th test')
    P_py = loadmat(py_path+f'slice_py_{i}.mat')
    
    #ax = zp.plot_dict_zono(JRS_py,plot_freq=10,hold_on=True)

    P_m = loadmat(m_path+f'slice_m_{i}.mat')




    #

    PZ1_py = zp.polyZonotope(torch.Tensor(P_py['c']).reshape(-1),torch.Tensor(P_py['g']),torch.Tensor(P_py['G']),torch.tensor(P_py['e']),torch.Tensor(P_py['i']).reshape(-1))  
    
    PZ2_py = zp.polyZonotope(torch.Tensor(P_py['cc'].reshape(-1)),torch.Tensor(P_py['gg']),torch.Tensor(P_py['GG']),torch.tensor(P_py['ee']),torch.Tensor(P_py['ii']).reshape(-1))
    PZ3_py = zp.polyZonotope(torch.Tensor(P_py['ccc'].reshape(-1)),torch.Tensor(P_py['ggg']),torch.Tensor(P_py['GGG']),torch.tensor(P_py['eee']),torch.Tensor(P_py['iii']).reshape(-1))
    PZ1_m = zp.polyZonotope(torch.Tensor(P_m['P']['c'][0,0].reshape(-1)),torch.Tensor(P_m['P']['g'][0,0]),torch.Tensor(P_m['P']['G'][0,0]),torch.tensor(P_m['P']['e'][0,0],dtype=int),torch.Tensor(P_m['P']['i'][0,0].reshape(-1)))  
    PZ2_m = zp.polyZonotope(torch.Tensor(P_m['P']['cc'][0,0].reshape(-1)),torch.Tensor(P_m['P']['gg'][0,0]),torch.Tensor(P_m['P']['GG'][0,0]),torch.tensor(P_m['P']['ee'][0,0],dtype=int),torch.Tensor(P_m['P']['ii'][0,0].reshape(-1)))  
    PZ3_m = zp.polyZonotope(torch.Tensor(P_m['P']['ccc'][0,0].reshape(-1)),torch.Tensor(P_m['P']['ggg'][0,0]),torch.Tensor(P_m['P']['GGG'][0,0]),torch.tensor(P_m['P']['eee'][0,0],dtype=int),torch.Tensor(P_m['P']['iii'][0,0].reshape(-1)))  
    # 7 th? 13th? 45th sin
    if zp.close(PZ1_py,PZ1_m,1e-5):
        print('1. single slice correct')
    if zp.close(PZ2_py,PZ2_m):
        print('2. double slice correct')
    if zp.close(PZ3_py,PZ3_m):
        print('3. triple slice correct') 
    import pdb; pdb.set_trace()
'''
    if torch.linalg.norm(torch.Tensor(P_py['c']-P_m['P']['c'][0,0].T)) < 1e-6:
        print(f'1. single-slice center correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['g']-P_m['P']['g'][0,0])) < 1e-3:
        print(f'2. single-slice dependent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['G']-P_m['P']['G'][0,0])) < 1e-3:
        print(f'3. single-slice independent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['e']-P_m['P']['e'][0,0])) < 1e-6:
        print(f'4. single-slice exponent matrix correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['i']-P_m['P']['i'][0,0].T)) < 1e-6:
        print(f'5. single-slice ID correct!')
    else:
        import pdb; pdb.set_trace()
    
    if torch.linalg.norm(torch.Tensor(P_py['cc']-P_m['P']['cc'][0,0].T)) < 1e-6:
        print(f'6. double-slice center correct!')
    else:
        import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['gg']-P_m['P']['gg'][0,0])) < 1e-3:
        print(f'7. double-slice dependent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['GG']-P_m['P']['GG'][0,0])) < 1e-3:
        print(f'8. double-slice independent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['ee']-P_m['P']['ee'][0,0])) < 1e-6:
        print(f'9. double-slice exponent matrix correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['ii']-P_m['P']['ii'][0,0].T)) < 1e-6:
        print(f'10. double-slice ID correct!')
    else:
        import pdb; pdb.set_trace()

    if torch.linalg.norm(torch.Tensor(P_py['ccc']-P_m['P']['ccc'][0,0].T)) < 1e-6:
        print(f'11. triple-slice center correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['ggg']-P_m['P']['ggg'][0,0])) < 1e-3:
        print(f'12. triple-slice dependent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['GGG']-P_m['P']['GGG'][0,0])) < 1e-3:
        print(f'13. triple-slice independent generators correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['eee']-P_m['P']['eee'][0,0])) < 1e-6:
        print(f'14. triple-slice exponent matrix correct!')
    else:
        import pdb; pdb.set_trace()
    if torch.linalg.norm(torch.Tensor(P_py['iii']-P_m['P']['iii'][0,0].T)) < 1e-6:
        print(f'15. triple-slice ID correct!')
    else:
        import pdb; pdb.set_trace()
    
    import pdb; pdb.set_trace()

'''    
