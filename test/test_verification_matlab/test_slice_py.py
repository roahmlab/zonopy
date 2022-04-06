import torch
import os
from scipy.io import loadmat,savemat

import zonopy as zp

dirname = os.path.dirname(__file__)
test_path = os.path.join(dirname,'test1/')
save_path = os.path.join(dirname,'saved_py/')
try:
    os.mkdir(save_path)
except:
    print('Directory already exist.') 


N_test = 50
for i in range(N_test):
    random_pz = loadmat(test_path+f'slice_test_{i}.mat')
    P = {}
    
    if i == 0:
        PZ = zp.polyZonotope(torch.Tensor(random_pz['c'][0]),torch.Tensor(random_pz['G']),torch.Tensor(random_pz['Grest']),torch.tensor(random_pz['expMat']))
    else:
        PZ = zp.polyZonotope(torch.Tensor(random_pz['c'][0]),torch.Tensor(random_pz['G']),torch.Tensor(random_pz['Grest']),torch.tensor(random_pz['expMat']),torch.tensor(random_pz['id'][0]))
    slice_i_one = torch.Tensor(random_pz['slice_i_one'][0])
    slice_v_one = torch.Tensor(random_pz['slice_v_one'][0])
    slice_i_two = torch.Tensor(random_pz['slice_i_two'][0])
    slice_v_two = torch.Tensor(random_pz['slice_v_two'][0])
    slice_i_three = torch.Tensor(random_pz['slice_i_three'][0])
    slice_v_three = torch.Tensor(random_pz['slice_v_three'][0])
    
    P1 = PZ.slice_dep(slice_i_three[:1],slice_v_three[:1])
    P2 = P1.slice_dep(slice_i_three[1:2],slice_v_three[1:2])
    P3 = P2.slice_dep(slice_i_three[2:],slice_v_three[2:])
    
    PZ1 = PZ.slice_dep(slice_i_one,slice_v_one)
    PZ2 = PZ.slice_dep(slice_i_two,slice_v_two)
    PZ3 = PZ.slice_dep(slice_i_three,slice_v_three)

    P['c'] = PZ1.c.tolist()
    P['g'] = PZ1.G.tolist()
    P['G'] = PZ1.Grest.tolist()
    P['e'] = PZ1.expMat.tolist()
    P['i'] = PZ1.id.tolist()
    P['cc'] = PZ2.c.tolist()
    P['gg'] = PZ2.G.tolist()
    P['GG'] = PZ2.Grest.tolist()
    P['ee'] = PZ2.expMat.tolist()
    P['ii'] = PZ2.id.tolist()
    P['ccc'] = PZ3.c.tolist()
    P['ggg'] = PZ3.G.tolist()
    P['GGG'] = PZ3.Grest.tolist()
    P['eee'] = PZ3.expMat.tolist()
    P['iii'] = PZ3.id.tolist()
    savemat(save_path+f'slice_py_{i}.mat',P)