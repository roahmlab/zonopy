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
    random_zonotope = loadmat(test_path+f'polytope_test_{i}.mat')
    polytopes = {}
    zono_2d = zp.zonotope(torch.Tensor(random_zonotope['two']))
    zono_3d = zp.zonotope(torch.Tensor(random_zonotope['three']))
    a, b, c = zono_2d.polytope()
    A, B, C = zono_3d.polytope()
    polytopes[f'a'] = a.tolist()
    polytopes[f'b'] = b.tolist()
    polytopes[f'c'] = c.tolist()
    polytopes[f'A'] = A.tolist()
    polytopes[f'B'] = B.tolist()
    polytopes[f'C'] = C.tolist()
    savemat(save_path+f'polytope_py_{i}.mat',polytopes)
