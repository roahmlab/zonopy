#import zonopy as zp
import torch
import os
from scipy.io import savemat

# directory for test 1
dirname = os.path.dirname(__file__)
test_path = os.path.join(dirname,'test1/')
try:
    os.mkdir(test_path)
except:
    print('Directory already exist.') 



N_test = 50
random_zonotope = {}

for i in range(N_test):
    a = torch.randint(3,40,(1,))
    b = torch.randint(4,40,(1,)) 
    random_zonotope['two']= (10*torch.rand(2,a)-5).tolist()
    random_zonotope['three']= (10*torch.rand(2,b)-5).tolist()
    savemat(test_path+f'polytope_test_{i}.mat', random_zonotope)

random_pz ={}
for i in range(N_test):
    dim = torch.randint(1,10,(1,))
    n_dep_gens = torch.randint(1,40,(1,))
    n_indep_gens = torch.randint(1,40,(1,))
    random_pz['c'] = (50*torch.rand(dim)-25).tolist()
    random_pz['G'] = (50*torch.rand(dim,n_dep_gens)-25).tolist()
    random_pz['Grest'] = (50*torch.rand(dim,n_indep_gens)-25).tolist()
    random_pz['expMat'] = torch.randint(0,15,(10,n_dep_gens)).tolist()
    random_pz['id'] = list(range(10))
    random_pz['slice_i_one'] = torch.randperm(10)[:1].tolist()
    random_pz['slice_v_one'] = (2*torch.rand(1)-1).tolist()
    random_pz['slice_i_two'] = torch.randperm(10)[:2].tolist()
    random_pz['slice_v_two'] = (2*torch.rand(2)-1).tolist()
    random_pz['slice_i_three'] = torch.randperm(10)[:3].tolist()
    random_pz['slice_v_three'] = (2*torch.rand(3)-1).tolist()

    savemat(test_path+f'slice_test_{i}.mat', random_pz)

n_test = 5

q = (2*torch.pi*torch.rand(2,n_test)-torch.pi).tolist()
dq = (2*torch.pi*torch.rand(2,n_test)-torch.pi).tolist()
random_config = {'q':q,'dq':dq}
savemat(test_path+'online_jrs_test.mat', random_config)