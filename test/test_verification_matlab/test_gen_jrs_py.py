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


n_test = 5
random_c = loadmat(test_path+'online_jrs_test.mat')
for i in range(n_test):
    q = torch.Tensor(random_c['q'][:,i])
    dq = torch.Tensor(random_c['dq'][:,i])
    import pdb; pdb.set_trace()
    Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t = zp.gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=True)