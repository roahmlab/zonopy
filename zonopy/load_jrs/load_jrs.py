
import torch
from .. import zonotope
from mat4py import loadmat

T_plan = 0.5
T_total = 1

def load_JRS(qpos,qvel):
    jrs_path = './jrs_mat_saved/'
    jrs_key = loadmat(jrs_path+'c_kvi.mat')
    jrs_key = torch.tensor(jrs_key['c_kvi'])

    time_idx = 5 # 6-1
    k_idx = 3 # 4-1
    acc_idx = 3 #4-1
    JRS = {}
    for i in range(len(qvel)):
        closest_idx = torch.argmin(abs(qvel[i]-jrs_key))
        jrs_filename = jrs_path+'JRS_mat_'+format(jrs_key[closest_idx],'.3f')+'.mat'
        jrs_mat_load = loadmat(jrs_filename)
        jrs_mat_load = torch.tensor(jrs_mat_load['JRS_mat'])

        n_time_steps = len(jrs_mat_load) # 100


        for jrs_idx in range(n_time_steps):
            c_qpos = torch.cos(qpos[i])
            s_qpos = torch.sin(qpos[i])
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]])
            A = torch.block_diag(Rot_qpos,torch.eye(4))
            JRS_zono_i = zonotope(jrs_mat_load[jrs_idx])
            JRS[(jrs_idx,i)] = A @ slice(JRS_zono_i,5,qvel[i])


        # fail-safe plan
        G = JRS[(0,i)].Z[k_idx,1:]
        k_idx = (G!=0).nonzero()[0]
        if k_idx.numel() != 1:
            if k_idx.numel() == 1:
                raise ValueError('no k-sliceable generator for slice index')
            else:
                raise ValueError('more than one no k-sliceable generators')

        delta_k = G[k_idx]
        c_braking = -qvel[i]/0.5
        delta_breaking = -delta_k/0.5
        for jrs_idx in range(int(n_time_steps/2),n_time_steps):
            Z = JRS[(jrs_idx,i)].Z
            Z[acc_idx,0] = c_braking
            Z[acc_idx,k_idx+1] = delta_breaking
            JRS[jrs_idx,i] = zonotope(Z)

    return JRS