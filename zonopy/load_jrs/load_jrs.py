import torch
from mat4py import loadmat
data =loadmat('./jrs_mat_saved/JRS_mat_-0.016.mat')

t = 0 # 0 ~ 99
JRS_mat = torch.tensor(data['JRS_mat'][t][0])

def load_JRS(qpos,qvel):
    jrs_path = './jrs_mat_saved/'
    jrs_key = loadmat(jrs_path+'c_kvi.mat')
    jrs_key = torch.tensor(jrs_key['c_kvi'])

    time_dim = 6
    k_dim = 4
    acc_dim = 4
    
    for i in range(len(qvel)):
        closest_idx = torch.argmin(abs(qvel[i]-jrs_key))
        jrs_filename = jrs_path+'JRS_mat_'+format(jrs_key[closest_idx],'.3f')+'.mat'
        jrs_mat_load = loadmat(jrs_filename)
        jrs_mat_load = torch.tensor(jrs_mat_load['JRS_mat'])



        for jrs_idx in range(len(jrs_mat_load)):
            c_qpos = torch.cos(qpos[i])
            s_qpos = torch.sin(qpos[i])
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]])
            A = torch.block_diag(Rot_qpos,torch.eye(4))
        



    return JRS