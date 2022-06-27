import torch
from zonopy import batchZonotope
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import JRS_KEY
from zonopy.transformations.rotation import gen_rotatotope_from_jrs_trig, gen_batch_rotatotope_from_jrs_trig

T_fail_safe = 0.5

cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

def process_batch_JRS_trig(jrs_tensor, q_0,qd_0,joint_axes):
    dtype, device = jrs_tensor.dtype, jrs_tensor.device 
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        JRS_batch_zono = batchZonotope(jrs_tensor[closest_idx])
        c_qpos = torch.cos(q_0[i],dtype=dtype,device=device)
        s_qpos = torch.sin(q_0[i],dtype=dtype,device=device)
        Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=dtype,device=device)
        A = torch.block_diag(Rot_qpos,torch.eye(4,dtype=dtype,device=device))
        JRS_batch_zono = A@JRS_batch_zono.slice(kv_dim,qd_0[i])
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim,prop='k_trig')
        '''
        delta_k = PZ_JRS.G[0,0,ka_dim]
        c_breaking = - qd_0[i]/T_fail_safe
        delta_breaking = - delta_k/T_fail_safe
        PZ_JRS.c[50:,acc_dim] = c_breaking
        PZ_JRS.G[50:,0,acc_dim] = delta_breaking
        '''
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])

        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)
    return PZ_JRS_batch, R_batch

def process_batch_JRS_trig_ic(jrs_tensor,q_0,qd_0,joint_axes):
    dtype, device = jrs_tensor.dtype, jrs_tensor.device 
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[:,i:i+1]-jrs_key),dim=-1)
        JRS_batch_zono = batchZonotope(jrs_tensor[closest_idx])
        c_qpos = torch.cos(q_0[:,i:i+1],dtype=dtype,device=device).unsqueeze(-1)
        s_qpos = torch.sin(q_0[:,i:i+1],dtype=dtype,device=device).unsqueeze(-1)
        A = c_qpos*torch.tensor([[1.0]+[0]*5,[0,1]+[0]*4]+[[0]*6]*4,dtype=dtype,device=device) 
        + s_qpos*torch.tensor([[0,-1]+[0]*4,[1]+[0]*5]+[[0]*6]*4,dtype=dtype,device=device) 
        + torch.tensor([[0.0]*6]*2+[[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=dtype,device=device)
        
        JRS_batch_zono = A.unsqueeze(1)@JRS_batch_zono.slice(kv_dim,qd_0[:,i:i+1].unsqueeze(1).repeat(1,100,1))
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim,prop='k_trig')
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])
        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)

    return PZ_JRS_batch, R_batch

if __name__ == '__main__':
    import zonopy as zp
    import time
    n_test = 1000
    ts = time.time()
    jrs_tensor = zp.preload_batch_JRS_trig()
    print(f'Elasped time for preload: {time.time()-ts}')
    t_load = 0 
    t_process = 0
    qpos = 2*torch.pi*torch.rand(n_test,2)-torch.pi
    qvel = 2*torch.pi*torch.rand(n_test,2)-torch.pi

    t00 = time.time()
    JRS0,_  = process_batch_JRS_trig_ic(jrs_tensor, qpos,qvel)
    print(f'Elasped time for batch process:{time.time()-t00}')
    for i in range(n_test):
        t1 = time.time()
        JRS1,_ = zp.load_batch_JRS_trig(qpos[i],qvel[i])
        t2 = time.time()
        JRS2,_ = process_batch_JRS_trig(jrs_tensor, qpos[i],qvel[i])
        t3 = time.time()
        t_load += t2-t1
        t_process += t3-t2
    

    print(f'Elasped time for load: {t_load}')
    print(f'Elasped time for process: {t_process}')
    print((t_load-t_process)/n_test)