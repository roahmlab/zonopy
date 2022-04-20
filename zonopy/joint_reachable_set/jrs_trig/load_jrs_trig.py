"""
Load trigonometry version of precomtuted joint reacheable set (precomputed by CORA)
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""
import torch
from zonopy.transformations.rotation import gen_rotatotope_from_jrs_trig
from zonopy import zonotope, polyZonotope
from mat4py import loadmat
import os


T_fail_safe = 0.5

dirname = os.path.dirname(__file__)
jrs_path = os.path.join(dirname,'jrs_trig_mat_saved/')
jrs_key = loadmat(jrs_path+'c_kvi.mat')
jrs_key = torch.tensor(jrs_key['c_kvi'],dtype=torch.float)

'''
qjrs_path = os.path.join(dirname,'qjrs_mat_saved/')
qjrs_key = loadmat(qjrs_path+'c_kvi.mat')
qjrs_key = torch.tensor(qjrs_key['c_kvi'])
'''
cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

def load_JRS_trig(q_0,qd_0,joint_axes=None):
    '''
    load joint reachable set precomputed by MATLAB CORA (look gen_jrs).
    Then, operate loaded JRS zonotope into JRS polyzonotope w/ k-sliceable dep. gen. 
    for initial joint pos. and intial joint vel.

    qpos: <torch.Tensor> initial joint position
    , size [N]
    qvel: <torch.Tensor> initial joint velocity
    , size [N]

    return <dict>, <polyZonotope> dictionary of polynomical zonotope
    JRS_poly[t][i]
    - t: t-th timestep \in [0,99] -> 0 ~ 1 sec
    - i: i-th joint
    

    ** dimension index
    0: cos(qpos_i)
    1: sin(qpos_i)
    2: qvel_i
    3: ka_i
    4: kv_i
    5: t

    '''
    jrs_key = jrs_key.to(device=q_0.device)
    if isinstance(q_0,list):
        q_0 = torch.tensor(q_0,dtype=torch.float)
    if isinstance(qd_0,list):
        qd_0 = torch.tensor(qd_0,dtype=torch.float)
    assert len(q_0.shape) == 1 and len(qd_0.shape) == 1 
    n_joints = len(qd_0)
    assert len(q_0) == n_joints
    if joint_axes is None:
        joint_axes = [torch.tensor([0,0,1],dtype=torch.float) for _ in range(n_joints)]
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        jrs_filename = jrs_path+'jrs_trig_mat_'+format(jrs_key[closest_idx],'.3f')+'.mat'
        jrs_mats_load = loadmat(jrs_filename)
        jrs_mats_load = jrs_mats_load['JRS_mat']
        n_time_steps = len(jrs_mats_load) # 100
        if i == 0:
            PZ_JRS = [[] for _ in range(n_time_steps)]
            R = [[] for _ in range(n_time_steps)]            
        for t in range(n_time_steps):
            c_qpos = torch.cos(q_0[i],dtype=torch.float)
            s_qpos = torch.sin(q_0[i],dtype=torch.float)
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=torch.float)
            A = torch.block_diag(Rot_qpos,torch.eye(4))
            jrs_mat_load = torch.tensor(jrs_mats_load[t],dtype=torch.float)[0]
            JRS_zono_i = zonotope(jrs_mat_load)
            JRS_zono_i = A @ JRS_zono_i.slice(kv_dim,qd_0[i])
            PZ_JRS[t].append(JRS_zono_i.deleteZerosGenerators().to_polyZonotope(ka_dim,prop='k_trig'))
            # fail safe
            if t == 0:
                delta_k = PZ_JRS[0][i].G[ka_dim,0]
                c_breaking = - qd_0[i]/T_fail_safe
                delta_breaking = - delta_k/T_fail_safe
            elif t >= int(n_time_steps/2):
                PZ_JRS[t][i].c[acc_dim] = c_breaking
                PZ_JRS[t][i].G[acc_dim] = delta_breaking
            R_temp= gen_rotatotope_from_jrs_trig(PZ_JRS[t][i],joint_axes[i])
            R[t].append(R_temp)
    return PZ_JRS, R

def load_traj_JRS_trig(q_0, qd_0, uniform_bound, Kr, joint_axes = None):
    n_joints = len(q_0)
    if joint_axes is None:
        joint_axes = [torch.tensor([0,0,1],dtype=torch.float) for _ in range(n_joints)]
    PZ_JRS, R = load_JRS_trig(q_0,qd_0,joint_axes)
    n_time_steps = len(PZ_JRS)
    jrs_dim = PZ_JRS[0][0].dimension
    # Error Zonotope
    gain = Kr[0,0].reshape(1,1) # controller gain
    e = uniform_bound/ gain # max pos error
    d = torch.tensor([2*uniform_bound],dtype=torch.float).reshape(1,1) # max vel error

    G_pos_err = torch.zeros(jrs_dim,2)
    G_pos_err[0,0] = (torch.cos(torch.zeros(1))-torch.cos(e))/2
    G_pos_err[1,1] = (torch.sin(e)-torch.sin(-e))/2

    G_vel_err = d

    # create trajectories
    q, qd, qd_a, qdd_a, r, R_t = [[[] for _ in range(n_time_steps)] for _ in range(6)]
    for t in range(n_time_steps):
        for i in range(n_joints):
            JRS = PZ_JRS[t][i]
            Z = JRS.Z
            C, G = Z[:,0], Z[:,1:]

            # desired traj.
            vel_C, vel_G = C[vel_dim].reshape(1), G[vel_dim].reshape(1,-1)
            vel_G = vel_G[:,torch.any(vel_G,axis=0)]

            acc_C, acc_G = C[acc_dim].reshape(1), G[acc_dim].reshape(1,-1)
            acc_G = acc_G[:,torch.any(acc_G,axis=0)]

            # actual traj.
            q[t].append(polyZonotope(C,G[:,0],torch.hstack((G[:,1:],G_pos_err))))
            qd[t].append(polyZonotope(vel_C,torch.hstack((vel_G,G_vel_err))))
            
            # modified trajectories
            qd_a[t].append(polyZonotope(vel_C, torch.hstack((vel_G,gain*e))))
            qdd_a[t].append(polyZonotope(acc_C, torch.hstack((acc_G,gain*d))))
            r[t].append(polyZonotope(torch.zeros(1,dtype=torch.float),torch.hstack((d,gain*e))))
            
            R_t[t].append(R[t][i].T)
    # return q_des, qd_des, qdd_des, q, qd, qd_a, qdd_a, r, c_k, delta_k, id, id_names
    return q, qd, qd_a, qdd_a, R, R_t #, r, c_k, delta_k
#if __name__ == '__main__':
    #aa
