
import torch
from zonopy import zonotope, polyZonotope
from mat4py import loadmat
import os


T_fail_safe = 0.5

dirname = os.path.dirname(__file__)
jrs_path = os.path.join(dirname,'jrs_mat_saved/')
jrs_key = loadmat(jrs_path+'c_kvi.mat')
jrs_key = torch.tensor(jrs_key['c_kvi'])

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

def load_JRS(qpos,qvel):
    '''
    load joint reachable set precomputed by MATLAB CORA (look gen_jrs).
    Then, operate loaded JRS zonotope into JRS polyzonotope w/ k-sliceable dep. gen. 
    for initial joint pos. and intial joint vel.

    qpos: <torch.Tensor> initial joint position
    , size [N]
    qvel: <torch.Tensor> initial joint velocity
    , size [N]

    return <dict>, <polyZonotope> dictionary of polynomical zonotope
    JRS_poly(i,t)
    - i: i-th joint
    - t: t-th timestep \in [0,99] -> 0 ~ 1 sec
    

    ** dimension index
    0: cos(qpos_i)
    1: sin(qpos_i)
    2: qvel_i
    3: ka_i
    4: kv_i
    5: t

    '''
    assert len(qpos.shape) == 1 and len(qvel.shape) == 1 
    assert len(qpos) == len(qvel)

    JRS_poly = {}
    for i in range(len(qvel)):
        closest_idx = torch.argmin(abs(qvel[i]-jrs_key))
        jrs_filename = jrs_path+'JRS_mat_'+format(jrs_key[closest_idx],'.3f')+'.mat'
        jrs_mats_load = loadmat(jrs_filename)
        jrs_mats_load = jrs_mats_load['JRS_mat']
        n_time_steps = len(jrs_mats_load) # 100

        for t in range(n_time_steps):
            c_qpos = torch.cos(qpos[i])
            s_qpos = torch.sin(qpos[i])
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]])
            A = torch.block_diag(Rot_qpos,torch.eye(4))
            jrs_mat_load = torch.tensor(jrs_mats_load[t])[0]
            JRS_zono_i = zonotope(jrs_mat_load)
            JRS_zono_i = A @ JRS_zono_i.slice(kv_dim,qvel[i])
            JRS_poly[(i,t)] = JRS_zono_i.deleteZerosGenerators().to_polyZonotope(ka_dim)
            # NOTE: fail safe
            if t == 0:
                delta_k = JRS_poly[(i,0)].G[ka_dim,0]
                c_breaking = - qvel[i]/T_fail_safe
                delta_breaking = - delta_k/T_fail_safe
            elif t >= int(n_time_steps/2):
                JRS_poly[(i,t)].c[acc_dim] = c_breaking
                JRS_poly[(i,t)].G[acc_dim] = delta_breaking
    return JRS_poly

def load_traj_JRS(q_0, qd_0, uniform_bound, Kr):
    JRS_poly = load_JRS(q_0,qd_0)
    max_key = max(JRS_poly.keys())
    jrs_dim = JRS_poly[(0,0)].dimension
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    # Error Zonotope
    gain = Kr[0,0].reshape(1,1) # controller gain
    e = uniform_bound/ gain # max pos error
    d = torch.Tensor([2*uniform_bound]).reshape(1,1) # max vel error

    G_pos_err = torch.zeros(jrs_dim,2)
    G_pos_err[0,0] = (torch.cos(torch.zeros(1))-torch.cos(e))/2
    G_pos_err[1,1] = (torch.sin(e)-torch.sin(-e))/2

    G_vel_err = d

    # NOTE: update IDs

    # create trajectories
    #q_des, qd_des, qdd_des = {}, {}, {}
    q, qd, qd_a, qdd_a, r = {},{},{},{},{}
    for t in range(n_time_steps):
        for i in range(n_joints):
            JRS = JRS_poly[(i,t)]
            Z = JRS.Z
            C, G = Z[:,0], Z[:,1:]
            # NOTE: update IDs

            # desired traj.
            #q_des[(i,t)] =  JRS

            vel_C, vel_G = C[vel_dim].reshape(1), G[vel_dim].reshape(1,-1)
            vel_G = vel_G[:,torch.any(vel_G,axis=0)]
            #qd_des[(i,t)] = polyZonotope(vel_C,vel_G)

            acc_C, acc_G = C[acc_dim].reshape(1), G[acc_dim].reshape(1,-1)
            acc_G = acc_G[:,torch.any(acc_G,axis=0)]
            #qdd_des[(i,t)] = polyZonotope(acc_C,acc_G)

            # actual traj.
            q[(i,t)] = polyZonotope(C,G[:,0],torch.hstack((G[:,1:],G_pos_err)))
            qd[(i,t)] = polyZonotope(vel_C,torch.hstack((vel_G,G_vel_err)))
            
            # modified trajectories
            qd_a[(i,t)] = polyZonotope(vel_C, torch.hstack((vel_G,gain*e)))
            qdd_a[(i,t)] = polyZonotope(acc_C, torch.hstack((acc_G,gain*d)))
            r[(i,t)] = polyZonotope(torch.zeros(1),torch.hstack((d,gain*e)))
    # return q_des, qd_des, qdd_des, q, qd, qd_a, qdd_a, r, c_k, delta_k, id, id_names
    return q, qd, qd_a, qdd_a #, r, c_k, delta_k
if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    q_0, qd_0, uniform_bound, Kr = torch.tensor([0,1]), torch.tensor([0,0.1]), 1.1, torch.eye(3)
    load_traj_JRS(q_0, qd_0, uniform_bound, Kr)

