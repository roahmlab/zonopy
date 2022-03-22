
import torch
from zonopy import zonotope
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

ka_dim = 3
acc_dim = 3 
kv_dim = 4

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

        for jrs_idx in range(n_time_steps):
            c_qpos = torch.cos(qpos[i])
            s_qpos = torch.sin(qpos[i])
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]])
            A = torch.block_diag(Rot_qpos,torch.eye(4))
            jrs_mat_load = torch.tensor(jrs_mats_load[jrs_idx])[0]
            JRS_zono_i = zonotope(jrs_mat_load)
            JRS_zono_i = A @ JRS_zono_i.slice(kv_dim,qvel[i])
            JRS_poly[(i,jrs_idx)] = JRS_zono_i.to_polyZonotope(ka_dim)
            # NOTE: fail safe
            if jrs_idx == 0:
                delta_k = JRS_poly[(i,0)].G[ka_dim,0]
                c_breaking = - qvel[i]/T_fail_safe
                delta_breaking = - delta_k/T_fail_safe
            elif jrs_idx >= int(n_time_steps/2):
                JRS_poly[(i,jrs_idx)].c[acc_dim] = c_breaking
                JRS_poly[(i,jrs_idx)].G[acc_dim] = delta_breaking
    return JRS_poly

def load_qJRS(qpos,qvel):
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
    0: cos(qpos_i/2)
    1: sin(qpos_i/2)
    2: qvel_i
    3: ka_i
    4: kv_i
    5: t

    '''
    assert len(qpos.shape) == 1 and len(qvel.shape) == 1 
    assert len(qpos) == len(qvel)

    qJRS_poly = {}
    for i in range(len(qvel)):
        closest_idx = torch.argmin(abs(qvel[i]-qjrs_key))
        qjrs_filename = qjrs_path+'JRS_mat_'+format(qjrs_key[closest_idx],'.3f')+'.mat'
        qjrs_mats_load = loadmat(qjrs_filename)
        qjrs_mats_load = qjrs_mats_load['JRS_mat']
        n_time_steps = len(qjrs_mats_load) # 100

        for qjrs_idx in range(n_time_steps):
            c_qpos = torch.cos(qpos[i]/2)
            s_qpos = torch.sin(qpos[i]/2)
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]])
            A = torch.block_diag(Rot_qpos,torch.eye(4))
            qjrs_mat_load = torch.tensor(qjrs_mats_load[qjrs_idx])[0]
            qJRS_zono_i = zonotope(qjrs_mat_load)
            qJRS_zono_i = A @ qJRS_zono_i.slice(kv_dim,qvel[i])
            qJRS_poly[(i,qjrs_idx)] = qJRS_zono_i.to_polyZonotope(ka_dim)
            # NOTE: fail safe
            if qjrs_idx == 0:
                delta_k = qJRS_poly[(i,0)].G[ka_dim,0]
                c_breaking = - qvel[i]/T_fail_safe
                delta_breaking = - delta_k/T_fail_safe
            if qjrs_idx >= int(n_time_steps/2):
                qJRS_poly[(i,qjrs_idx)].c[acc_dim] = c_breaking
                qJRS_poly[(i,qjrs_idx)].G[acc_dim] = delta_breaking
    return qJRS_poly
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    qp = torch.tensor([0])
    qv = torch.tensor([2])
    JRS = load_JRS(qp,qv)
    qJRS = load_qJRS(qp,qv)
    fig = plt.figure()    
    ax = fig.gca() 
    for t in range(100):
        if t == 99:
            qJRS[(0,t)].to_zonotope().plot2d(ax,edgecolor='red',facecolor='none')
            JRS[(0,t)].to_zonotope().plot2d(ax,facecolor='none')

    import zonopy
    from zonopy.forward_kinematics.rotatotope import get_rotato_from_jrs, get_quatato_from_jrs

    #from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
    #from zonopy.conSet.polynomial_zonotope.quat_poly_zono import quatPolyZonotope

    rot_ax = [torch.tensor([0,0,1])]
    rotato = get_rotato_from_jrs(JRS,rot_ax)
    quatato = get_quatato_from_jrs(qJRS,rot_ax)
    link = zonopy.zonotope(torch.tensor([[0.5,0.5,0],[0,0,0.01],[0,0,0]])).to_polyZonotope()

    (quatato[(0,99)]@link).to_zonotope().plot2d(ax,edgecolor='red',facecolor='none')
    (rotato[(0,99)]@link).to_zonotope().plot2d(ax,facecolor='none')
    plt.axis([-1.1,1.1,-1.1,1.1])
    plt.show()
