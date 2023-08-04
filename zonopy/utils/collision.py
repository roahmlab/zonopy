import zonopy as zp
import torch

P_zero = torch.zeros(3,dtype=torch.float32)
R_identity = torch.eye(3,dtype=torch.float32)

def obstacle_collison_free_check(qpos,robot_params,obs_zono,R0=None,P0=None):
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    link_zonos = robot_params['link_zonos']
    joint_axes = robot_params['joint_axes']
    if isinstance(R0,list):
        R_motor = torch.Tensor(R0)
    elif isinstance(R0,type(None)):
        R_motor = R_identity
    if isinstance(P0,list):
        P_motor = torch.Tensor(P0)
    elif isinstance(P0,type(None)):
        P_motor = P_zero
    for i in range(n_joints):
        Rot = zp.transformations.rotation.gen_rot_from_q(qpos[i],joint_axes[i])
        P_motor = R_motor@P[i] + P_motor
        R_motor = R_motor@R[i]@Rot
        FO_link = R_motor@link_zonos[i] + P_motor
        buff = FO_link - obs_zono
        A,b,_ = buff.polytope()
        if max(A@torch.zeros(3)-b)<1e-6:
            return False    
    return True

def config_safety_check(qpos,robot_params,obs_zonos, R0=None, P0=None):
    '''
    True: safe, False: unsafe
    '''
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    link_zonos = robot_params['link_zonos']
    joint_axes = robot_params['joint_axes']
    if isinstance(R0,list):
        R_motor = torch.Tensor(R0)
    elif isinstance(R0,type(None)):
        R_motor = R_identity
    if isinstance(P0,list):
        P_motor = torch.Tensor(P0)
    elif isinstance(P0,type(None)):
        P_motor = P_zero

    for i in range(n_joints):
        Rot = zp.transformations.rotation.gen_rot_from_q(qpos[i],joint_axes[i])
        P_motor = R_motor@P[i] + P_motor
        R_motor = R_motor@R[i]@Rot
        FO_link = R_motor@link_zonos[i] + P_motor
        for obs in obs_zonos:
            buff = FO_link - obs
            A,b,_ = buff.polytope()
            if max(A@torch.zeros(3)-b)<1e-6:
                return False
    return True


def traj_safety_check(qpos,qvel,ka,robot_params,obs_zonos,R0=None, P0=None):
    '''
    True: safe, False: unsafe
    '''
    zono_order = 10
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    link_zonos = robot_params['link_zonos']
    joint_axes = robot_params['joint_axes']
    JRS,rotatos = zp.load_JRS_trig(qpos,qvel,joint_axes)
    #import pdb; pdb.set_trace()
    n_time_steps = len(rotatos)
    n_time_steps = 50
    # K_ID = zp.conSet.PROPERTY_ID['k_trig']
    # TODO FIX


    if isinstance(R0,list):
        R0 = torch.Tensor(R0)
    elif isinstance(R0,type(None)):
        R0 = R_identity
    if isinstance(P0,list):
        P0 = torch.Tensor(P0)
    elif isinstance(P0,type(None)):
        P0 = P_zero

    P_motor = zp.polyZonotope(P0)
    R_motor = zp.matPolyZonotope(R0)
    for t in reversed(range(n_time_steps)): # last time step might be more probable to collide
        print(t)
        k_id = [K_ID[i*100+t] for i in range(n_joints)]
        for i in range(n_joints):
            #print(f'FO:{i}')
            P_motor = R_motor@P[i] + P_motor
            P_motor = P_motor.reduce(zono_order)
            #print(f'P_motor:{P_motor.n_generators}')
            R_motor = R_motor@R[i]@rotatos[t][i]
            R_motor = R_motor.reduce(zono_order)
            #print(f'R_motor:{R_motor.n_generators}')
            FO_link_0 = R_motor@link_zonos[i].to_polyZonotope()
            FO_link_0 = FO_link_0.reduce(zono_order)
            FO_link = FO_link_0 + P_motor
            FO_link = FO_link.reduce(zono_order).slice_dep(k_id,ka).to_zonotope()
            #print(f'FO_link:{FO_link.n_generators}')
            #FO_link = FO_link.slice_dep(k_id,ka).to_zonotope()
            for obs in obs_zonos:
                buff = FO_link - obs
                A,b,_ = buff.polytope()
                if max(A@torch.zeros(3)-b)<1e-6:
                    return False
    return True
