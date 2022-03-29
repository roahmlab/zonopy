import torch

def gen_arm_traj(q_0, qd_0, k, T_len=200,down_sample=2):
    assert q_0.shape == qd_0.shape
    assert int(T_len/2) == T_len/2 
    # Time
    T_plan, T_total = 0.5, 1
    t_traj = torch.linspace(0,T_total,T_len+1)
    t_to_peak = t_traj[:int(T_plan/T_total*T_len)+1]
    t_to_stop = t_traj[int(T_plan/T_total*T_len)+1:]
    
    # to peak
    q_to_peak = q_0.reshape(-1,1) + torch.outer(qd_0,t_to_peak) + .5*torch.outer(k,t_to_peak**2)
    qd_to_peak = qd_0.reshape(-1,1) + torch.outer(k,t_to_peak)
    qdd_to_peak = torch.outer(k,torch.ones_like(t_to_peak))

    q_peak = q_to_peak[:,-1]
    qd_peak = qd_to_peak[:,-1]

    #to stop
    bracking_accel = (0 - qd_peak)/(T_total - T_plan)
    t_to_brake = t_to_stop - T_plan
    
    q_to_stop = q_peak.reshape(-1,1) + torch.outer(qd_peak,t_to_brake) + .5*torch.outer(bracking_accel,t_to_brake**2)
    qd_to_stop = qd_peak.reshape(-1,1) + torch.outer(bracking_accel,t_to_brake)
    qdd_to_stop = torch.outer(bracking_accel,torch.ones_like(t_to_brake))

    # combine
    q_traj = torch.hstack((q_to_peak,q_to_stop))
    qd_traj = torch.hstack((qd_to_peak,qd_to_stop))
    qdd_traj = torch.hstack((qdd_to_peak,qdd_to_stop))
    
    # downsample
    if down_sample is not None:
        idx_downsample = range(1,T_len+1,down_sample)
        q_traj = q_traj[idx_downsample]
        qd_traj = qd_traj[idx_downsample]
        qdd_traj = qdd_traj[idx_downsample]
        t_traj = t_traj[idx_downsample]
        
    return q_traj, qd_traj, qdd_traj, t_traj