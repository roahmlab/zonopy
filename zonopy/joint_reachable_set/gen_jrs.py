import torch
from zonopy.transformations.rotation import gen_rotatotope_from_jrs
from zonopy.joint_reachable_set.utils import remove_dependence_and_compress
import zonopy as zp
PI = torch.tensor(torch.pi)

def gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=True):
    n_q = len(q)
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0]) for _ in range(n_q)]
    
    traj_type = 'orig'
    T_full,T_plan, dt = 1, 0.5 ,0.01
    n_t = int(T_full/dt)
    n_t_p = int(T_plan/T_full*n_t)

    c_k = torch.zeros(n_q)
    g_k = torch.min(torch.max(PI/24,abs(dq/3)),PI/3)
    T, K = [],[]
    Q, R= [[[] for _ in range(n_t)]for _ in range(2)]
    for _ in range(n_q):
        K.append(zp.polyZonotope([0],[[1]],prop='k'))
    for t in range(n_t):
        T.append(zp.polyZonotope([dt*t+dt/2],[[dt/2]]))

    Qd_plan, Qdd_brake, Q_plan = [[] for _ in range(3)]
    if traj_type == 'orig':
        for j in range(n_q):
            Qd_plan.append(dq[j]+(c_k[j]+g_k[j]*K[j])*T_plan)
            Qdd_brake.append((-1)*Qd_plan[-1]*(1/(T_full-T_plan))) 
            Q_plan.append(q[j]+dq[j]*T_plan+0.5*(c_k[j]+g_k[j]*K[j])*T_plan**2)

    # main loop
    for t in range(n_t):
        for j in range(n_q):
            if traj_type == 'orig':
                if t < n_t_p:
                   Q[t].append(q[j]+dq[j]*T[t]+0.5*(c_k[j]+g_k[j]*K[j])*T[t]*T[t])
                else:
                   Q[t].append(Q_plan[j]+Qd_plan[j]*(T[t]-T_plan)+0.5*Qdd_brake[j]*(T[t]-T_plan)*(T[t]-T_plan))

            R_temp= gen_rotatotope_from_jrs(Q[t][-1],joint_axes[j],taylor_degree)
            R[t].append(R_temp)

    

    # throw away time/error gens and compress indep. gens to just one (if 1D) 
    if make_gens_independent:
        for t in range(n_t):
            for j in range(n_q):
                k_id = zp.conSet.PROPERTY_ID['k'][j]
                Q[t][j] = remove_dependence_and_compress(Q[t][j], k_id)
                R[t][j] = remove_dependence_and_compress(R[t][j], k_id)
        
    return Q, R



def gen_traj_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=True):

    n_q = len(q)
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0]) for _ in range(n_q)]
    
    traj_type = 'orig'
    ultimate_bound = 0.0191
    k_r = 10 # Kr = kr*eye(n_q)

    T_full,T_plan, dt = 1, 0.5 ,0.01
    n_t = int(T_full/dt)
    n_t_p = int(T_plan/T_full*n_t)

    c_k = torch.zeros(n_q)
    g_k = torch.min(torch.max(PI/24,abs(dq/3)),PI/3)
    T, K = [],[]
    Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t = [[[] for _ in range(n_t)]for _ in range(11)]
    for _ in range(n_q):
        K.append(zp.polyZonotope([0],[[1]],prop='k'))
    for t in range(n_t):
        T.append(zp.polyZonotope([dt*t+dt/2],[[dt/2]]))
    E_p = zp.polyZonotope([0],[[ultimate_bound/k_r]])
    E_v = zp.polyZonotope([0],[[2*ultimate_bound]])

    Qd_plan, Qdd_brake, Q_plan = [[] for _ in range(3)]
    if traj_type == 'orig':
        for j in range(n_q):
            Qd_plan.append(dq[j]+(c_k[j]+g_k[j]*K[j])*T_plan)
            Qdd_brake.append((-1)*Qd_plan[-1]*(1/(T_full-T_plan))) 
            Q_plan.append(q[j]+dq[j]*T_plan+0.5*(c_k[j]+g_k[j]*K[j])*T_plan**2)

    # main loop
    for t in range(n_t):
        for j in range(n_q):
            if traj_type == 'orig':
                if t < n_t_p:
                   Q_des[t].append(q[j]+dq[j]*T[t]+0.5*(c_k[j]+g_k[j]*K[j])*T[t]*T[t])
                   Qd_des[t].append(dq[j]+(c_k[j]+g_k[j]*K[j])*T[t])
                   Qdd_des[t].append(c_k[j]+g_k[j]*K[j])
                else:
                   Q_des[t].append(Q_plan[j]+Qd_plan[j]*(T[t]-T_plan)+0.5*Qdd_brake[j]*(T[t]-T_plan)*(T[t]-T_plan))
                   Qd_des[t].append(Qd_plan[j]+Qdd_brake[j]*(T[t]-T_plan))
                   Qdd_des[t].append(Qdd_brake[j])

            Q[t].append(Q_des[t][-1]+E_p)
            Qd[t].append(Qd_des[t][-1]+E_v)
            Qd_a[t].append(Qd_des[t][-1]+k_r*E_p)
            Qdd_a[t].append(Qd_des[t][-1]+k_r*E_v)
            R_temp= gen_rotatotope_from_jrs(Q_des[t][-1],joint_axes[j],taylor_degree)
            R_des[t].append(R_temp)
            R_t_des[t].append(R_temp.T)
            R_temp = gen_rotatotope_from_jrs(Q[t][-1],joint_axes[j],taylor_degree)
            R[t].append(R_temp)
            R_t[t].append(R_temp.T)
    

    # throw away time/error gens and compress indep. gens to just one (if 1D) 
    if make_gens_independent:
        for t in range(n_t):
            for j in range(n_q):
                k_id = zp.conSet.PROPERTY_ID['k'][j]
                Q_des[t][j] = remove_dependence_and_compress(Q_des[t][j], k_id)
                Qd_des[t][j] = remove_dependence_and_compress(Qd_des[t][j], k_id)
                Qdd_des[t][j] = remove_dependence_and_compress(Qdd_des[t][j], k_id)
                Q[t][j] = remove_dependence_and_compress(Q[t][j], k_id)
                Qd[t][j] = remove_dependence_and_compress(Qd[t][j], k_id)
                Qd_a[t][j] = remove_dependence_and_compress(Qd_a[t][j], k_id)
                Qdd_a[t][j] = remove_dependence_and_compress(Qdd_a[t][j], k_id)
                R_des[t][j] = remove_dependence_and_compress(R_des[t][j], k_id)
                R_t_des[t][j] = remove_dependence_and_compress(R_t_des[t][j], k_id)
                R[t][j] = remove_dependence_and_compress(R[t][j], k_id)
                R_t[t][j] = remove_dependence_and_compress(R_t[t][j], k_id)
    #import pdb; pdb.set_trace()
    return Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t





if __name__ == '__main__':
    q = torch.tensor([0])
    dq = torch.tensor([torch.pi])
    Q,R = gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=False)
    import pdb; pdb.set_trace()
    

    PZ_JRS,_ = zp.load_JRS_trig(q,dq)
    ax = zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='blue',hold_on=True)
    #zp.plot_polyzonos(PZ_JRS,plot_freq=1,ax=ax)
    
    zp.plot_JRSs(Q,deg=1,plot_freq=1,ax=ax)
    
    
    