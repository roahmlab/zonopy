import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.linalg import block_diag

from torch.multiprocessing import Pool
import zonopy as zp

from zonopy.layer.nlp_setup import NlpSetup3D

# torch.multiprocessing.set_start_method('spawn', force=True)
# os.environ['OMP_NUM_THREADS'] = '2'
import time 

T_PLAN, T_FULL = 0.5, 1.0
NUM_PROCESSES = 40

def rts_pass(A, b, FO_link, qpos, qvel, qgoal, n_timesteps, n_links, n_obs_in_frs, n_pos_lim, actual_pos_lim, vel_lim, lim_flag, dimension, g_ka, ka_0, lambd_hat):
    M_obs = n_links * n_timesteps * int(n_obs_in_frs)
    M = M_obs+2*n_links+6*n_pos_lim
    nlp_obj = NlpSetup3D(A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,int(n_obs_in_frs),n_pos_lim,actual_pos_lim,vel_lim,lim_flag,dimension,g_ka)
    NLP = cyipopt.Problem(
        n=n_links,
        m=M,
        problem_obj=nlp_obj,
        lb = [-1]*n_links,
        ub = [1]*n_links,
        cl = [-1e20]*M,
        cu = [-1e-6]*M,
        )
    NLP.add_option('sb', 'yes')
    NLP.add_option('print_level', 0)
    NLP.add_option('max_cpu_time', 0.2)
    k_opt, info = NLP.solve(ka_0)

    # NOTE: for training, dont care about fail-safe
    if info['status'] == 0:
        lambd_opt = k_opt.tolist()
        flag = 0
    else:
        lambd_opt = lambd_hat.tolist()
        flag = 1
    info['jac_g'] = nlp_obj.jacobian(k_opt)
    return lambd_opt, flag, info


# batch

def gen_grad_RTS_3D_Layer(link_zonos, joint_axes, n_links, n_obs, pos_lim, vel_lim, lim_flag, params, num_processes=NUM_PROCESSES, dtype = torch.float, device=torch.device('cpu'), multi_process=False):
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 3
    n_timesteps = 100

    PI_vel = torch.tensor(torch.pi - 1e-6,dtype=dtype, device=device)
    g_ka = torch.pi / 24

    actual_pos_lim = pos_lim[lim_flag]
    n_pos_lim = int(lim_flag.sum().cpu())

    max_combs = 200 
    combs = [torch.tensor([],device=device)]
    for i in range(max_combs):
        combs.append(torch.combinations(torch.arange(i+1,device=device),2))

    class RTS_grad_3D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation, FO_link):
            zp.reset()
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            ctx.lambd = lambd.clone().reshape(-1, n_links).to(dtype=dtype,device=device)
            # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=dtype,device=device)
            ka = g_ka * ctx.lambd 

            n_batches = observation.shape[0]
            qpos = observation[:, :n_links]
            qvel = observation[:, n_links:2 * n_links]
            obstacle_center = observation[:, -6 * n_obs:-3 * n_obs].reshape(n_batches,n_obs,1,dimension)
            obstacle_generators = torch.diag_embed(observation[:, -3 * n_obs:].reshape(n_batches,n_obs,dimension))
            obs_Z = torch.cat((obstacle_center,obstacle_generators),-2).unsqueeze(-3).repeat(1,1,n_timesteps,1,1)
            
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
            batch_FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)

            As = np.zeros((n_batches,n_links),dtype=object)
            bs = np.zeros((n_batches,n_links),dtype=object)

            FO_links_nlp = np.zeros((n_batches,n_links),dtype=object)
            FO_links = np.zeros((n_links,),dtype=object)
            lambda_to_slc = ctx.lambd.reshape(n_batches, 1, n_links).repeat(1, n_timesteps, 1)

            # pos and vel lim

            # position and velocity constraints
            t_peak_optimum = -qvel/ka # time to optimum of first half traj.
            qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(qpos+qvel*t_peak_optimum+0.5*ka*t_peak_optimum**2).nan_to_num()
            qpos_peak = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN**2
            qvel_peak = qvel + ka * T_PLAN

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = qpos + 0.5*qvel*(T_FULL+T_PLAN) + 0.5 * ka * T_PLAN * T_FULL

            # qpos+qvel*t_peak_optimum+0.5*ka*t_peak_optimum**2 < ub
            # qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN**2 < ub
            # qpos + 0.5*qvel*(T_FULL+T_PLAN) + 0.5 * ka * T_PLAN * T_FULL < ub

            # ka < (ub - qpos - qvel*t_peak_optimum) / (0.5*t_peak_optimum**2)
            # ka  < (ub - qpos - qvel * T_PLAN) / (0.5 * T_PLAN**2)
            # ka  < (ub - qpos - 0.5*qvel*(T_FULL+T_PLAN)) / (0.5 * T_PLAN * T_FULL)

            qpos_possible_max_min = torch.cat((qpos_peak_optimum.unsqueeze(-2),qpos_peak.unsqueeze(-2),qpos_brake.unsqueeze(-2)),-2)[:,:,lim_flag] 
            qpos_ub = (qpos_possible_max_min - actual_pos_lim[:,0]).reshape(n_batches,-1)
            qpos_lb = (actual_pos_lim[:,1] - qpos_possible_max_min).reshape(n_batches,-1)
            
            unsafe_flag = (abs(qvel_peak)>vel_lim).any(-1) + (qpos_ub>0).any(-1) + (qpos_lb>0).any(-1)
            obs_in_reach_idx = torch.zeros(n_batches,n_obs, dtype=bool,device=device)
            
            lambd0 = ctx.lambd.clamp((-PI_vel-qvel)/(g_ka *T_PLAN),(PI_vel-qvel)/(g_ka *T_PLAN)).cpu().numpy()
            
            for j in range(n_links):
                FO_link_temp = batch_FO_link[j]
                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).reshape(n_batches,1,n_timesteps,dimension,1)  # FOR, safety check
                obs_buff_Grest = zp.batchZonotope(torch.cat((obs_Z,FO_link_temp.Grest.unsqueeze(1).repeat(1,n_obs,1,1,1)),-2))
                A_Grest, b_Grest  = obs_buff_Grest.polytope(combs)
                h_obs = ((A_Grest @ c_k).squeeze(-1) - b_Grest).nan_to_num(-torch.inf)
                unsafe_flag += (torch.max(h_obs, -1)[0] < 1e-6).any(-1).any(1)  # NOTE: this might not work on gpu FOR, safety check
                                
                obs_buff = obs_buff_Grest - zp.batchZonotope(FO_link_temp.Z[FO_link_temp.batch_idx_all+(slice(FO_link_temp.n_dep_gens+1),)].unsqueeze(1).repeat(1,n_obs,1,1,1))
                _, b_obs = obs_buff.reduce(6).polytope(combs)
                
                obs_in_reach_idx += (torch.min(b_obs.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)

                As[-1,j] = A_Grest.cpu().numpy()
                bs[-1,j] = b_Grest.cpu().numpy()
                FO_links_nlp[:,j] = [fo for fo in FO_link_temp.cpu()]
                FO_links[j] = FO_link_temp

            unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rts always
            rts_pass_indices = unsafe_flag.nonzero().reshape(-1).tolist()
            n_problems = len(rts_pass_indices)
    
            ctx.flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            ctx.infos = [None for _ in range(n_batches)]

            if n_problems > 0:
                qpos_np = qpos.cpu().numpy()
                qvel_np = qvel.cpu().numpy()
                qgoal_np = qgoal.cpu().numpy()
                lambd_np = ctx.lambd.cpu().numpy()

                actual_pos_lim_np = actual_pos_lim.cpu().numpy()
                vel_lim_np =  vel_lim.cpu().numpy()
                lim_flag_np = lim_flag.cpu().numpy()         
                
                obs_in_reach_idx_list = obs_in_reach_idx.tolist()
                N_obs_in_frs = obs_in_reach_idx.sum(-1).cpu().numpy()
                
                if multi_process:
                    for idx in rts_pass_indices:
                        obs_idx = obs_in_reach_idx_list[idx]
                        for j in range(n_links):
                            As[idx,j] = As[-1,j][idx,obs_idx]
                            bs[idx,j] = bs[-1,j][idx,obs_idx]
                    with Pool(processes=min(num_processes, n_problems)) as pool:
                        results = pool.starmap(
                            rts_pass,
                            [x for x in
                            zip(As[rts_pass_indices],
                                bs[rts_pass_indices],
                                FO_links_nlp[rts_pass_indices],
                                qpos_np[rts_pass_indices],
                                qvel_np[rts_pass_indices],
                                qgoal_np[rts_pass_indices],
                                [n_timesteps] * n_problems,
                                [n_links] * n_problems,
                                N_obs_in_frs[rts_pass_indices], 
                                [n_pos_lim] * n_problems,
                                [actual_pos_lim_np] * n_problems,
                                [vel_lim_np] * n_problems,
                                [lim_flag_np] * n_problems, 
                                [dimension] * n_problems,
                                [g_ka] * n_problems,
                                [lambd0[idx] for idx in rts_pass_indices],  #[ka_0] * n_problems,
                                lambd_np[rts_pass_indices]
                            )
                            ]
                        )
                    rts_lambd_opts, rts_flags = [], []
                    for idx, res in enumerate(results):
                        rts_lambd_opts.append(res[0])
                        rts_flags.append(res[1])
                        ctx.infos[rts_pass_indices[idx]] = res[2]
                    ctx.lambd[rts_pass_indices] = torch.tensor(rts_lambd_opts,dtype=dtype,device=device)
                    ctx.flags[rts_pass_indices] = torch.tensor(rts_flags, dtype=ctx.flags.dtype, device=device)
                else:
                    rts_lambd_opts, rts_flags = [], []                    
                    for idx in rts_pass_indices:
                        
                        obs_idx = obs_in_reach_idx_list[idx]
                        for j in range(n_links):
                            As[idx,j] = As[-1,j][idx,obs_idx]
                            bs[idx,j] = bs[-1,j][idx,obs_idx]
                        rts_lambd_opt, rts_flag, info = rts_pass(
                                                                As[idx],
                                                                bs[idx],
                                                                FO_links_nlp[idx],
                                                                qpos_np[idx],
                                                                qvel_np[idx],
                                                                qgoal_np[idx],
                                                                n_timesteps,
                                                                n_links,
                                                                N_obs_in_frs[idx],
                                                                n_pos_lim,
                                                                actual_pos_lim_np,
                                                                vel_lim_np,
                                                                lim_flag_np, 
                                                                dimension,
                                                                g_ka,
                                                                lambd0[idx],
                                                                lambd_np[idx])
                        ctx.infos[idx] = info
                        rts_lambd_opts.append(rts_lambd_opt)
                        rts_flags.append(rts_flag)
                    ctx.lambd[rts_pass_indices] = torch.tensor(rts_lambd_opts,dtype=dtype,device=device)
                    ctx.flags[rts_pass_indices] = torch.tensor(rts_flags, dtype=ctx.flags.dtype, device=device)
            
            zp.reset()
            return ctx.lambd, FO_links, ctx.flags, ctx.infos

        @staticmethod
        def backward(ctx, *grad_ouput):
            direction = grad_ouput[0]
            grad_input = torch.zeros_like(direction,dtype=dtype,device=device)
            # COMPUTE GRADIENT
            tol = 1e-6
            # direct pass
            direct_pass = (ctx.flags == -1) + (ctx.flags == 1) # NOTE: (ctx.flags == -1)
            grad_input[direct_pass] = direction[direct_pass]

            rts_success_pass = (ctx.flags == 0).nonzero().reshape(-1)
            n_batch = rts_success_pass.numel()
            if n_batch > 0:
                QP_EQ_CONS = []
                QP_INEQ_CONS = []
                lambd = ctx.lambd[rts_success_pass].cpu().numpy()
                for j,i in enumerate(rts_success_pass):
                    k_opt = lambd[j]
                    # compute jacobian of each smooth constraint which will be constraints for QP
                    jac = ctx.infos[i]['jac_g']
                    cons = ctx.infos[i]['g']

                    qp_cons1 = jac  # [A*c(k)-b].T*lambda  and vel. lim # NOTE
                    EYE = np.eye(n_links)
                    qp_cons4 = -EYE  # lb
                    qp_cons5 = EYE  # ub
                    qp_cons = np.vstack((qp_cons1, qp_cons4, qp_cons5))

                    # compute duals for smooth constraints                
                    mult_smooth_cons1 = ctx.infos[i]['mult_g'] * (ctx.infos[i]['mult_g'] > tol)
                    mult_smooth_cons4 = ctx.infos[i]['mult_x_L'] * (ctx.infos[i]['mult_x_L'] > tol)
                    mult_smooth_cons5 = ctx.infos[i]['mult_x_U'] * (ctx.infos[i]['mult_x_U'] > tol)
                    mult_smooth = np.hstack((mult_smooth_cons1, mult_smooth_cons4, mult_smooth_cons5))

                    # compute smooth constraints
                    smooth_cons1 = cons * (cons < -1e-6 - tol)
                    smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt < -1e-6 - tol)
                    smooth_cons5 = (k_opt - 1) * (k_opt - 1 < -1e-6 - tol)
                    smooth_cons = np.hstack((smooth_cons1, smooth_cons4, smooth_cons5))

                    active = (smooth_cons >= -1e-6 - tol)
                    strongly_active = (mult_smooth > tol) * active
                    weakly_active = (mult_smooth <= tol) * active

                    QP_EQ_CONS.append(qp_cons[strongly_active])
                    QP_INEQ_CONS.append(qp_cons[weakly_active])

                # reduced batch QP
                # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
                qp_size = n_batch * n_links
                H = 0.5 * sp.csr_matrix(([1.] * qp_size, (range(qp_size), range(qp_size))))
                f_d = sp.csr_matrix((-direction[rts_success_pass].cpu().flatten(), ([0] * qp_size, range(qp_size))))
                qp = gp.Model("back_prop")
                qp.Params.LogToConsole = 0
                z = qp.addMVar(shape=qp_size, name="z", vtype=GRB.CONTINUOUS, ub=np.inf, lb=-np.inf)
                qp.setObjective(z @ H @ z + f_d @ z, GRB.MINIMIZE)
                qp_eq_cons = sp.csr_matrix(block_diag(*QP_EQ_CONS))
                rhs_eq = np.zeros(qp_eq_cons.shape[0])
                qp_ineq_cons = sp.csr_matrix(block_diag(*QP_INEQ_CONS))
                rhs_ineq = -0 * np.ones(qp_ineq_cons.shape[0])
                qp.addConstr(qp_eq_cons @ z == rhs_eq, name="eq")
                qp.addConstr(qp_ineq_cons @ z <= rhs_ineq, name="ineq")
                qp.optimize()
                grad_input[rts_success_pass] = torch.tensor(z.X.reshape(n_batch, n_links),dtype=dtype,device=device)

                # NOTE: for fail-safe, keep into zeros             
            return (grad_input.reshape(ctx.lambd_shape), torch.zeros(ctx.obs_shape,dtype=dtype,device=device), None)

    return RTS_grad_3D_Layer.apply


if __name__ == '__main__':
    
    from zonopy.environments.parallel_arm_3d import Parallel_Arm_3D
    import time
    ##### 0. SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### 1. SET ENVIRONMENT #####
    n_batch = 9
    env = Parallel_Arm_3D(n_envs = n_batch, n_obs=10, n_plots = 4)

    ##### 2. GENERATE RTS LAYER #####    
    P,R,link_zonos = [],[],[]
    for p,r,l in zip(env.P0,env.R0,env.link_zonos):
        P.append(p.to(device=device,dtype=dtype))
        R.append(r.to(device=device,dtype=dtype))
        link_zonos.append(l.to(device=device,dtype=dtype))
    params = {'n_joints': env.n_links, 'P': P, 'R': R}
    joint_axes = [j for j in env.joint_axes.to(device=device,dtype=dtype)]
    RTS = gen_grad_RTS_3D_Layer(link_zonos, joint_axes, env.n_links, env.n_obs, env.pos_lim, env.vel_lim, env.lim_flag, params,device=device,dtype=dtype)

    ##### 3. RUN RTS #####
    t_forward, t_backward = 0, 0 
    t_render = 0
    n_steps = 30
    
    print('='*90)
    observation = env.reset()
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].reshape(n_batch,-1) for key in observation.keys()])

        lam = torch.tensor([0.8]*7,device=device,dtype=dtype)
        bias = torch.full((n_batch, 1), 0.0, requires_grad=True,device=device,dtype=dtype)
        lam, FO_link, flag, nlp_info = RTS(torch.vstack(([lam] * n_batch))+bias, observ_temp, None)
              
        print(f'action: {lam[0]}')
        print(f'flag: {flag[0]}')

        t_elasped = time.time() - ts
        t_forward += t_elasped
        print(f'Time elasped for RTS forward:{t_elasped}')

        ts = time.time()        
        lam.sum().backward(retain_graph=True)
        t_elasped = time.time() - ts       
        t_backward += t_elasped
        print(f'Time elasped for RTS backward:{t_elasped}')
        print('='*90)
        
        observation, reward, done, info = env.step(lam.cpu().to(dtype=torch.get_default_dtype()) * torch.pi / 24, flag.cpu().to(dtype=torch.get_default_dtype()))

        ts = time.time()
        #env.render(FO_link)
        env.render()        
        t_render += time.time()-ts 



    print(f'Total time elasped for RTS forward with {n_steps} steps: {t_forward}')
    print(f'Total time elasped for RTS backward with {n_steps} steps: {t_backward}')
    print(f'Total time elasped for rendering with {n_steps} steps: {t_render}')