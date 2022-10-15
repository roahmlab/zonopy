import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
from zonopy.conSet import PROPERTY_ID
import zonopy as zp
import cyipopt

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.linalg import block_diag

#import os
from torch.multiprocessing import Pool

from zonopy.layer.nlp_setup import NlpSetup2D

#os.environ['OMP_NUM_THREADS'] = '2'

T_PLAN, T_FULL = 0.5, 1.0

NUM_PROCESSES = 40

EPS = 1e-6
TOL = 1e-4

def rts_pass(A, b, FO_link, qpos, qvel, qgoal, n_timesteps, n_links, n_obs_in_frs, dimension, g_ka, ka_0, lambd_hat):
    M_obs = n_timesteps * n_links * int(n_obs_in_frs)
    M = M_obs + 2 * n_links
    nlp_obj = NlpSetup2D(A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,int(n_obs_in_frs),dimension,g_ka)
    NLP = cyipopt.Problem(
        n=n_links,
        m=M,
        problem_obj=nlp_obj,
        lb=[-1] * n_links,
        ub=[1] * n_links,
        cl=[-1e20] * M_obs + [-1e20] * 2 * n_links,
        cu=[-EPS] * M_obs + [-EPS] * 2 * n_links,
    )
    NLP.add_option('sb', 'yes')
    NLP.add_option('print_level', 0)
    #NLP.add_option('max_cpu_time', 0.2)
    NLP.add_option('max_iter',15)
    #NLP.add_option('hessian_approximation','limited-memory')
    NLP.add_option('tol', TOL)
    NLP.add_option('linear_solver', 'ma27')
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


def gen_grad_RTS_2D_Layer(link_zonos, joint_axes, n_links, n_obs, params, num_processes=NUM_PROCESSES, dtype = torch.float, device=torch.device('cpu'), multi_process=False, gradient_step_sign = '-'):
    assert gradient_step_sign == '-' or gradient_step_sign == '+'
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 2
    n_timesteps = 100
    #ka_0 = np.zeros(n_links)
    PI_vel = torch.tensor(torch.pi - 1e-6,dtype=dtype, device=device)
    g_ka = torch.pi / 24

    class grad_RTS_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation, batch_FO_link):
            zp.reset()
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            ctx.lambd_shape = lambd.shape
            ctx.lambd = lambd.clone().reshape(-1, n_links).to(dtype=dtype,device=device)
            
            ka = g_ka * ctx.lambd
            if isinstance(observation,dict):
                ctx.obs_type = 'dict'
                ctx.obs_shape = observation["observation"].shape 
                n_batches, obs_dim = ctx.obs_shape
                qpos = observation["achieved_goal"].to(dtype=dtype,device=device)
                observation_observation = observation["observation"].to(dtype=dtype,device=device)
                qvel = observation_observation[:,:n_links]            
                obstacle_center = observation_observation[:, obs_dim -4 * n_obs: obs_dim -2 * n_obs].reshape(n_batches,n_obs,1,dimension)
                obstacle_generators = torch.diag_embed(observation_observation[:, obs_dim -2 * n_obs:].reshape(n_batches,n_obs,dimension))

            else:
                ctx.obs_type = 'tensor'
                ctx.obs_shape = observation.shape
                observation = observation.to(dtype=dtype,device=device)
                n_batches, obs_dim = observation.shape
                qpos = observation[:, :n_links]
                qvel = observation[:, n_links:2 * n_links]
                obstacle_center = observation[:, -4 * n_obs:-2 * n_obs].reshape(n_batches,n_obs,1,dimension)
                obstacle_generators = torch.diag_embed(observation[:, -2 * n_obs:].reshape(n_batches,n_obs,dimension))


            obs_Z = torch.cat((obstacle_center,obstacle_generators),-2).unsqueeze(-3).repeat(1,1,n_timesteps,1,1)
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            if batch_FO_link is None:
                _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
                batch_FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)
            else:
                zp.reset(n_links)

            As = np.zeros((n_batches,n_links),dtype=object)
            bs = np.zeros((n_batches,n_links),dtype=object)
            FO_links_nlp = np.zeros((n_batches,n_links),dtype=object)
            FO_links = np.zeros((n_links,),dtype=object)
            lambda_to_slc = ctx.lambd.reshape(n_batches, 1, n_links).repeat(1, n_timesteps, 1)

            # unsafe_flag = torch.zeros(n_batches)
            unsafe_flag = (abs(qvel + lambd * g_ka * T_PLAN) > PI_vel).any(-1)
            obs_in_reach_idx = torch.zeros(n_batches,n_obs, dtype=bool,device=device)
            lambd0 = lambd.clamp((-PI_vel-qvel)/(g_ka *T_PLAN),(PI_vel-qvel)/(g_ka *T_PLAN)).cpu().numpy()

            for j in range(n_links):
                FO_link_temp = batch_FO_link[j].project([0, 1])
                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).reshape(n_batches,1,n_timesteps,dimension,1)  # FOR, safety check
                obs_buff_Grest = zp.batchZonotope(torch.cat((obs_Z,FO_link_temp.Grest.unsqueeze(1).repeat(1,n_obs,1,1,1)),-2))
                A_Grest, b_Grest  = obs_buff_Grest.polytope()
                h_obs = ((A_Grest @ c_k).squeeze(-1) - b_Grest).nan_to_num(-torch.inf)
                unsafe_flag += (torch.max(h_obs, -1)[0] < 1e-6).any(-1).any(1)  # NOTE: this might not work on gpu FOR, safety check
                                
                obs_buff = obs_buff_Grest - zp.batchZonotope(FO_link_temp.Z[FO_link_temp.batch_idx_all+(slice(FO_link_temp.n_dep_gens+1),)].unsqueeze(1).repeat(1,n_obs,1,1,1))
                _, b_obs = obs_buff.reduce(6).polytope()

                obs_in_reach_idx += (torch.min(b_obs.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)

                As[-1,j] = A_Grest.cpu().numpy()
                bs[-1,j] = b_Grest.cpu().numpy()
                FO_links_nlp[:,j] = [fo for fo in FO_link_temp.cpu()]
                FO_links[j] = FO_link_temp

            #unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rts always
            rts_pass_indices = unsafe_flag.nonzero().reshape(-1).tolist()
            n_problems = len(rts_pass_indices)

            ctx.flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            ctx.infos = [None for _ in range(n_batches)]

            if n_problems > 0:
                qpos_np = qpos.cpu().numpy()
                qvel_np = qvel.cpu().numpy()
                qgoal_np = qgoal.cpu().numpy()
                lambd_np = ctx.lambd.cpu().numpy()

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
                                N_obs_in_frs,
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
                                                                dimension,
                                                                g_ka,lambd0[idx],
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
            tol = 0.9*TOL
            # direct pass
            direct_pass = (ctx.flags == -1) + (ctx.flags == 1) # NOTE: (ctx.flags == -1)
            grad_input[direct_pass] = direction[direct_pass]

            rts_success_pass = (ctx.flags == 0).nonzero().reshape(-1)
            n_batch = rts_success_pass.numel()
            if n_batch > 0:
                QP_EQ_CONS = []
                QP_INEQ_CONS = []
                qp_solve_ind = []
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
                    smooth_cons1 = cons * (cons < -EPS - tol)
                    smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt < - tol)
                    smooth_cons5 = (k_opt - 1) * (k_opt - 1 < - tol)
                    smooth_cons = np.hstack((smooth_cons1, smooth_cons4, smooth_cons5))

                    active = (smooth_cons >= -EPS - tol)
                    strongly_active = (mult_smooth > tol) * active
                    weakly_active = (mult_smooth <= tol) * active

                    strong_qp_cons = qp_cons[strongly_active] 
                    weak_qp_cons = qp_cons[weakly_active]
                    
                    # normalize constraint for numerical stability
                    strong_qp_cons = np.nan_to_num(strong_qp_cons/np.linalg.norm(strong_qp_cons,axis=-1,keepdims=True))
                    weak_qp_cons = np.nan_to_num(weak_qp_cons/np.linalg.norm(weak_qp_cons,axis=-1,keepdims=True))

                    if strongly_active.sum() < n_links or np.linalg.matrix_rank(strong_qp_cons) < n_links:
                        QP_EQ_CONS.append(strong_qp_cons)
                        QP_INEQ_CONS.append(weak_qp_cons)
                        qp_solve_ind.append(int(i))

                # reduced batch QP
                n_qp = len(qp_solve_ind)
                if n_qp > 0:
                    # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
                    qp_size = n_qp * n_links
                    H = 0.5 * sp.csr_matrix(([1.] * qp_size, (range(qp_size), range(qp_size))))
                    if gradient_step_sign == '-':
                        f_d_unscale = direction[qp_solve_ind].cpu().numpy().flatten()
                    else:
                        f_d_unscale = - direction[qp_solve_ind].cpu().numpy().flatten()
                    scale_factor_f_d = abs(f_d_unscale).min() # scale f_d for numerical stability of Gurobi
                    f_d = sp.csr_matrix((f_d_unscale/scale_factor_f_d, ([0] * qp_size, range(qp_size))))
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
                    try:
                        if gradient_step_sign == '-':
                            grad_input[qp_solve_ind] = - scale_factor_f_d * torch.tensor(z.X.reshape(n_qp, n_links),dtype=dtype,device=device)
                        else:
                            grad_input[qp_solve_ind] = scale_factor_f_d * torch.tensor(z.X.reshape(n_qp, n_links),dtype=dtype,device=device)
                    except:
                        import pickle
                        dump = {'flags':ctx.flags.cpu(), 'lambd':ctx.lambd.cpu(), 'infos':ctx.infos, 'rts_success_pass':rts_success_pass.cpu(),'direction':direction.cpu()}
                        with open('gurobi_fail_data.pickle', 'wb') as handle:
                            pickle.dump(dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Training is quit due to GUROBI.')
                        exit()
                # NOTE: for fail-safe, keep into zeros             
            if ctx.obs_type == 'dict':
                return (grad_input.reshape(ctx.lambd_shape), None, None)
            else:
                return (grad_input.reshape(ctx.lambd_shape), torch.zeros(ctx.obs_shape,dtype=dtype,device=device),None)

    return grad_RTS_2D_Layer.apply


if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    from zonopy.environments.parallel_arm_2d import Parallel_Arm_2D
    import time

    ##### 0.DEVICE CUDA #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float64
    else:
        device = 'cpu'
        dtype = torch.float


    ##### 1. SET ENVIRONMENT #####
    n_links = 2
    n_batch = 9
    env = Parallel_Arm_2D(n_envs = n_batch, n_links=n_links, n_obs=1, n_plots = 4)
    observation = env.set_initial(qpos=torch.tensor([[0.1 * torch.pi, 0.1 * torch.pi]]).repeat(n_batch,1), 
                                  qvel=torch.zeros(n_batch,n_links),
                                  qgoal=torch.tensor([[-0.5 * torch.pi, -0.8 * torch.pi]]).repeat(n_batch,1),
                                  obs_pos=[torch.tensor([[-1, -0.9]]).repeat(n_batch,1)])

    ##### 2. GENERATE RTS LAYER #####    
    P,R,link_zonos = [],[],[]
    for p,r,l in zip(env.P0,env.R0,env.link_zonos):
        P.append(p.to(device=device,dtype=dtype))
        R.append(r.to(device=device,dtype=dtype))
        link_zonos.append(l.to(device=device,dtype=dtype))
    params = {'n_joints': env.n_links, 'P': P, 'R': R}
    joint_axes = [j for j in env.joint_axes.to(device=device,dtype=dtype)]
    RTS = gen_grad_RTS_2D_Layer(link_zonos, joint_axes, env.n_links, env.n_obs, params,device=device,dtype=dtype)

    ##### 3. RUN RTS #####
    t_forward, t_backward = 0, 0 
    t_render = 0
    n_steps = 30
    print('='*90)
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].reshape(n_batch,-1) for key in observation.keys()])

        lam_hat = torch.tensor([0.8, 0.8],device=device,dtype=dtype)
        bias = torch.full((n_batch, 1), 0.0, requires_grad=True,device=device,dtype=dtype)
        lam, FO_link, flag, nlp_info = RTS(torch.vstack([lam_hat] * n_batch)+bias, observ_temp, None)
        
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

        env.render(FO_link)


    print(f'Total time elasped for RTS forward with {n_steps} steps: {t_forward}')
    print(f'Total time elasped for RTS backward with {n_steps} steps: {t_backward}')