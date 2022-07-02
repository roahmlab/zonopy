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

from zonopy.layer.nlp_setup import NLP_setup

#os.environ['OMP_NUM_THREADS'] = '2'

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi


T_PLAN, T_FULL = 0.5, 1.0

NUM_PROCESSES = 40


def rts_pass(A, b, FO_link, qpos, qvel, qgoal, n_timesteps, n_links, n_obs, dimension, g_ka, ka_0, lambd_hat):
    M_obs = n_timesteps * n_links * n_obs
    M = M_obs + 2 * n_links
    nlp_obj = NLP_setup(qpos,qvel,qgoal,n_timesteps,n_links,dimension,n_obs,g_ka,FO_link,A,b)
    NLP = cyipopt.Problem(
        n=n_links,
        m=M,
        problem_obj=nlp_obj,
        lb=[-1] * n_links,
        ub=[1] * n_links,
        cl=[-1e20] * M_obs + [-1e20] * 2 * n_links,
        cu=[-1e-6] * M_obs + [-1e-6] * 2 * n_links,
    )
    NLP.add_option('sb', 'yes')
    NLP.add_option('print_level', 0)
    k_opt, info = NLP.solve(ka_0)

    # NOTE: for training, dont care about fail-safe
    if info['status'] == 0:
        lambd_opt = torch.tensor(k_opt, dtype=torch.get_default_dtype())
        flag = 0
    else:
        lambd_opt = lambd_hat
        flag = 1
    
    info['jac_g'] = nlp_obj.jacobian(k_opt)
    #nlp_info['cons'] = nlp_obj.cons

    return lambd_opt, flag, info


def gen_grad_RTS_2D_Layer(link_zonos, joint_axes, n_links, n_obs, params, num_processes=NUM_PROCESSES, dtype = torch.float, device='cpu',multi_process=True):
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 2
    n_timesteps = 100
    #ka_0 = np.zeros(n_links)
    PI_vel = torch.tensor(torch.pi - 1e-6,dtype=dtype, device=device)
    g_ka = torch.pi / 24

    class grad_RTS_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation, FO_link):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            ctx.lambd = lambd.clone().reshape(-1, n_links).to(dtype=dtype,device=device)
            # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=dtype,device=device)
            ka = g_ka * ctx.lambd

            n_batches = observation.shape[0]
            qpos = observation[:, :n_links]
            qvel = observation[:, n_links:2 * n_links]
            obstacle_pos = observation[:, -4 * n_obs:-2 * n_obs]
            obstacle_size = observation[:, -2 * n_obs:]
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            # g_ka = torch.maximum(PI/24,abs(qvel/3))
            if FO_link is None:
                _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
                FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)
            else:
                PROPERTY_ID.update(n_links)
                '''
                FO_new = []
                for j in range(n_links):
                    max_size =  max([frs.n_generators for frs in FO_link[:,j]]) 
                    zero_pad = torch.zeros(n_timesteps,max_size,dimension)
                    Z = torch.cat([torch.cat((frs.Z,zero_pad[:,:max_size-frs.n_generators,:]),1).unsqueeze(0) for frs in FO_link[:,j]],0)
                    temp = FO_link[0,j]
                    FO_new.append(batchPolyZonotope(Z,temp.n_dep_gens,temp.expMat,compress=0)) 
                FO_link = FO_new
                '''

            As = np.zeros((n_batches,n_links,n_obs),dtype=object)
            bs = np.zeros((n_batches,n_links,n_obs),dtype=object)
            FO_links = np.zeros((n_batches,n_links),dtype=object)
            lambda_to_slc = ctx.lambd.reshape(n_batches, 1, dimension).repeat(1, n_timesteps, 1)

            # unsafe_flag = torch.zeros(n_batches)
            unsafe_flag = (abs(qvel + ctx.lambd * g_ka * T_PLAN) > PI_vel).any(-1)  # NOTE: this might not work on gpu, velocity lim check
            lambd0 = ctx.lambd.clamp((-PI_vel-qvel)/(g_ka *T_PLAN),(PI_vel-qvel)/(g_ka *T_PLAN)).cpu().numpy()
            for j in range(n_links):
                FO_link_temp = FO_link[j].project([0, 1])

                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).unsqueeze(-1)  # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:, 2 * o:2 * (o + 1)].unsqueeze(-2), torch.diag_embed(obstacle_size[:, 2 * o:2 * (o + 1)])), -2).unsqueeze(-3).repeat(1, n_timesteps, 1, 1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z, FO_link_temp.Grest),-2)).polytope()  # A: n_timesteps,*,dimension
                    unsafe_flag += (torch.max((A_temp @ c_k).squeeze(-1) - b_temp, -1)[0] < 1e-6).any(-1)  # NOTE: this might not work on gpu FOR, safety check
                    As[:,j,o] = list(A_temp.cpu().numpy())
                    bs[:,j,o] = list(b_temp.cpu().numpy())
                FO_links[:,j] = [fo for fo in FO_link_temp.cpu()]
            #unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rts all ways

            ctx.flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            ctx.infos = [None for _ in range(n_batches)]

            rts_pass_indices = unsafe_flag.nonzero().reshape(-1).tolist()
            n_problems = len(rts_pass_indices)
            if n_problems > 0:
                if multi_process:
                    with Pool(processes=min(num_processes, n_problems)) as pool:
                        results = pool.starmap(
                            rts_pass,
                            [x for x in
                            zip(As[rts_pass_indices],
                                bs[rts_pass_indices],
                                FO_links[rts_pass_indices],
                                qpos.cpu().numpy()[rts_pass_indices],
                                qvel.cpu().numpy()[rts_pass_indices],
                                qgoal.cpu().numpy()[rts_pass_indices],
                                [n_timesteps] * n_problems,
                                [n_links] * n_problems,
                                [n_obs] * n_problems,
                                [dimension] * n_problems,
                                [g_ka] * n_problems,
                                [lambd0[idx] for idx in rts_pass_indices],  #[ka_0] * n_problems,
                                lambd[rts_pass_indices]
                            )
                            ]
                        )
                    rts_lambd_opt, rts_flags = [], []
                    for idx, res in enumerate(results):
                        rts_lambd_opt.append(res[0])
                        rts_flags.append(res[1])
                        ctx.infos[rts_pass_indices[idx]] = res[2]
                    ctx.lambd[rts_pass_indices] = torch.cat(rts_lambd_opt, 0).view(n_problems, dimension).to(dtype=dtype,device=device)
                    ctx.flags[rts_pass_indices] = torch.tensor(rts_flags, dtype=ctx.flags.dtype, device=device)
                else:
                    for idx in rts_pass_indices:
                        #import pdb;pdb.set_trace()
                        rts_lambd_opt, rts_flag, info = rts_pass(As[idx],bs[idx],FO_links[idx],qpos.cpu().numpy()[idx],qvel.cpu().numpy()[idx],qgoal.cpu().numpy()[idx],n_timesteps,n_links,n_obs,dimension,g_ka,lambd0[idx],lambd[idx])
                        ctx.infos[idx] = info
                        ctx.lambd[idx] = rts_lambd_opt
                        ctx.flags[idx] = int(rts_flag)


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
                f_d = sp.csr_matrix((-direction[rts_success_pass].flatten(), ([0] * qp_size, range(qp_size))))
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

    return grad_RTS_2D_Layer.apply


if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time

    n_links = 2
    env = Arm_2D(n_links=n_links, n_obs=1)
    observation = env.set_initial(qpos=torch.tensor([0.1 * torch.pi, 0.1 * torch.pi]), qvel=torch.zeros(n_links),
                                  qgoal=torch.tensor([-0.5 * torch.pi, -0.8 * torch.pi]),
                                  obs_pos=[torch.tensor([-1, -0.9])])

    t_armtd = 0
    params = {'n_joints': env.n_links, 'P': env.P0, 'R': env.R0}
    joint_axes = [j for j in env.joint_axes]
    RTS = gen_grad_RTS_2D_Layer(env.link_zonos, joint_axes, env.n_links, env.n_obs, params)

    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys()])
        # k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
        lam = torch.tensor([0.8, 0.8])
        #bias1 = torch.full((30, 1), 0.0, requires_grad=True)
        lam, FO_link, flag = RTS(torch.vstack([lam] * 2), torch.vstack([observ_temp] * 2))
        #lam.sum().backward()

        # ka, FO_link, flag = RTS(k,observ_temp)
        print(f'action: {lam}')
        print(f'flag: {flag}')

        t_elasped = time.time() - ts
        print(f'Time elasped for ARMTD-2d:{t_elasped}')
        t_armtd += t_elasped
        # print(ka[0])
        observation, reward, done, info = env.step(lam[0] * torch.pi / 24, flag[0])

        FO_link = [fo[0] for fo in FO_link]
        env.render(FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''

    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    import pdb;

    pdb.set_trace()
