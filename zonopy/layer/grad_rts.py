import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import zonopy as zp
import cyipopt

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.linalg import block_diag

import os
from torch.multiprocessing import Pool


def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi


T_PLAN, T_FULL = 0.5, 1.0

NUM_PROCESSES = 48


def rts_pass(FO_link, As, bs, qpos, qvel, qgoal, n_timesteps, n_links, n_obs, dimension, g_ka, ka_0, lambd_hat, idx):
    M_obs = n_timesteps * n_links * n_obs
    M = M_obs + 2 * n_links

    class nlp_setup():
        x_prev = np.zeros(n_links) * np.nan

        def objective(nlp, x):
            qplan = qpos + qvel * T_PLAN + 0.5 * g_ka * x * T_PLAN ** 2
            return torch.sum(wrap_to_pi(qplan - qgoal) ** 2)

        def gradient(nlp, x):
            qplan = qpos + qvel * T_PLAN + 0.5 * g_ka * x * T_PLAN ** 2
            return (g_ka * T_PLAN ** 2 * wrap_to_pi(qplan - qgoal)).numpy()

        def constraints(nlp, x):
            ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
            if (nlp.x_prev != x).any():
                Cons = torch.zeros(M)
                Jac = torch.zeros(M, n_links)
                # velocity constraints
                q_peak = qvel + g_ka * x * T_PLAN
                grad_q_peak = g_ka * T_PLAN * torch.eye(n_links)
                Cons[-2 * n_links:] = torch.hstack((q_peak - torch.pi, -torch.pi - q_peak))
                Jac[-2 * n_links:] = torch.vstack((grad_q_peak, -grad_q_peak))
                # velocity constraints 
                for j in range(n_links):
                    c_k = FO_link[j][idx].center_slice_all_dep(ka)
                    grad_c_k = FO_link[j][idx].grad_center_slice_all_dep(ka)
                    for o in range(n_obs):
                        cons, ind = torch.max((As[j][o][idx] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][idx],-1)  # shape: n_timsteps, SAFE if >=1e-6
                        jac = (As[j][o][idx].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1, dimension)) @ grad_c_k).squeeze(-2)  # shape: n_timsteps, n_links safe if >=1e-6
                        Cons[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = - cons
                        Jac[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = - jac
                nlp.cons = Cons.numpy()
                nlp.jac = Jac.numpy()
                nlp.x_prev = np.copy(x)
            return nlp.cons

        def jacobian(nlp, x):
            ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
            if (nlp.x_prev != x).any():
                Cons = torch.zeros(M)
                Jac = torch.zeros(M, n_links)
                # velocity constraints
                q_peak = qvel + g_ka * x * T_PLAN
                grad_q_peak = g_ka * T_PLAN * torch.eye(n_links)
                Cons[-2 * n_links:] = torch.hstack((q_peak - torch.pi, -torch.pi - q_peak))
                Jac[-2 * n_links:] = torch.vstack((grad_q_peak, -grad_q_peak))
                # velocity constraints 
                for j in range(n_links):
                    c_k = FO_link[j][idx].center_slice_all_dep(ka)
                    grad_c_k = FO_link[j][idx].grad_center_slice_all_dep(ka)
                    for o in range(n_obs):
                        cons, ind = torch.max((As[j][o][idx] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][idx],
                                              -1)  # shape: n_timsteps, SAFE if >=1e-6
                        jac = (As[j][o][idx].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1,
                                                                                              dimension)) @ grad_c_k).squeeze(
                            -2)  # shape: n_timsteps, n_links safe if >=1e-6
                        Cons[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = - cons
                        Jac[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = - jac
                nlp.cons = Cons.numpy()
                nlp.jac = Jac.numpy()
                nlp.x_prev = np.copy(x)
            return nlp.jac

        def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                         d_norm, regularization_size, alpha_du, alpha_pr,
                         ls_trials):
            pass

    nlp_obj = nlp_setup()
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
    print('ho!')
    k_opt, nlp_info = NLP.solve(ka_0)
    print('ho!!!')

    # NOTE: for training, dont care about fail-safe
    if nlp_info['status'] == 0:
        lambd_opt = torch.tensor(k_opt, dtype=torch.get_default_dtype())
        flag = 0
    else:
        lambd_opt = lambd_hat
        flag = 1

    return lambd_opt, flag, nlp_info, nlp_obj


def gen_grad_RTS_2D_Layer(link_zonos, joint_axes, n_links, n_obs, params, num_processes=NUM_PROCESSES):
    jrs_tensor = preload_batch_JRS_trig()
    dimension = 2
    n_timesteps = 100
    ka_0 = torch.zeros(n_links)
    PI_vel = torch.tensor(torch.pi - 1e-6)
    g_ka = torch.pi / 24

    class grad_RTS_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            zp.reset()
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            ctx.lambd = lambd.clone().reshape(-1, n_links).to(dtype=torch.get_default_dtype())
            # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=torch.get_default_dtype())
            ka = g_ka * ctx.lambd

            n_batches = observation.shape[0]
            qpos = observation[:, :n_links]
            qvel = observation[:, n_links:2 * n_links]
            obstacle_pos = observation[:, -4 * n_obs:-2 * n_obs]
            obstacle_size = observation[:, -2 * n_obs:]
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            # g_ka = torch.maximum(PI/24,abs(qvel/3))

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
            FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)

            As = [[] for _ in range(n_links)]
            bs = [[] for _ in range(n_links)]

            lambda_to_slc = ctx.lambd.reshape(n_batches, 1, dimension).repeat(1, n_timesteps, 1)

            # unsafe_flag = torch.zeros(n_batches)
            unsafe_flag = (abs(qvel + ctx.lambd * g_ka * T_PLAN) > PI_vel).any(
                -1)  # NOTE: this might not work on gpu, velocity lim check
            for j in range(n_links):
                FO_link[j] = FO_link[j].project([0, 1])
                c_k = FO_link[j].center_slice_all_dep(lambda_to_slc).unsqueeze(-1)  # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:, 2 * o:2 * (o + 1)].unsqueeze(-2),
                                       torch.diag_embed(obstacle_size[:, 2 * o:2 * (o + 1)])), -2).unsqueeze(-3).repeat(
                        1, n_timesteps, 1, 1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z, FO_link[j].Grest),
                                                             -2)).polytope()  # A: n_timesteps,*,dimension
                    As[j].append(A_temp)
                    bs[j].append(b_temp)
                    unsafe_flag += (torch.max((A_temp @ c_k).squeeze(-1) - b_temp, -1)[0] < 1e-6).any(
                        -1)  # NOTE: this might not work on gpu FOR, safety check

            unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rts all ways

            M_obs = n_timesteps * n_links * n_obs
            M = M_obs + 2 * n_links
            ctx.flags = -torch.ones(n_batches, dtype=torch.int)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            ctx.nlp_obj = [None for _ in range(n_batches)]
            ctx.nlp_info = [None for _ in range(n_batches)]

            rts_pass_indices = unsafe_flag.nonzero().reshape(-1)
            n_problems = rts_pass_indices.numel()
            if n_problems > 0:
                with Pool(processes=min(num_processes, n_problems)) as pool:
                    results = pool.starmap(
                        rts_pass,
                        [(FO_link, As, bs) + x for x in
                         zip(
                             qpos[rts_pass_indices],
                             qvel[rts_pass_indices],
                             qgoal[rts_pass_indices],
                             [n_timesteps] * n_problems,
                             [n_links] * n_problems,
                             [n_obs] * n_problems,
                             [dimension] * n_problems,
                             [g_ka] * n_problems,
                             [ka_0] * n_problems,
                             ctx.lambd[rts_pass_indices],
                             rts_pass_indices
                         )
                         ]
                    )
                rts_lambd_opt, rts_flags = [], []
                for idx, res in enumerate(results):
                    rts_lambd_opt.append(res[0])
                    rts_flags.append(res[1])
                    ctx.nlp_obj[rts_pass_indices[idx]] = res[2]
                    ctx.nlp_info[rts_pass_indices[idx]] = res[3]

                ctx.lambd[rts_pass_indices] = torch.cat(rts_lambd_opt, 0).view(n_problems, dimension)
                ctx.flags[rts_pass_indices] = torch.tensor(rts_flags, dtype=ctx.flags.dtype)

            return ctx.lambd, FO_link, ctx.flags

        @staticmethod
        def backward(ctx, *grad_ouput):
            direction = grad_ouput[0]
            grad_input = torch.zeros_like(direction)
            # COMPUTE GRADIENT
            tol = 1e-6
            # direct pass
            direct_pass = ctx.flags == -1
            grad_input[direct_pass] = torch.tensor(direction)[direct_pass]

            rts_success_pass = (ctx.flags == 0).nonzero().reshape(-1)
            n_batch = rts_success_pass.numel()
            if n_batch > 0:
                QP_EQ_CONS = []
                QP_INEQ_CONS = []
                for i in rts_success_pass:
                    k_opt = ctx.lambd[i].cpu().numpy()
                    # compute jacobian of each smooth constraint which will be constraints for QP
                    jac = ctx.nlp_obj[i].jacobian(k_opt)
                    cons = ctx.nlp_obj[i].cons

                    qp_cons1 = jac  # [A*c(k)-b].T*lambda  and vel. lim # NOTE
                    EYE = np.eye(n_links)
                    qp_cons4 = -EYE  # lb
                    qp_cons5 = EYE  # ub
                    qp_cons = np.vstack((qp_cons1, qp_cons4, qp_cons5))

                    # compute duals for smooth constraints                
                    mult_smooth_cons1 = ctx.nlp_info[i]['mult_g'] * (ctx.nlp_info[i]['mult_g'] > tol)
                    mult_smooth_cons4 = ctx.nlp_info[i]['mult_x_L'] * (ctx.nlp_info[i]['mult_x_L'] > tol)
                    mult_smooth_cons5 = ctx.nlp_info[i]['mult_x_U'] * (ctx.nlp_info[i]['mult_x_U'] > tol)
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
                grad_input[rts_success_pass] = torch.tensor(z.X.reshape(n_batch, n_links))

                # NOTE: for fail-safe, keep into zeros             
            return (grad_input.reshape(ctx.lambd_shape), torch.zeros(ctx.obs_shape))

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
        bias1 = torch.full((30, 1), 0.0, requires_grad=True)
        lam, FO_link, flag = RTS(torch.vstack([lam] * 30) + bias1, torch.vstack([observ_temp] * 30) + bias1)
        lam.sum().backward()

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
