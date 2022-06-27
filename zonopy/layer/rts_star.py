import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt
import os
from torch.multiprocessing import Pool
import zonopy as zp

from zonopy.layer.nlp_setup import NLP_setup

# torch.multiprocessing.set_start_method('spawn', force=True)
os.environ['OMP_NUM_THREADS'] = '2'

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
    return lambd_opt, flag, info


# batch

def gen_RTS_star_2D_Layer(link_zonos, joint_axes, n_links, n_obs, params, num_processes=NUM_PROCESSES, dtype = torch.float, device='cpu'):
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 2
    n_timesteps = 100
    #ka_0 = np.zeros(n_links)
    PI_vel = torch.tensor(torch.pi - 1e-6,dtype=dtype, device=device)
    zono_order = 40
    g_ka = torch.pi / 24

    class RTS_star_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            zp.reset()
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            lambd = lambd.clone().reshape(-1, n_links).to(dtype=dtype,device=device)
            # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=dtype,device=device)
            ka = g_ka * lambd

            n_batches = observation.shape[0]
            qpos = observation[:, :n_links]
            qvel = observation[:, n_links:2 * n_links]
            obstacle_pos = observation[:, -4 * n_obs:-2 * n_obs]
            obstacle_size = observation[:, -2 * n_obs:]
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            # g_ka = torch.maximum(PI/24,abs(qvel/3))

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
            batch_FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)

            As = [[[] for _ in range(n_links)] for _ in range(n_batches)]
            bs = [[[] for _ in range(n_links)] for _ in range(n_batches)]
            FO_links = [[] for _ in range(n_batches)]
            lambda_to_slc = lambd.reshape(n_batches, 1, dimension).repeat(1, n_timesteps, 1)

            # unsafe_flag = torch.zeros(n_batches)
            unsafe_flag = (abs(qvel + lambd * g_ka * T_PLAN) > PI_vel).any(-1)  # NOTE: this might not work on gpu, velocity lim check
            lambd0 = lambd.clamp((-PI_vel-qvel)/(g_ka *T_PLAN),(PI_vel-qvel)/(g_ka *T_PLAN)).cpu().numpy()
            for j in range(n_links):
                FO_link_temp = batch_FO_link[j].project([0, 1])
                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).unsqueeze(-1)  # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:, 2 * o:2 * (o + 1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:, 2 * o:2 * (o + 1)])), -2).unsqueeze(-3).repeat(1, n_timesteps, 1, 1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z, FO_link_temp.Grest),-2)).polytope()  # A: n_timesteps,*,dimension
                    unsafe_flag += (torch.max((A_temp @ c_k).squeeze(-1) - b_temp, -1)[0] < 1e-6).any(-1)  # NOTE: this might not work on gpu FOR, safety check
                for b in range(n_batches):
                    As[b][j].append(A_temp[b].cpu().numpy())
                    bs[b][j].append(b_temp[b].cpu().numpy())
                    FO_links[b].append(FO_link_temp[b].cpu())

            #unsafe_flag = torch.ones(n_batches)
            flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            infos = [None for _ in range(n_batches)]
            rts_pass_indices = unsafe_flag.nonzero().reshape(-1)

            n_problems = rts_pass_indices.numel()
            # rts_pass(As, bs, FO_links, qpos, qvel, qgoal, n_timesteps, n_links, n_obs, dimension, g_ka, ka_0, lambd_hat, rts_pass_indices)
            if n_problems > 0:
                with Pool(processes=min(num_processes, n_problems)) as pool:
                    results = pool.starmap(
                        rts_pass,
                        [x for x in
                         zip([As[idx] for idx in rts_pass_indices],
                             [bs[idx] for idx in rts_pass_indices],
                             [FO_links[idx] for idx in rts_pass_indices],
                             qpos.cpu().numpy()[rts_pass_indices],
                             qvel.cpu().numpy()[rts_pass_indices],
                             qgoal.cpu().numpy()[rts_pass_indices],
                             [n_timesteps] * n_problems,
                             [n_links] * n_problems,
                             [n_obs] * n_problems,
                             [dimension] * n_problems,
                             [g_ka] * n_problems,
                             [lambd0[idx] for idx in rts_pass_indices],  #[ka_0] * n_problems,
                             lambd.cpu()[rts_pass_indices]
                         )
                         ]
                    )
                rts_lambd_opt, rts_flags = [], []
                for idx, res in enumerate(results):
                    rts_lambd_opt.append(res[0])
                    rts_flags.append(res[1])
                    infos[rts_pass_indices[idx]] = res[2]
                lambd[rts_pass_indices] = torch.cat(rts_lambd_opt, 0).view(n_problems, dimension).to(dtype=dtype,device=device)
                flags[rts_pass_indices] = torch.tensor(rts_flags, dtype=flags.dtype, device=device)
            return lambd, FO_links, flags, infos

        @staticmethod
        def backward(ctx, *grad_ouput):
            return (torch.zeros(ctx.lambd_shape,dtype=dtype,device=device), torch.zeros(ctx.obs_shape,dtype=dtype,device=device))

    return RTS_star_2D_Layer.apply


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
    RTS = gen_RTS_star_2D_Layer(env.link_zonos, joint_axes, env.n_links, env.n_obs, params)

    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys()])
        # k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
        lam = torch.tensor([0.8, 0.8])
        lam, FO_link, flag = RTS(torch.vstack(([lam] * 3)), torch.vstack(([observ_temp] * 3)))
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
