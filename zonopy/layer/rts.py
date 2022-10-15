import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
#import os
import zonopy as zp

T_PLAN, T_FULL = 0.5, 1.0

def gen_RTS_2D_Layer(link_zonos, joint_axes, n_links, n_obs, params, dtype = torch.float, device=torch.device('cpu'), budget = 200, std = 0.3, goal_bias = 0.2):
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 2
    n_timesteps = 100
    #ka_0 = np.zeros(n_links)
    PI_vel = torch.tensor(torch.pi - 1e-6,dtype=dtype, device=device)
    ONE = torch.ones(1,dtype=dtype,device=device)
    g_ka = torch.pi / 24

    class RTS_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation):
            zp.reset()
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            lambd = lambd.clone().reshape(-1, n_links).to(dtype=dtype,device=device)
            # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=dtype,device=device)

            n_batches = observation.shape[0]
            qpos = observation[:, :n_links]
            qvel = observation[:, n_links:2 * n_links]
            obstacle_pos = observation[:, -4 * n_obs:-2 * n_obs]
            obstacle_size = observation[:, -2 * n_obs:]

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, joint_axes)
            batch_FO_link, _, _ = forward_occupancy(R_trig, link_zonos, params)

            As = np.zeros((n_links,n_obs),dtype=object)
            bs = np.zeros((n_links,n_obs),dtype=object)
            FO_links = np.zeros((n_links),dtype=object)
            lambda_to_slc = lambd.reshape(n_batches, 1, n_links).repeat(1, n_timesteps, 1)

            # unsafe_flag = torch.zeros(n_batches)
            unsafe_flag = (abs(qvel + lambd * g_ka * T_PLAN) > PI_vel).any(-1)
            for j in range(n_links):
                FO_link_temp = batch_FO_link[j].project([0, 1])
                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).unsqueeze(-1)  # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:, 2 * o:2 * (o + 1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:, 2 * o:2 * (o + 1)])), -2).unsqueeze(-3).repeat(1, n_timesteps, 1, 1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z, FO_link_temp.Grest),-2)).polytope()  # A: n_batches, n_timesteps, * ,dimension
                    h_obs = ((A_temp @ c_k).squeeze(-1) - b_temp).nan_to_num(-torch.inf)
                    unsafe_flag += (torch.max(h_obs, -1)[0] < 1e-6).any(-1)  # NOTE: this might not work on gpu FOR, safety check
                    As[j,o] = A_temp.unsqueeze(0).repeat(budget,1,1,1,1) # A: budget, n_batches, n_timesteps, * ,dimension
                    bs[j,o] = b_temp.unsqueeze(0).repeat(budget,1,1,1) # b: budget, n_batches, n_timesteps, dimension
                FO_links[j] = FO_link_temp
                #FO_links[j] = [fo for fo in FO_link_temp.cpu()]

            #unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rtd always
            rtd_pass_indices_tensor = unsafe_flag.nonzero().reshape(-1)
            rtd_pass_indices = rtd_pass_indices_tensor.tolist()
            n_problems = len(rtd_pass_indices)
            
            flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            
            if n_problems > 0:
                # Uncorrected actions
                lambd_unsafe = lambd[rtd_pass_indices]
                # Lower and upper bound for [-1,1] and velocity limit
                lb = torch.maximum((-PI_vel-qvel[rtd_pass_indices])/T_PLAN,-ONE)
                ub = torch.minimum((PI_vel-qvel[rtd_pass_indices])/T_PLAN,ONE)
                # Gaussian sample with clamp and Uniform sample
                Nsampler = torch.distributions.normal.Normal(lambd_unsafe,scale=std)
                Usampler = torch.distributions.uniform.Uniform(lb,ub)
                Nbudget = int(goal_bias*budget)
                Ubudget = budget - Nbudget

                lambda_candidates = torch.cat((Nsampler.sample((Nbudget,)).clamp(lb,ub),Usampler.sample((Ubudget,))),0)          
                lambda_candidates_to_slc = lambda_candidates.unsqueeze(-2).repeat(1,1,n_timesteps,1) # budget, n_batches, n_timesteps, dimension
                # Check safety of sampled actions
                rtd_flags = torch.zeros(budget,n_problems,dtype=bool,device=device) # (false) success, (true) fail
                for j in range(n_links):
                    FO_link_temp = FO_links[j]
                    FO_link_budgets = zp.batchPolyZonotope(FO_link_temp.Z[rtd_pass_indices].unsqueeze(0).repeat(budget,1,1,1,1), # Z: budget, n_problems, n_timesteps, * ,dimension
                                                        FO_link_temp.n_dep_gens,
                                                        FO_link_temp.expMat,
                                                        FO_link_temp.id,
                                                        compress=0)
                    c_k = FO_link_budgets.center_slice_all_dep(lambda_candidates_to_slc).unsqueeze(-1) 
                    for o in range(n_obs): 
                        h_obs = ((As[j,o][:,rtd_pass_indices] @ c_k).squeeze(-1) - bs[j,o][:,rtd_pass_indices]).nan_to_num(-torch.inf) 
                        rtd_flags += (torch.max(h_obs, -1)[0] < 1e-6).any(-1) 
                # Pick the closest safe action from the original action
                lambda_candidates[rtd_flags.to(dtype=bool)] = torch.inf # set infinity for unsafe action
                best_idx = torch.min(torch.linalg.norm(lambda_candidates-lambd_unsafe,dim=-1),0).indices
                lambda_best = lambda_candidates[best_idx,torch.arange(n_problems,device=device)] 
                # Identify indices for rtd success
                rtd_success = (~rtd_flags).any(0) # (true) success, (false) fail
                rtd_success_indices = rtd_pass_indices_tensor[rtd_success]
                # Parse rtd output

                lambd[rtd_success_indices] = lambda_best
                flags[rtd_pass_indices] = (~rtd_success).to(dtype=flags.dtype)

            zp.reset()
            return lambd, FO_links, flags

        @staticmethod
        def backward(ctx, *grad_ouput):
            return (torch.zeros(ctx.lambd_shape,dtype=dtype,device=device), torch.zeros(ctx.obs_shape,dtype=dtype,device=device))

    return RTS_2D_Layer.apply


if __name__ == '__main__':
    
    from zonopy.environments.arm_2d import Arm_2D
    from zonopy.environments.parallel_arm_2d import Parallel_Arm_2D
    import time
    from zonopy.conSet import PROPERTY_ID
    ##### 0. SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float
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
    rts = gen_RTS_2D_Layer(link_zonos, joint_axes, env.n_links, env.n_obs, params,device=device,dtype=dtype)

    ##### 3. RUN RTS #####
    t_forward = 0
    t_render = 0
    n_steps = 30
    print('='*90)
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].reshape(n_batch,-1) for key in observation.keys()])

        lam_hat = torch.tensor([0.8, 0.8],device=device,dtype=dtype)
        lam, FO_link, flag = rts(torch.vstack(([lam_hat] * n_batch)), observ_temp)
              
        print(f'action: {lam[0]}')
        print(f'flag: {flag[0]}')

        t_elasped = time.time() - ts
        print(f'Time elasped for RTS forward:{t_elasped}')
        print('='*90)
        t_forward += t_elasped
        observation, reward, done, info = env.step(lam.cpu().to(dtype=torch.get_default_dtype()) * torch.pi / 24, flag.cpu().to(dtype=torch.get_default_dtype()))
        
        env.render(FO_link)

    print(f'Total time elasped for RTS forward with {n_steps} steps: {t_forward}')
