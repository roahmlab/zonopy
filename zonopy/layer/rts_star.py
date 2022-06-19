import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt
import os
from torch.multiprocessing import Process, Pool
import multiprocessing
from zonopy.conSet import PROPERTY_ID
import zonopy as zp
#torch.multiprocessing.set_start_method('spawn', force=True)
os.environ['OMP_NUM_THREADS'] = '2'

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0

NUM_PROCESSES = 48

def solve_rts_star(FO_link, As, bs, n_links, n_obs, dimension, qpos, qvel, qgoal, n_timesteps, M_obs, M, g_ka, ka_0, lambda_i ,i):
    # print(f"Solve rts star called! FO_link[0][i]={FO_link[0][i]}")
    class nlp_setup():
        x_prev = np.zeros(n_links) * np.nan
        num_iter = 0
        def objective(nlp, x):
            qplan = qpos + qvel * T_PLAN + 0.5 * x * T_PLAN ** 2
            return torch.sum(wrap_to_pi(qplan - qgoal) ** 2)

        def gradient(nlp, x):
            qplan = qpos + qvel * T_PLAN + 0.5 * x * T_PLAN ** 2
            return (T_PLAN ** 2 * wrap_to_pi(qplan - qgoal)).numpy()

        def constraints(nlp, x):
            ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
            if (nlp.x_prev != x).any():
                cons_obs = torch.zeros(M)
                grad_cons_obs = torch.zeros(M, n_links)
                # velocity min max constraints
                possible_max_min_q_dot = torch.vstack((qvel, qvel + x * T_PLAN, torch.zeros_like(qvel)))
                q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                grad_q_max = torch.diag(T_PLAN * (q_dot_max_idx % 2))
                grad_q_min = torch.diag(T_PLAN * (q_dot_min_idx % 2))
                cons_obs[-2 * n_links:] = torch.hstack((q_dot_max, q_dot_min))
                grad_cons_obs[-2 * n_links:] = torch.vstack((grad_q_max, grad_q_min))
                # velocity min max constraints
                for j in range(n_links):
                    c_k = FO_link[j][i].center_slice_all_dep(ka / g_ka)
                    grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka / g_ka) / g_ka
                    for o in range(n_obs):
                        cons, ind = torch.max((As[j][o][i] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i],
                                              -1)  # shape: n_timsteps, SAFE if >=1e-6
                        grad_cons = (As[j][o][i].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1,
                                                                                                  dimension)) @ grad_c_k).squeeze(
                            -2)  # shape: n_timsteps, n_links safe if >=1e-6
                        cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = cons
                        grad_cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = grad_cons
                nlp.cons_obs = cons_obs.numpy()
                nlp.grad_cons_obs = grad_cons_obs.numpy()
                nlp.x_prev = np.copy(x)
            return nlp.cons_obs

        def jacobian(nlp, x):
            nlp.num_iter += 1
            #print(f"Curr num_iter={nlp.num_iter}")


            ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
            if (nlp.x_prev != x).any():
                cons_obs = torch.zeros(M)
                grad_cons_obs = torch.zeros(M, n_links)
                # velocity min max constraints
                possible_max_min_q_dot = torch.vstack((qvel, qvel + x * T_PLAN, torch.zeros_like(qvel)))
                q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                grad_q_max = torch.diag(T_PLAN * (q_dot_max_idx % 2))
                grad_q_min = torch.diag(T_PLAN * (q_dot_min_idx % 2))
                cons_obs[-2 * n_links:] = torch.hstack((q_dot_max, q_dot_min))
                grad_cons_obs[-2 * n_links:] = torch.vstack((grad_q_max, grad_q_min))
                # velocity min max constraints
                for j in range(n_links):
                    #print(multiprocessing.current_process())
                    #print(f"j={j}, i={i}")
                    #print(f"FO_link[j][i]:{FO_link[j][i]}")
                    c_k = FO_link[j][i].center_slice_all_dep(ka / g_ka)
                    grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka / g_ka) / g_ka
                    for o in range(n_obs):
                        cons, ind = torch.max((As[j][o][i] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i],
                                              -1)  # shape: n_timsteps, SAFE if >=1e-6
                        grad_cons = (As[j][o][i].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1,
                                                                                                  dimension)) @ grad_c_k).squeeze(
                            -2)  # shape: n_timsteps, n_links safe if >=1e-6
                        cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = cons
                        grad_cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = grad_cons
                nlp.cons_obs = cons_obs.numpy()
                nlp.grad_cons_obs = grad_cons_obs.numpy()
                nlp.x_prev = np.copy(x)
            return nlp.grad_cons_obs

        def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                         d_norm, regularization_size, alpha_du, alpha_pr,
                         ls_trials):
            pass

    NLP = cyipopt.Problem(
        n=n_links,
        m=M,
        problem_obj=nlp_setup(),
        lb=[-g_ka] * n_links,
        ub=[g_ka] * n_links,
        cl=[1e-6] * M_obs + [-1e20] * n_links + [-torch.pi + 1e-6] * n_links,
        cu=[1e20] * M_obs + [torch.pi - 1e-6] * n_links + [1e20] * n_links,
    )
    NLP.add_option('sb', 'yes')
    NLP.add_option('print_level', 0)

    k_opt, info = NLP.solve(ka_0)

    # NOTE: for training, dont care about fail-safe
    if info['status'] == 0:
        l = torch.tensor(k_opt, dtype=torch.get_default_dtype()) / g_ka
        flag = 0
    else:
        l = lambda_i
        flag = 1
    return l, flag
# batch

def gen_RTS_star_2D_Layer(link_zonos,joint_axes,n_links,n_obs,params, num_processes=NUM_PROCESSES):
    jrs_tensor = preload_batch_JRS_trig()
    dimension = 2
    n_timesteps = 100
    ka_0 = torch.zeros(n_links)
    PI_vel = torch.tensor(torch.pi-1e-6)
    zono_order=40
    g_ka = torch.pi/24
    class RTS_star_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx,lambd,observation):
            # zp.reset()
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            lambd =lambd.reshape(-1,n_links).to(dtype=torch.get_default_dtype()) 
            #observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=torch.get_default_dtype())
            ka = g_ka*lambd
            
            n_batches = observation.shape[0]
            qpos = observation[:,:n_links]
            qvel = observation[:,n_links:2*n_links]
            obstacle_pos = observation[:,-4*n_obs:-2*n_obs]
            obstacle_size = observation[:,-2*n_obs:]
            qgoal = qpos + qvel*T_PLAN + 0.5*ka*T_PLAN**2

            #g_ka = torch.maximum(PI/24,abs(qvel/3))

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor,qpos,qvel,joint_axes)
            FO_link,_,_ = forward_occupancy(R_trig,link_zonos,params)
            
            As = [[] for _ in range(n_links)]
            bs = [[] for _ in range(n_links)]

            lambda_to_slc = lambd.reshape(n_batches,1,dimension).repeat(1,n_timesteps,1)
            
            #unsafe_flag = torch.zeros(n_batches) 
            unsafe_flag = (abs(qvel+lambd*g_ka*T_PLAN)>PI_vel).any(-1)#NOTE: this might not work on gpu, velocity lim check
            for j in range(n_links):
                FO_link[j] = FO_link[j].project([0,1]) 
                c_k = FO_link[j].center_slice_all_dep(lambda_to_slc).unsqueeze(-1) # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:,2*o:2*(o+1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:,2*o:2*(o+1)])),-2).unsqueeze(-3).repeat(1,n_timesteps,1,1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension                     
                    As[j].append(A_temp)
                    bs[j].append(b_temp)
                    unsafe_flag += (torch.max((A_temp@c_k).squeeze(-1)-b_temp,-1)[0]<1e-6).any(-1)  #NOTE: this might not work on gpu FOR, safety check

            M_obs = n_timesteps*n_links*n_obs
            M = M_obs+2*n_links
            flags = -torch.ones(n_batches) # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            nonzero_flag_indices = unsafe_flag.nonzero().reshape(-1)

            num_problems = nonzero_flag_indices.numel()
            # solve_rts_star(n_links, n_obs, dimension, qpos, qvel, qgoal, n_timesteps, M_obs, M, g_ka, As, bs, ka_0, FO_link, lambda_i)
            if num_problems > 0:
                #print(f"Before mp call, FO link[0][{nonzero_flag_indices[0]}] is {FO_link[0][nonzero_flag_indices[0]]}")
                #print(f"non zero flag indices: {nonzero_flag_indices}")
                #print(f"unsafe flagas:{unsafe_flag}")
                #print("Entered mp call...")
                pool = Pool(processes=min(num_processes, num_problems))
                results = pool.starmap(solve_rts_star,
                                       [(FO_link, As, bs) + x for x in
                                       zip([n_links] * num_problems,
                                           [n_obs] * num_problems,
                                           [dimension] * num_problems,
                                           qpos[nonzero_flag_indices],
                                           qvel[nonzero_flag_indices],
                                           qgoal[nonzero_flag_indices],
                                           [n_timesteps] * num_problems,
                                           [M_obs] * num_problems,
                                           [M] * num_problems,
                                           [g_ka] * num_problems,
                                           [ka_0] * num_problems,
                                           lambd[nonzero_flag_indices],
                                           nonzero_flag_indices
                                           )
                                    ]
                                    )
                #print("mp call succeeded")
                nonzero_indices_lambdas = torch.cat([result[0] for result in results], 0).view(num_problems,dimension)
                nonzero_indices_flags = torch.tensor([result[1] for result in results])
                lambd[nonzero_flag_indices] = nonzero_indices_lambdas
                flags[nonzero_flag_indices] = nonzero_indices_flags.to(flags.dtype)
            return lambd, FO_link, flags

        @staticmethod 
        def backward(ctx,grad_ouput):
            return (torch.zeros(ctx.lambd_shape),torch.zeros(ctx.obs_shape))

    return RTS_star_2D_Layer.apply

if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time
    n_links = 2
    env = Arm_2D(n_links=n_links,n_obs=1)
    observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    
    t_armtd = 0
    params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
    joint_axes = [j for j in env.joint_axes]
    RTS = gen_RTS_star_2D_Layer(env.link_zonos,joint_axes,env.n_links,env.n_obs,params)

    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys() ])
        #k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
        lam = torch.tensor([0.8,0.8])
        lam, FO_link, flag = RTS(torch.vstack(([lam] * 40)),torch.vstack(([observ_temp]*40)))
        #ka, FO_link, flag = RTS(k,observ_temp)
        print(f'action: {lam}')
        print(f'flag: {flag}')
            
        t_elasped = time.time()-ts
        print(f'Time elasped for ARMTD-2d:{t_elasped}')
        t_armtd += t_elasped
        #print(ka[0])
        observation, reward, done, info = env.step(lam[0]*torch.pi/24,flag[0])
        
        FO_link = [fo[0] for fo in FO_link]
        env.render(FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''


    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')

