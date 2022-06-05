import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0

# batch

def gen_RTS_star_2D_Layer(link_zonos,n_links,n_obs,params):
    jrs_tensor = preload_batch_JRS_trig()
    dimension = 2
    n_timesteps = 100
    ka_0 = torch.zeros(n_links)
    PI = torch.tensor(torch.pi)
    joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
    zono_order=40
    class RTS_star_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx,ka,observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            qpos = observation[:n_links]
            qvel = observation[n_links:2*n_links]
            obstacle_pos = observation[-4*n_obs:-2*n_obs]
            obstacle_size = observation[-2*n_obs:]
            qgoal = qpos + qvel*T_PLAN + 0.5*ka*T_PLAN**2
            g_ka = torch.maximum(PI/24,abs(qvel/3))
            _, R_trig = process_batch_JRS_trig(jrs_tensor,qpos,qvel,joint_axes)
            FO_link,_,_ = forward_occupancy(R_trig,link_zonos,params)
            As = [[] for _ in range(n_links)]
            bs = [[] for _ in range(n_links)]
            for j in range(n_links):
                FO_link[j] = FO_link[j].project([0,1])
                for o in range(n_obs):
                    obs_Z = torch.vstack((obstacle_pos[2*o:2*(o+1)],torch.diag(obstacle_size[2*o:2*(o+1)]))).unsqueeze(0).repeat(n_timesteps,1,1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension
                    As[j].append(A_temp)
                    bs[j].append(b_temp)
            M_obs = n_timesteps*n_links*n_obs
            M = M_obs+n_links

            class nlp_setup():
                x_prev = np.zeros(n_links)*np.nan
                def objective(nlp,x):
                    qplan = qpos + qvel*T_PLAN + 0.5*x*T_PLAN**2
                    return torch.sum(wrap_to_pi(qplan-qgoal)**2)

                def gradient(nlp,x):
                    qplan = qpos + qvel*T_PLAN + 0.5*x*T_PLAN**2
                    return (T_PLAN**2*wrap_to_pi(qplan-qgoal)).numpy()

                def constraints(nlp,x): 
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                    if (nlp.x_prev!=x).any():                
                        cons_obs = torch.zeros(M)                   
                        grad_cons_obs = torch.zeros(M,n_links)
                        cons_obs[-n_links:] = qvel+x*T_PLAN
                        grad_cons_obs[-n_links:] = torch.eye(n_links)*T_PLAN
                        for j in range(n_links):
                            c_k = FO_link[j].center_slice_all_dep(ka/g_ka)
                            grad_c_k = FO_link[j].grad_center_slice_all_dep(ka/g_ka)@torch.diag(1/g_ka)
                            for o in range(n_obs):
                                cons, ind = torch.max((As[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                                grad_cons = (As[j][o].gather(-2,ind.reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = cons
                                grad_cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = grad_cons
                        nlp.cons_obs = cons_obs.numpy()
                        nlp.grad_cons_obs = grad_cons_obs.numpy()
                        nlp.x_prev = np.copy(x)                
                    return nlp.cons_obs

                def jacobian(nlp,x):
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                    if (nlp.x_prev!=x).any():                
                        cons_obs = torch.zeros(M)                   
                        grad_cons_obs = torch.zeros(M,n_links)
                        cons_obs[-n_links:] = qvel+x*T_PLAN
                        grad_cons_obs[-n_links:] = torch.eye(n_links)*T_PLAN
                        for j in range(n_links):
                            c_k = FO_link[j].center_slice_all_dep(ka/g_ka)
                            grad_c_k = FO_link[j].grad_center_slice_all_dep(ka/g_ka)@torch.diag(1/g_ka)
                            for o in range(n_obs):
                                cons, ind = torch.max((As[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                                grad_cons = (As[j][o].gather(-2,ind.reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = cons
                                grad_cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = grad_cons
                        nlp.cons_obs = cons_obs.numpy()
                        nlp.grad_cons_obs = grad_cons_obs.numpy()
                        nlp.x_prev = np.copy(x)                   
                    return nlp.grad_cons_obs
                
                def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                        d_norm, regularization_size, alpha_du, alpha_pr,
                        ls_trials):
                    pass
            
            NLP = cyipopt.problem(
            n = n_links,
            m = M,
            problem_obj=nlp_setup(),
            lb = (-g_ka).tolist(),
            ub = g_ka.tolist(),
            cl = [1e-6]*M_obs+[-torch.pi]*n_links,
            cu = [1e20]*M_obs+[torch.pi]*n_links,
            )
            NLP.addOption('sb', 'yes')
            NLP.addOption('print_level', 0)
            
            k_opt, info = NLP.solve(ka_0)
            if info['status'] != 0:
                k_opt = -qvel/(T_FULL-T_PLAN)
            return k_opt, FO_link

        @staticmethod 
        def backward(ctx,grad_ouput):
            return grad_ouput

    return RTS_star_2D_Layer.apply

if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time
    n_links = 2
    env = Arm_2D(n_links=n_links,n_obs=1)
    observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    
    t_armtd = 0
    params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
    RTS = gen_RTS_star_2D_Layer(env.link_zonos,env.n_links,env.n_obs,params)

    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys() ])
        k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
        ka, FO_link = RTS(k,observ_temp) 
        t_elasped = time.time()-ts
        print(f'Time elasped for ARMTD-2d:{t_elasped}')
        t_armtd += t_elasped
        
        observation, reward, done, info = env.step(torch.tensor(ka,dtype=torch.get_default_dtype()))
        #import pdb;pdb.set_trace()
        env.render(FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''
        #import pdb;pdb.set_trace()
        #(env.safe_con.numpy() - planner.safe_con > 1e-2).any()
    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    import pdb;pdb.set_trace()

