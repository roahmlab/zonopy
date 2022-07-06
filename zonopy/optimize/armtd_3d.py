import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import cyipopt


import time

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0
# NOTE: optimize ka instead of lambda
class ARMTD_3D_planner():
    def __init__(self,env,zono_order=40,max_combs=200):
        self.wrap_env(env)
        self.n_timesteps = 100
        self.eps = 1e-6
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.generate_combinations_upto()
        self.PI = torch.tensor(torch.pi)
        self.JRS_tensor = zp.preload_batch_JRS_trig()
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def wrap_env(self,env):
        assert env.dimension == 3
        self.dimension = 3
        self.n_links = env.n_links
        self.n_obs = env.n_obs
        self.link_zonos = env.link_zonos
        self.params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
        self.joint_axes = env.joint_axes
        self.vel_lim =  env.vel_lim 
        self.pos_lim = env.pos_lim 
        self.actual_pos_lim = env.pos_lim[env.lim_flag]
        self.n_pos_lim = int(env.lim_flag.sum())
        self.lim_flag = env.lim_flag
    def generate_combinations_upto(self):
        self.combs = [torch.tensor([0])]
        for i in range(self.max_combs):
            self.combs.append(torch.combinations(torch.arange(i+1),2))

    def prepare_constraints(self,qpos,qvel,obstacles):
        _, R_trig = zp.process_batch_JRS_trig(self.JRS_tensor,qpos,qvel,self.joint_axes)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params) # NOTE: zono_order
        self.A = [[] for _ in range(self.n_links)]
        self.b = [[] for _ in range(self.n_links)]
        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))
        
        for j in range(self.n_links):
            for o in range(self.n_obs):                
                obs_Z = obstacles[o].Z.unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                self.A[j].append(A)
                self.b[j].append(b)
        self.qpos = qpos
        self.qvel = qvel

    def trajopt(self,qgoal,ka_0):
        # NOTE: torch OR numpy ?

        M_obs = self.n_links*self.n_timesteps*self.n_obs
        M = M_obs+2*self.n_links+6*self.n_pos_lim

        class nlp_setup():
            x_prev = np.zeros(self.n_links)*np.nan
            def objective(p,x):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*x*T_PLAN**2
                return torch.sum(wrap_to_pi(qplan-qgoal)**2)

            def gradient(p,x):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*x*T_PLAN**2
                qplan_grad = 0.5*T_PLAN**2
                return (2*qplan_grad*wrap_to_pi(qplan-qgoal)).numpy()

            def constraints(p,x): 
                p.compute_constraints_jacobian(x)      
                return p.Cons

            def jacobian(p,x):
                p.compute_constraints_jacobian(x)
                return p.Jac

            def compute_constraints_jacobian(p,x):
                if (p.x_prev!=x).any():                
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                    Cons = torch.zeros(M)      
                    Jac = torch.zeros(M,self.n_links)
                    # position and velocity constraints
                    t_peak_optimum = -self.qvel/(self.g_ka*ka[0]) # time to optimum of first half traj.
                    qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*(self.g_ka*ka[0])*t_peak_optimum**2).nan_to_num()
                    grad_qpos_peak_optimum = torch.diag((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.g_ka*t_peak_optimum**2).nan_to_num())
                    qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * (self.g_ka * ka[0]) * T_PLAN**2
                    grad_qpos_peak = 0.5 * self.g_ka * T_PLAN**2 * torch.eye(self.n_links)
                    qvel_peak = self.qvel + self.g_ka * ka[0] * T_PLAN
                    grad_qvel_peak = self.g_ka * T_PLAN * torch.eye(self.n_links)

                    bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
                    qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*T_PLAN**2
                    grad_qpos_brake = 0.5 * self.g_ka * T_PLAN * T_FULL * torch.eye(self.n_links) # NOTE: need to verify equation

                    qpos_possible_max_min = torch.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,self.lim_flag] 
                    qpos_ub = (qpos_possible_max_min - self.actual_pos_lim[:,0]).flatten()
                    qpos_lb = (self.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
                    
                    grad_qpos_ub = torch.vstack((grad_qpos_peak_optimum[self.lim_flag],grad_qpos_peak[self.lim_flag],grad_qpos_brake[self.lim_flag]))
                    grad_qpos_lb = - grad_qpos_ub

                    Cons[M_obs:] = torch.hstack((qvel_peak-self.vel_lim, -self.vel_lim-qvel_peak,qpos_ub,qpos_lb))
                    Jac[M_obs:] = torch.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))                    

                    for j in range(self.n_links):
                        c_k = self.FO_link[j].center_slice_all_dep(ka/self.g_ka)
                        grad_c_k = self.FO_link[j].grad_center_slice_all_dep(ka/self.g_ka)/self.g_ka
                        for o in range(self.n_obs):
                            h_obs = (self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o]
                            cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1) # shape: n_timsteps, SAFE if >=1e-6 
                            grad_obs = (self.A[j][o].gather(-2,ind.reshape(self.n_timesteps,1,1).repeat(1,1,self.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                            Cons[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = - cons_obs
                            Jac[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = - grad_obs
                    p.Cons = Cons.numpy()
                    p.Jac = Jac.numpy()
                    p.x_prev = np.copy(x)   



            def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
                pass
        


        nlp = cyipopt.problem(
        n = self.n_links,
        m = M,
        problem_obj=nlp_setup(),
        lb = [-self.g_ka]*self.n_links,
        ub = [self.g_ka]*self.n_links,
        cl = [-1e20]*M,
        cu = [-1e-6]*M,
        )
        
        #nlp.add_option('mu_strategy', 'adaptive')
        #nlp.add_option('tol', 1e-7)

        nlp.addOption('sb', 'yes')
        nlp.addOption('print_level', 0)
        #ts = time.time()
        k_opt, info = nlp.solve(ka_0)
        #print(f'opt time: {time.time()-ts}')

        '''
        Problem = nlp_setup()
        self.safe_con = Problem.constraints(k_opt)[:-self.n_links]
        safe = self.safe_con>1e-6
        print(f'safe: {not any(~safe)}')
        print(f'safe distance: {self.safe_con.min()}')
        print(info['status'])

        if any(~safe):
            import pdb;pdb.set_trace()
        '''
        return k_opt, info['status']
        
    def plan(self,env,ka_0):
        zp.reset()
        t1 = time.time()
        self.prepare_constraints(env.qpos,env.qvel,env.obs_zonos)
        t2 = time.time()
        k_opt, flag = self.trajopt(env.qgoal,ka_0)
        t3 = time.time()
        print(f'FO time: {t2-t1}')
        print(f'NLP time: {t3-t2}')
        return k_opt, flag


if __name__ == '__main__':
    from zonopy.environments.arm_3d import Arm_3D
    import time
    #cyipopt.setLoggingLevel(1000)
    env = Arm_3D(n_obs=1)
    t_armtd = 0
    planner = ARMTD_3D_planner(env)
    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        ka, flag = planner.plan(env,torch.zeros(env.n_links))
        #print(env.qpos)
        t_elasped = time.time()-ts
        print(f'Time elasped for ARMTD-3d:{t_elasped}')
        t_armtd += t_elasped
        print(ka)
        env.step(torch.tensor(ka,dtype=torch.get_default_dtype()),flag)
        env.render()
        #env.render(planner.FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''
        #import pdb;pdb.set_trace()
        #(env.safe_con.numpy() - planner.safe_con > 1e-2).any()
    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    import pdb;pdb.set_trace()