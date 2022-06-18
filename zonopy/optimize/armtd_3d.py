import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import cyipopt

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
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*env.n_links)

    def generate_combinations_upto(self):
        self.combs = [torch.tensor([0])]
        for i in range(self.max_combs):
            self.combs.append(torch.combinations(torch.arange(i+1),2))

    def prepare_constraints(self,qpos,qvel,obstacles):
        _, R_trig = zp.process_batch_JRS_trig(self.JRS_tensor,qpos,qvel,self.joint_axes)
        #_, R_trig = zp.load_batch_JRS_trig(qpos,qvel)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params) # NOTE: zono_order
        self.A = [[] for _ in range(self.n_links)]
        self.b = [[] for _ in range(self.n_links)]
        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))
        
        for j in range(self.n_links):
            for o in range(self.n_obs):                
                obs_Z = obstacles[o].Z.unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension  
                self.A[j].append(A)
                self.b[j].append(b)
        self.qpos = qpos
        self.qvel = qvel

    def trajopt(self,qgoal,ka_0):
        # NOTE: torch OR numpy ?

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
                ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                if (p.x_prev!=x).any():                
                    cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs+2*self.n_links)                   
                    grad_cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs+2*self.n_links,self.n_links)
                    # velocity min max constraints
                    possible_max_min_q_dot = torch.vstack((self.qvel,self.qvel+x*T_PLAN,torch.zeros_like(self.qvel)))
                    q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                    q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                    grad_q_max = torch.diag(T_PLAN*(q_dot_max_idx%2))
                    grad_q_min = torch.diag(T_PLAN*(q_dot_min_idx%2))
                    cons_obs[-2*self.n_links:] = torch.hstack((q_dot_max,q_dot_min))
                    grad_cons_obs[-2*self.n_links:] = torch.vstack((grad_q_max,grad_q_min))
                    # velocity min max constraints
                    for j in range(self.n_links):
                        c_k = self.FO_link[j].center_slice_all_dep(ka/self.g_ka)
                        grad_c_k = self.FO_link[j].grad_center_slice_all_dep(ka/self.g_ka)/self.g_ka
                        for o in range(self.n_obs):
                            cons, ind = torch.max((self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                            grad_cons = (self.A[j][o].gather(-2,ind.reshape(self.n_timesteps,1,1).repeat(1,1,self.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                            cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = cons
                            grad_cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = grad_cons
                    p.cons_obs = cons_obs.numpy()
                    p.grad_cons_obs = grad_cons_obs.numpy()
                    p.x_prev = np.copy(x)                
                return p.cons_obs

            def jacobian(p,x):
                ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                if (p.x_prev!=x).any():                
                    cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs+2*self.n_links)                   
                    grad_cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs+2*self.n_links,self.n_links)
                    # velocity min max constraints
                    possible_max_min_q_dot = torch.vstack((self.qvel,self.qvel+x*T_PLAN,torch.zeros_like(self.qvel)))
                    q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                    q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                    grad_q_max = torch.diag(T_PLAN*(q_dot_max_idx%2))
                    grad_q_min = torch.diag(T_PLAN*(q_dot_min_idx%2))
                    cons_obs[-2*self.n_links:] = torch.hstack((q_dot_max,q_dot_min))
                    grad_cons_obs[-2*self.n_links:] = torch.vstack((grad_q_max,grad_q_min))
                    # velocity min max constraints
                    for j in range(self.n_links):
                        c_k = self.FO_link[j].center_slice_all_dep(ka/self.g_ka)
                        grad_c_k = self.FO_link[j].grad_center_slice_all_dep(ka/self.g_ka)/self.g_ka
                        for o in range(self.n_obs):
                            cons, ind = torch.max((self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                            grad_cons = (self.A[j][o].gather(-2,ind.reshape(self.n_timesteps,1,1).repeat(1,1,self.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                            cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = cons
                            grad_cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = grad_cons
                    p.cons_obs = cons_obs.numpy()
                    p.grad_cons_obs = grad_cons_obs.numpy()
                    p.x_prev = np.copy(x)                   
                return p.grad_cons_obs
            
            def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
                pass
        
        M_obs = self.n_links*self.n_timesteps*self.n_obs
        M = M_obs+2*self.n_links

        nlp = cyipopt.problem(
        n = self.n_links,
        m = M,
        problem_obj=nlp_setup(),
        lb = [-self.g_ka]*self.n_links,
        ub = [self.g_ka]*self.n_links,
        cl = [1e-6]*M_obs+[-1e20]*self.n_links+[-torch.pi+1e-6]*self.n_links,
        cu = [1e20]*M_obs+[torch.pi-1e-6]*self.n_links+[1e20]*self.n_links,
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
        self.prepare_constraints(env.qpos,env.qvel,env.obs_zonos)
        k_opt, flag = self.trajopt(env.qgoal,ka_0)
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
        env.render(planner.FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''
        #import pdb;pdb.set_trace()
        #(env.safe_con.numpy() - planner.safe_con > 1e-2).any()
    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    import pdb;pdb.set_trace()