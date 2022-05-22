import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import cyipopt

T_PLAN, T_FULL = 0.5, 1.0


class ARMTD_planner():
    def __init__(self,env,zono_order=40,max_combs=200):
        self.wrap_env(env)
        self.n_timesteps = 100
        self.eps = 1e-6
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.generate_combinations_upto()
        self.PI = torch.tensor(torch.pi)

    def wrap_env(self,env):
        self.dimension = env.dimension
        self.n_links = env.n_links
        self.n_obs = env.n_obs
        self.link_zonos = env.link_zonos
        self.params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*env.n_links)

    def generate_combinations_upto(self):
        self.combs = [torch.tensor([0])]
        for i in range(1,self.max_combs):
            self.combs.append(torch.combinations(torch.arange(i+1),2))

    def prepare_constraints(self,qpos,qvel,obstacles):
        _, R_trig = zp.load_batch_JRS_trig(qpos,qvel)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params)
        self.A = [[] for _ in range(self.n_links)]
        self.b = [[] for _ in range(self.n_links)]
        self.eval_slc_c_k = [[] for _ in range(self.n_links)]
        self.grad_slc_c_k = [[] for _ in range(self.n_links)]
        self.g_ka = torch.minimum(torch.maximum(self.PI/24,abs(qvel/3)),self.PI/3)
        
        for j in range(self.n_links):
            if self.dimension == 2:
                self.FO_link[j] = self.FO_link[j].project([0,1])
            for o in range(self.n_obs):                
                obs_Z = obstacles[o].Z[:,:self.dimension].unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                eval_slc_c_k = lambda ka:self.FO_link[j].center_slice_all_dep(ka/self.g_ka) # c_k: n_timesteps, dimension 
                grad_slc_c_k = lambda ka:self.FO_link[j].grad_center_slice_all_dep(ka/self.g_ka)@torch.diag(1/self.g_ka) # c_k: n_timesteps, dimension, n_joints 
                self.A[j].append(A)
                self.b[j].append(b)
                self.eval_slc_c_k[j].append(eval_slc_c_k)
                self.grad_slc_c_k[j].append(grad_slc_c_k)
        self.qpos = qpos
        self.qvel = qvel

    def trajopt(self,qgoal,ka_0):
        # NOTE: torch OR numpy ?
        class nlp_setup():
            def objective(p,ka):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
                return torch.sum((qplan-qgoal)**2)

            def gradient(p,ka):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
                qplan_grad = 0.5*T_PLAN**2
                return (2*qplan_grad*(qplan-qgoal)).numpy()

            def constraints(p,ka): 
                if isinstance(ka,np.ndarray):
                    ka = torch.tensor(ka,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs)
                for j in range(self.n_links):
                    for o in range(self.n_obs):
                        c_k = self.eval_slc_c_k[j][o](ka)
                        cons, _ = torch.max((self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                        cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = cons # shape: n_timsteps,
                return cons_obs.numpy()

            def jacobian(p,ka):
                if isinstance(ka,np.ndarray):
                    ka = torch.tensor(ka,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                grad_cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs,self.n_links)
                for j in range(self.n_links):
                    for o in range(self.n_obs):                        
                        c_k = self.eval_slc_c_k[j][o](ka)
                        _, ind = torch.max((self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o],-1) # shape: n_timsteps, SAFE if >=1e-6
                        grad_c_k = self.grad_slc_c_k[j][o](ka)    
                        grad_cons = (self.A[j][o].gather(-2,ind.reshape(self.n_timesteps,1,1).repeat(1,1,self.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                        #import pdb;pdb.set_trace()
                        grad_cons_obs[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = grad_cons
                return grad_cons_obs.numpy()
            
            def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
                pass

        M = self.n_links*self.n_timesteps*self.n_obs

        nlp = cyipopt.problem(
        n = self.n_links,
        m = M,
        problem_obj=nlp_setup(),
        lb = (-self.g_ka).tolist(),
        ub = self.g_ka.tolist(),
        cl = [1e-6]*M,
        cu = [1e20]*M,
        )
        #nlp.add_option('mu_strategy', 'adaptive')
        #nlp.addOption('mu_strategy', 'adaptive')
        
        #nlp.add_option('tol', 1e-7)
        ts = time.time()
        k_opt, info = nlp.solve(ka_0)
        print(f'opt time: {time.time()-ts}')

        Problem = nlp_setup()
        safe = Problem.constraints(k_opt)>1e-6 
        print(f'safe: {not any(~safe)}')
        print(f'safe distance: {Problem.constraints(k_opt).min()}')
        print(info['status'])
        '''
        if any(~safe):
            import pdb;pdb.set_trace()
        '''

        return k_opt, info['status']
        
    def plan(self,env,ka_0):
        self.prepare_constraints(env.qpos,env.qvel,env.obs_zonos)
        k_opt, flag = self.trajopt(env.qgoal,ka_0)
        return k_opt, flag

if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time
    n_links = 2
    cyipopt.setLoggingLevel(1000)
    env = Arm_2D(n_links=n_links,n_obs=1)
    env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    #from zonopy.optimize.armtd import ARMTD_planner
    planner = ARMTD_planner(env)
    for _ in range(100):
        ts = time.time()
        ka, flag = planner.plan(env,torch.zeros(n_links))
        print(env.qpos)
        print(time.time()-ts)
        #import pdb;pdb.set_trace()
        done = env.step(torch.tensor(ka,dtype=torch.get_default_dtype()),flag)
        env.render(planner.FO_link)
        if done:
            import pdb;pdb.set_trace()
            break
    import pdb;pdb.set_trace()
