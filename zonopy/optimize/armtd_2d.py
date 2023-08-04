import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import cyipopt
import time
def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0

class ARMTD_2D_planner():
    def __init__(self,env,zono_order=40,max_combs=200,dtype=torch.float,device=torch.device('cpu')):
        self.dtype, self.device = dtype, device
        self.wrap_env(env)
        self.n_timesteps = 100
        self.eps = 1e-6
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS_tensor = zp.preload_batch_JRS_trig(dtype=self.dtype,device=self.device)
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def wrap_env(self,env):
        assert env.dimension == 2
        self.dimension = 2
        self.n_links = env.n_links
        self.n_obs = env.n_obs
        P,R,self.link_zonos = [], [], []
        for p,r,l in zip(env.P0,env.R0,env.link_zonos):
            P.append(p.to(dtype=self.dtype,device=self.device))
            R.append(r.to(dtype=self.dtype,device=self.device))
            self.link_zonos.append(l.to(dtype=self.dtype,device=self.device))
        self.params = {'n_joints':env.n_links, 'P':P, 'R':R}     
        self.joint_axes = env.joint_axes.to(dtype=self.dtype,device=self.device)

    def prepare_constraints(self,qpos,qvel,obstacles):
        _, R_trig = zp.process_batch_JRS_trig(self.JRS_tensor,qpos.to(dtype=self.dtype,device=self.device),qvel.to(dtype=self.dtype,device=self.device),self.joint_axes)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params) # NOTE: zono_order
        self.A = np.zeros((self.n_links,self.n_obs),dtype=object)
        self.b = np.zeros((self.n_links,self.n_obs),dtype=object)
        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))
        for j in range(self.n_links):
            self.FO_link[j] = self.FO_link[j].project([0,1]).cpu()
            for o in range(self.n_obs):                
                obs_Z = obstacles[o].Z[:,:self.dimension].unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension  
                self.A[j,o] = A.cpu()
                self.b[j,o] = b.cpu()
        self.qpos = qpos.to(dtype=self.dtype,device='cpu')
        self.qvel = qvel.to(dtype=self.dtype,device='cpu')

    def prepare_constraints2(self,qpos,qvel,obstacles):
        _, R_trig = zp.process_batch_JRS_trig(self.JRS_tensor,qpos.to(dtype=self.dtype,device=self.device),qvel.to(dtype=self.dtype,device=self.device),self.joint_axes)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params) # NOTE: zono_order
        self.A = np.zeros((self.n_links),dtype=object)
        self.b = np.zeros((self.n_links),dtype=object)
        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))

        obs_Z = []
        for obs in obstacles:
            #Z = obs.Z.clone()
            #Z[1:] += torch.eye(3) * BUFFER_AMOUNT
            #obs_Z.append(Z.unsqueeze(0))
            # import pdb;pdb.set_trace()
            obs_Z.append(obs.Z[:,:self.dimension].unsqueeze(0))
        obs_Z = torch.cat(obs_Z,0).to(dtype=self.dtype, device=self.device).unsqueeze(1).repeat(1,self.n_timesteps,1,1)

        obs_in_reach_idx = torch.zeros(self.n_obs,dtype=bool,device=self.device)
        for j in range(self.n_links):
            temp = self.FO_link[j].project([0,1])
            obs_buff_Grest = zp.batchZonotope(torch.cat((obs_Z,temp.Grest.unsqueeze(0).repeat(self.n_obs,1,1,1)),-2))
            A_Grest, b_Grest  = obs_buff_Grest.polytope()
            obs_buff = obs_buff_Grest - zp.batchZonotope(temp.Z[temp.batch_idx_all+(slice(temp.n_dep_gens+1),)].unsqueeze(0).repeat(self.n_obs,1,1,1))
            _, b_obs = obs_buff.reduce(3).polytope()
            obs_in_reach_idx += (torch.min(b_obs.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)
            self.FO_link[j] = self.FO_link[j].project([0,1]).cpu()
            self.A[j] = A_Grest
            self.b[j] = b_Grest
        
        for j in range(self.n_links):
            self.A[j] = self.A[j][obs_in_reach_idx].cpu()
            self.b[j] = self.b[j][obs_in_reach_idx].cpu()

        self.n_obs_in_frs = int(sum(obs_in_reach_idx))
        self.qpos = qpos.to(dtype=self.dtype,device='cpu')
        self.qvel = qvel.to(dtype=self.dtype,device='cpu')

    def trajopt(self,qgoal,ka_0):
        n_obs_cons = self.n_timesteps*self.n_obs_in_frs
        M_obs = self.n_links*n_obs_cons
        M = M_obs+2*self.n_links

        class nlp_setup():
            x_prev = np.zeros(self.n_links)*np.nan
            def objective(p,x):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
                return torch.sum(wrap_to_pi(qplan-qgoal)**2)

            def gradient(p,x):
                qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
                qplan_grad = 0.5*self.g_ka*T_PLAN**2
                return (2*qplan_grad*wrap_to_pi(qplan-qgoal)).numpy()

            def constraints(p,x): 
                p.compute_constraints(x)     
                return p.Cons

            def jacobian(p,x):
                p.compute_constraints(x)
                return p.Jac
            '''
            def hessianstructure(p):
                return np.nonzero(np.tril(np.ones((self.n_links,self.n_links))))

            def hessian(p, x, lagrange, obj_factor):
                p.compute_constraints(x)
                H = obj_factor*0.5*T_PLAN**4*np.eye(self.n_links) + np.sum(lagrange.reshape(-1,1,1)*p.hess_cons_obs,0)
                row, col = p.hessianstructure()
                return H[row,col]
                #return H 
            '''
            def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
                pass
            
            def compute_constraints(p,x):
                if (p.x_prev!=x).any():
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(self.n_timesteps,1)
                    Cons = torch.zeros(M,dtype=self.dtype)
                    Jac = torch.zeros(M,self.n_links,dtype=self.dtype)
                    # hess_cons_obs = torch.zeros(self.n_timesteps*self.n_links*self.n_obs+2*self.n_links,self.n_links,self.n_links)
                    # velocity min max constraints
                    
                    qvel_peak = self.qvel + self.g_ka * ka[0] * T_PLAN
                    grad_qvel_peak = self.g_ka * T_PLAN * torch.eye(self.n_links,dtype=self.dtype)

                    Cons[-2*self.n_links:] = torch.hstack((qvel_peak,qvel_peak))
                    Jac[-2*self.n_links:] = torch.vstack((grad_qvel_peak,grad_qvel_peak))
                    # velocity min max constraints
                    if self.n_obs_in_frs > 0:
                        for j in range(self.n_links):
                            c_k = self.FO_link[j].center_slice_all_dep(ka)
                            grad_c_k = self.FO_link[j].grad_center_slice_all_dep(ka)
                            # hess_c_k = self.FO_link[j].hess_center_slice_all_dep(ka)
                            h_obs = (self.A[j]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j]
                            cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1)
                            A_max = self.A[j].gather(-2,ind.reshape(self.n_obs_in_frs,self.n_timesteps,1,1).repeat(1,1,1,self.dimension))
                            grad_obs = (A_max@grad_c_k).reshape(n_obs_cons,self.n_links)
                            Cons[j*n_obs_cons:(j+1)*n_obs_cons] = - cons_obs.reshape(n_obs_cons)
                            Jac[j*n_obs_cons:(j+1)*n_obs_cons] = - grad_obs
                            # Hess[j*n_obs_cons:(j+1)*n_obs_cons] = - hess_obs
                            
                    
                    p.Cons = Cons.numpy()
                    p.Jac = Jac.numpy()
                    # p.Hess = Hess.numpy()
                    p.x_prev = np.copy(x)        
        
        nlp = cyipopt.Problem(
        n = self.n_links,
        m = M,
        problem_obj=nlp_setup(),
        lb = [-1]*self.n_links,
        ub = [1]*self.n_links,
        cl = [-1e20]*M_obs+[-1e20]*self.n_links+[-torch.pi/2+1e-6]*self.n_links,
        cu = [-1e-6]*M_obs+[torch.pi/2-1e-6]*self.n_links+[1e20]*self.n_links,
        )
        #nlp.add_option('mu_strategy', 'adaptive')

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        #nlp.add_option('derivative_test','first-order')
        #nlp.add_option('derivative_test_perturbation',1e-8)

        k_opt, self.info = nlp.solve(ka_0.cpu().numpy())    
        # 
        # prob = nlp_setup() 
        #(prob.constraints(ka_0.cpu().numpy()+np.array([1e-8,0])) - prob.constraints(ka_0.cpu().numpy())) /1e-8
        # prob.jacobian(ka_0.cpu().numpy())
        return torch.tensor(self.g_ka*k_opt,dtype=self.dtype,device=self.device), self.info['status']
        
    def plan(self,env,ka_0):
        self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)
        t1 = time.time()
        k_opt, flag = self.trajopt(env.qgoal,ka_0)
        return k_opt, flag, time.time()-t1


if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        device = 'cpu'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### 1. SET ENVIRONMENT #####        
    n_links = 2
    env = Arm_2D(n_links=n_links,n_obs=1)
    observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),
                                  qvel= torch.zeros(n_links),   
                                  qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),
                                  obs_pos=[torch.tensor([-1,-0.9])])

    ##### 2. RUN ARMTD #####
    planner = ARMTD_2D_planner(env,device=device,dtype=dtype)
    t_armtd = 0
    t_render = 0
    T_NLP = 0
    n_steps = 100
    print('='*90)    
    for _ in range(n_steps):
        ts = time.time()
        ka, flag, tnlp = planner.plan(env,torch.zeros(n_links))
        t_elasped = time.time()-ts
        #print(f'Time elasped for ARMTD-2d:{t_elasped}')
        t_armtd += t_elasped
        T_NLP += tnlp
        observations, reward, done, info = env.step(ka.cpu(),flag)

        ts =time.time()
        env.render(planner.FO_link)
        t_render += time.time()-ts
    print(f'Total time elasped for ARMTD-2D with {n_steps} steps: {t_armtd}')
    print(f'Total time elasped for rendering with {n_steps} steps: {t_render}')
    print(T_NLP)
    print(env.qpos)
    print(env.qvel)
    #import pdb;pdb.set_trace()

