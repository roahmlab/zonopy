# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy_old as forward_occupancy
import cyipopt


import time

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0
BUFFER_AMOUNT = 0.03
class ARMTD_3D_planner():
    def __init__(self,env,zono_order=40,max_combs=200,dtype=torch.float,device=torch.device('cpu')):
        self.dtype, self.device = dtype, device
        self.wrap_env(env)
        self.n_timesteps = 100
        self.eps = 1e-6
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.generate_combinations_upto()
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS_tensor = zp.preload_batch_JRS_trig(dtype=self.dtype,device=self.device)
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def wrap_env(self,env):
        assert env.dimension == 3
        self.dimension = 3
        self.n_links = env.n_links
        self.n_obs = env.n_obs

        P,R,self.link_zonos = [], [], []
        for p,r,l in zip(env.P0,env.R0,env.link_zonos):
            P.append(p.to(dtype=self.dtype,device=self.device))
            R.append(r.to(dtype=self.dtype,device=self.device))
            self.link_zonos.append(l.to(dtype=self.dtype,device=self.device))
        self.params = {'n_joints':env.n_links, 'P':P, 'R':R}        
        self.joint_axes = env.joint_axes.to(dtype=self.dtype,device=self.device)
        self.vel_lim =  env.vel_lim.cpu()
        self.pos_lim = env.pos_lim.cpu()
        self.actual_pos_lim = env.pos_lim[env.lim_flag].cpu()
        self.n_pos_lim = int(env.lim_flag.sum().cpu())
        self.lim_flag = env.lim_flag.cpu()

    def wrap_cont_joint_to_pi(self,phases):
        phases_new = torch.clone(phases)
        phases_new[~self.lim_flag] = (phases[~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new

    def generate_combinations_upto(self):
        self.combs = [torch.combinations(torch.arange(i,device=self.device),2) for i in range(self.max_combs+1)]

    def prepare_constraints0(self,qpos,qvel,obstacles):
        t1 = time.time()
        _, R_trig = zp.process_batch_JRS_trig(self.JRS_tensor,qpos.to(dtype=self.dtype,device=self.device),qvel.to(dtype=self.dtype,device=self.device),self.joint_axes)
        self.FO_link,_, _ = forward_occupancy(R_trig,self.link_zonos,self.params) # NOTE: zono_order
                
        t2 = time.time()
        print(f'FO time: {t2-t1}')
        self.A = np.zeros((self.n_links,self.n_obs),dtype=object)
        self.b = np.zeros((self.n_links,self.n_obs),dtype=object)

        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))
        for j in range(self.n_links):
            self.FO_link[j] = self.FO_link[j].cpu()
            for o in range(self.n_obs):                
                obs_Z = obstacles[o].Z.unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                self.A[j,o] = A.cpu()
                self.b[j,o] = b.cpu()
                #A2, b2 = zp.batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope2(self.combs) # A: n_timesteps,*,dimension  
                #self.A2[j,o] = A2.cpu()
                #self.b2[j,o] = b2.cpu()

        print(f'Polytope time: {time.time()-t2}')
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
            obs_Z.append(obs.Z.unsqueeze(0))
        obs_Z = torch.cat(obs_Z,0).to(dtype=self.dtype, device=self.device).unsqueeze(1).repeat(1,self.n_timesteps,1,1)

        obs_in_reach_idx = torch.zeros(self.n_obs,dtype=bool,device=self.device)
        for j in range(self.n_links):
            
            temp = self.FO_link[j]

            obs_buff_Grest = zp.batchZonotope(torch.cat((obs_Z,temp.Grest.unsqueeze(0).repeat(self.n_obs,1,1,1)),-2))
            A_Grest, b_Grest  = obs_buff_Grest.polytope(self.combs)
            obs_buff = obs_buff_Grest - zp.batchZonotope(temp.Z[temp.batch_idx_all+(slice(temp.n_dep_gens+1),)].unsqueeze(0).repeat(self.n_obs,1,1,1))
            _, b_obs = obs_buff.reduce(3).polytope(self.combs)
            
            
            obs_in_reach_idx += (torch.min(b_obs.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)
            self.FO_link[j] = self.FO_link[j].cpu()
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
        M = M_obs+2*self.n_links+6*self.n_pos_lim

        # Moved to another file
        from zonopy.optimize.nlp_setup_extracted import nlp_setup
        problem_obj = nlp_setup(
            self.qpos,
            self.qvel,
            self.g_ka,
            self.wrap_cont_joint_to_pi,
            qgoal,
            M,
            self.dtype,
            self.n_timesteps,
            self.n_links,
            M_obs,
            self.lim_flag,
            self.actual_pos_lim,
            self.vel_lim,
            self.n_obs_in_frs,
            self.FO_link,
            self.A,
            self.b,
            self.dimension,
            n_obs_cons
        )

        nlp = cyipopt.Problem(
        n = self.n_links,
        m = M,
        problem_obj=problem_obj,
        lb = [-1]*self.n_links,
        ub = [1]*self.n_links,
        cl = [-1e20]*M,
        cu = [-1e-6]*M,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)

        k_opt, self.info = nlp.solve(ka_0.cpu().numpy())                
        return torch.tensor(self.g_ka*k_opt,dtype=self.dtype,device=self.device), self.info['status']
        
    def plan(self,env,ka_0):
        self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)
        t1 = time.time()
        k_opt, flag = self.trajopt(env.qgoal,ka_0)
        return k_opt, flag, time.time()-t1


if __name__ == '__main__':
    from zonopy.environments.arm_3d import Arm_3D
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### 1. SET ENVIRONMENT #####        
    env = Arm_3D(n_obs=5)
    env.set_initial(qpos=torch.tensor([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),qvel=torch.tensor([0,0,0,0,0,0,0.]),qgoal = torch.tensor([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]), obs_pos=[torch.tensor([0.65,-0.46,0.33]),torch.tensor([0.5,-0.43,0.3]),torch.tensor([0.47,-0.45,0.15]),torch.tensor([-0.3,0.2,0.23]),torch.tensor([0.3,0.2,0.31])])
    ##### 2. RUN ARMTD #####    
    planner = ARMTD_3D_planner(env)
    t_armtd = 0
    T_NLP = 0
    n_steps = 100
    for _ in range(n_steps):
        ts = time.time()
        ka, flag, tnlp = planner.plan(env,torch.zeros(env.n_links))
        t_elasped = time.time()-ts
        #print(f'Time elasped for ARMTD-3d:{t_elasped}')
        T_NLP += tnlp
        t_armtd += t_elasped
        env.step(ka,flag)
        env.render()
    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    print(T_NLP)