import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
#from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3
import os
import numpy as np

from zonopy.conSet import PROPERTY_ID

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Parallel_Arm_3D:
    def __init__(self,
            n_envs = 1, # number of environments
            robot='Kinova3', # robot model
            n_obs=1, # number of obstacles
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            check_joint_limit = True,
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.1, # goal threshold
            hyp_effort = 1.0, # hyperpara
            hyp_dist_to_goal = 1.0,
            hyp_collision = 2500,
            hyp_success = 150,
            hyp_fail_safe = 1,
            hyp_stuck =2000,
            stuck_threshold = None,
            reward_shaping=True,
            gamma = 0.99, # discount factor on reward
            max_episode_steps = 300,
            n_plots = None,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            FO_render_freq = 10,
            ticks = False,
            scale = 1,
            max_combs = 200,
            dtype= torch.float,
            device = torch.device('cpu')
            ):
        self.n_envs = n_envs
        self.max_combs = max_combs
        self.dtype = dtype
        self.device = device
        self.generate_combinations_upto()
        self.dimension = 3
        
        self.n_obs = n_obs
        self.scale = scale

        #### load
        params, _ = zp.load_sinlge_robot_arm_params(robot)
        self.dof = self.n_links = params['n_joints']
        self.joint_id = torch.arange(self.n_links,dtype=int,device=device)
        link_zonos = [(self.scale*params['link_zonos'][j]).to(dtype=dtype,device=device) for j in range(self.n_links)] # NOTE: zonotope, should it be poly zonotope?
        self.link_polyhedron = [link_zonos[j].polyhedron_patch() for j in range(self.n_links)]
        self.link_zonos = [link_zonos[j].to_polyZonotope() for j in range(self.n_links)]
        self.__link_zonos = [zp.batchZonotope(link_zonos[j].Z.unsqueeze(0).repeat(n_envs,1,1)) for j in range(self.n_links)]

        self.P0 = [self.scale*P.to(dtype=dtype,device=device) for P in params['P']]
        self.R0 = [R.to(dtype=dtype,device=device) for R in params['R']]
        self.joint_axes = torch.vstack(params['joint_axes']).to(dtype=dtype,device=device)
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]],dtype=dtype,device=device)
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)
        
        self.pos_lim = torch.tensor(params['pos_lim'],dtype=dtype,device=device)
        self.vel_lim = torch.tensor(params['vel_lim'],dtype=dtype,device=device)
        self.tor_lim = torch.tensor(params['tor_lim'],dtype=dtype,device=device)
        self.lim_flag = torch.tensor(params['lim_flag'],dtype=bool,device=device)

        self._pos_lim = self.pos_lim.clone()
        self._vel_lim = self.vel_lim.clone()
        self._tor_lim = self.tor_lim.clone()
        self._lim_flag = self.lim_flag.clone()
        self._actual_pos_lim = self._pos_lim[self._lim_flag]

        self.pos_sampler = torch.distributions.Uniform(self.pos_lim[:,1],self.pos_lim[:,0])
        self.full_radius = self.scale*0.8

        #self.full_radius = sum([(abs(self.P0[j])).max() for j in range(self.n_links)])        
        #### load

        self.fig_scale = 1
        self.interpolate = interpolate
        self.PI = torch.tensor(torch.pi,dtype=dtype,device=device)

        if interpolate:
            if T_len % 2 != 0:
                self.T_len = T_len + 1
            else: 
                self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1,dtype=dtype,device=device).reshape(-1,1,1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len):] - T_PLAN
        
        self.obs_buffer_length = torch.tensor([0.001,0.001],dtype=dtype,device=device)
        self.obstacle_generators = 0.1*torch.eye(3,dtype=dtype,device=device).unsqueeze(0).repeat(n_envs,1,1)
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
        self.check_joint_limit = check_joint_limit
        self.collision_threshold = collision_threshold
        
        self.goal_threshold = goal_threshold
        self.hyp_effort = hyp_effort
        self.hyp_dist_to_goal = hyp_dist_to_goal
        self.hyp_collision = hyp_collision
        self.hyp_success = hyp_success
        self.hyp_fail_safe = hyp_fail_safe
        self.hyp_stuck = hyp_stuck
        if stuck_threshold is None:
            self.stuck_threshold = max_episode_steps
        else:
            self.stuck_threshold = stuck_threshold
        self.reward_shaping = reward_shaping
        self.gamma = gamma

        self.fig = None
        self.render_flag = True
        assert FO_render_level<4
        self.FO_render_level = FO_render_level
        self.FO_render_freq = FO_render_freq
        self.ticks = ticks

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = torch.zeros(self.n_envs,dtype=int,device=device)
        self._frame_steps = 0     
        self.dtype = dtype
        self.device = device

        self.get_plot_grid_size(n_plots)
        self.reset()

    def wrap_cont_joint_to_pi(self,phases,internal=True):
        phases_new = torch.clone(phases)
        phases_new[:,~self.lim_flag] = (phases[:,~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new

    def generate_combinations_upto(self):
        self.combs = [torch.tensor([0],device=self.device)]
        for i in range(self.max_combs):
            self.combs.append(torch.combinations(torch.arange(i+1,device=self.device),2))
    
    def __reset(self,idx):
        n_envs = idx.sum()        
        self.qpos[idx] = self.pos_sampler.sample((n_envs,))
        self.qpos_int[idx] = self.qpos[idx]
        self.qvel[idx] = torch.zeros(n_envs,self.n_links,dtype=self.dtype,device=self.device)
        self.qpos_prev[idx] = self.qpos[idx]
        self.qvel_prev[idx] = self.qvel[idx]
        self.qgoal[idx] = self.pos_sampler.sample((n_envs,))

        if self.interpolate:
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1 
            self.qpos_to_brake[:,idx] = self.qpos[idx].unsqueeze(0).repeat(T_len_to_brake,1,1) 
            self.qvel_to_brake[:,idx] = torch.zeros(T_len_to_brake,n_envs,self.n_links,dtype=self.dtype,device=self.device) 
        else:
            self.qpos_brake[idx] = self.qpos[idx] + 0.5*self.qvel[idx]*(T_FULL-T_PLAN)
            self.qvel_brake[idx] = torch.zeros(n_envs,self.n_links,dtype=self.dtype,device=self.device) 

        R_qi = self.rot(self.qpos[idx])
        R_qg = self.rot(self.qgoal[idx])
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device) 
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)   
        link_init, link_goal = [], []
        for j in range(self.n_links):    
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            
            link_zonos_idx = self.__link_zonos[j][idx]
            link = Ri@link_zonos_idx+Pi
            link_init.append(link)
            link = Rg@link_zonos_idx+Pg
            link_goal.append(link)

        idx_nonzero = idx.nonzero().reshape(-1)
        for o in range(self.n_obs):
            safe_flag = torch.zeros(n_envs,dtype=bool,device=self.device)
            while True:
                obs_z, safe_idx = self.obstacle_sample(link_init,link_goal,~safe_flag)
                self.obs_zonos[o].Z[idx_nonzero[safe_idx]] = obs_z 
                safe_flag += safe_idx 
                if safe_flag.all():
                    break
        
        self._elapsed_steps[idx] = 0
        self.reward_com[idx] = 0

    def reset(self):
        self.qpos = self.pos_sampler.sample((self.n_envs,))
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
        self.qvel_init = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = self.pos_sampler.sample((self.n_envs,))

        if self.interpolate:
            T_len_to_peak = int(T_PLAN/T_FULL*self.T_len)+1
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
            self.qpos_to_peak = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1,1)
            self.qvel_to_peak = torch.zeros(T_len_to_peak,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1,1)
            self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
        
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)

        for _ in range(self.n_obs):
            safe_flag = torch.zeros(self.n_envs,dtype=bool,device=self.device)
            obs_Z = torch.zeros(self.n_envs,4,3,dtype=self.dtype,device=self.device)
            while True:
                obs_z, safe_idx = self.obstacle_sample(link_init,link_goal,~safe_flag)
                obs_Z[safe_idx] = obs_z 
                safe_flag += safe_idx 
                if safe_flag.all():
                    obs = zp.batchZonotope(obs_Z)
                    self.obs_zonos.append(obs)
                    break
        self.fail_safe_count = torch.zeros(self.n_envs,dtype=int,device=self.device)
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.obs_patches[b].remove()
                self.link_goal_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
        self.render_flag = True

        self.done = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self.collision = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self._elapsed_steps = torch.zeros(self.n_envs,dtype=int,device=self.device)
        self.reward_com = torch.zeros(self.n_envs,dtype=self.dtype,device=self.device)
        return self.get_observations()

    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos.to(dtype=self.dtype,device=self.device)
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = qvel.to(dtype=self.dtype,device=self.device)
        self.qvel_int = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = qgoal.to(dtype=self.dtype,device=self.device)   

        if self.interpolate:
            T_len_to_peak = int(T_PLAN/T_FULL*self.T_len)+1
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
            self.qpos_to_peak = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1,1)
            self.qvel_to_peak = torch.zeros(T_len_to_peak,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1,1)
            self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)     
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)

        for pos in obs_pos:
            po = pos.to(dtype=self.dtype,device=self.device).unsqueeze(1)
            obs = zp.batchZonotope(torch.cat((po,self.obstacle_generators),1))
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.polytope(self.combs)
                if any(min(b.min(1).values) > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.polytope(self.combs)
                if any(min(b.min(1).values) > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'    
            self.obs_zonos.append(obs)
        self.fail_safe_count = torch.zeros(self.n_envs,dtype=int,device=self.device)
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.obs_patches[b].remove()
                self.link_goal_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
        self.render_flag = True

        self.done = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self.collision = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self._elapsed_steps = torch.zeros(self.n_envs,dtype=int,device=self.device)
        self.reward_com = torch.zeros(self.n_envs,dtype=self.dtype,device=self.device)
        return self.get_observations()

    def obstacle_sample(self,link_init,link_goal,idx):
        '''
        if idx is None:
            n_envs= self.n_envs
            idx = torch.ones(n_envs,dtype=bool)
        else:
        ''' 
        n_envs = idx.sum()
        obs_pos = self.scale*(torch.rand(n_envs,1,3,dtype=self.dtype,device=self.device)*2*0.8-0.8)
        obs_Z = torch.cat((obs_pos,self.obstacle_generators[:n_envs]),1)
        obs = zp.batchZonotope(obs_Z)
        safe_flag = torch.zeros(len(idx),dtype=bool,device=self.device)
        safe_flag[idx] = True
        for j in range(self.n_links):            
            buff = link_init[j][idx]-obs
            _,bi = buff.polytope()
            buff = link_goal[j][idx]-obs
            _,bg = buff.polytope()   
            safe_flag[idx] *= ((bi.min(1).values < -1e-5) * (bg.min(1).values < -1e-5)) # Ture for safe envs, -1e-6: more conservative, 1e-6: less conservative

        return obs_Z[safe_flag[idx]], safe_flag


    def step(self,ka,flag=None):
        ka = ka.detach()
        self.ka = ka.clamp((-self.PI-self.qvel)/T_PLAN,(self.PI-self.qvel)/T_PLAN) # velocity clamp
        self.joint_limit_check()

        if flag is None:
            self.step_flag = torch.zeros(self.n_envs,dtype=int,device=self.device)
        else:
            self.step_flag = flag.to(dtype=int,device=self.device).detach()
        self.safe = (self.step_flag <= 0) + self.exceed_joint_limit
        
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        if self.interpolate:
            unsafe = ~self.safe
            self.fail_safe_count = (unsafe)*(self.fail_safe_count+1)
                
            safe_qpos = self.qpos[self.safe].unsqueeze(0)
            safe_qvel = self.qvel[self.safe].unsqueeze(0)
            safe_action = self.ka[self.safe].unsqueeze(0)
            self.qpos_to_peak[:,self.safe] = wrap_to_pi(safe_qpos + self.t_to_peak*safe_qvel + .5*(self.t_to_peak**2)*safe_action)
            self.qvel_to_peak[:,self.safe] = safe_qvel + self.t_to_peak*safe_action
            self.qpos_to_peak[:,unsafe] = self.qpos_to_brake[:,unsafe]
            self.qvel_to_peak[:,unsafe] = self.qvel_to_brake[:,unsafe]
    
            self.qpos = self.qpos_to_peak[-1]
            self.qvel = self.qvel_to_peak[-1]

            qpos = self.qpos.unsqueeze(0)
            qvel = self.qvel.unsqueeze(0)
            bracking_accel = (0 - qvel)/(T_FULL - T_PLAN)
            self.qpos_to_brake = wrap_to_pi(qpos + self.t_to_brake*qvel + .5*(self.t_to_brake**2)*bracking_accel)
            self.qvel_to_brake = qvel + self.t_to_brake*bracking_accel 
            self.collision = self.collision_check(self.qpos_to_peak[1:])            
            #self.collision = self.collision_check(torch.cat((self.qpos_to_peak,self.qpos_to_brake[1:]),0))
        else:
            unsafe = ~self.safe
            self.fail_safe_count = (unsafe)*(self.fail_safe_count+1)
            self.qpos[self.safe] = wrap_to_pi(self.qpos[self.safe] + self.qvel[self.safe]*T_PLAN + 0.5*self.ka[self.safe]*T_PLAN**2)
            self.qvel[self.safe] += self.ka[self.safe]*T_PLAN

            bracking_accel = (0 - self.qvel[self.safe])/(T_FULL - T_PLAN)            
            self.qpos_brake[self.safe] = wrap_to_pi(self.qpos[self.safe] + self.qvel[self.safe]*(T_FULL-T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2)            
            self.qvel_brake[self.safe] = 0
            self.qpos[unsafe] = self.qpos_brake[unsafe]
            self.qvel[unsafe] = self.qvel_brake[unsafe] 
            self.collision = self.collision_check(self.qpos)
            #self.collision = self.collision_check(torch.cat((self.qpos.unsqueeze(0),self.qpos_brake.unsqueeze(0)),0))
                    
        self._elapsed_steps += 1
        
        self.reward = self.get_reward(ka) # NOTE: should it be ka or self.ka ??
        self.reward_com *= self.gamma
        self.reward_com += self.reward
        self.stuck = self.fail_safe_count > self.stuck_threshold
        self.done = self.success + self.collision + self.stuck

        infos = self.get_info()
        if self.done.sum()>0:
            self.__reset(self.done)
        observations = self.get_observations()
        return observations, self.reward, self.done, infos
    
    def get_info(self):
        infos = []
        for idx in range(self.n_envs):
            info = {
                'is_success':bool(self.success[idx]),
                'collision':bool(self.collision[idx]),
                'safe_flag':bool(self.safe[idx]),
                'step_flag':int(self.step_flag[idx]),
                'stuck': bool(self.stuck[idx])
                }
            if self.collision[idx]:
                collision_info = {
                    'qpos_init':self.qpos_int[idx],
                    'qvel_int':torch.zeros(self.n_links),
                    'obs_pos':[self.obs_zonos[o].center[idx] for o in range(self.n_obs)],
                    'qgoal':self.qgoal[idx]
                }
                info['collision_info'] = collision_info
            if self._elapsed_steps[idx] >= self._max_episode_steps:
                info["TimeLimit.truncated"] = not self.done[idx]
                self.done[idx] = True       
            info['episode'] = {"r":float(self.reward_com[idx]),"l":int(self._elapsed_steps[idx])}
            infos.append(info)
        return tuple(infos)

    def get_observations(self):
        observation = {'qpos':torch.clone(self.qpos),'qvel':torch.clone(self.qvel),'qgoal':torch.clone(self.qgoal)}
        
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.cat([self.obs_zonos[o].center.unsqueeze(1) for o in range(self.n_obs)],1)
            observation['obstacle_size'] = torch.cat([self.obs_zonos[o].generators[:,[0,1,2],[0,1,2]].unsqueeze(1) for o in range(self.n_obs)],1)
        return observation
    
    def get_reward(self, action, qpos=None, qgoal=None, collision=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None:
            collision = self.collision 
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(self.qpos-self.qgoal,internal=True),dim=-1)
            self.success = goal_dist < self.goal_threshold 
            success = self.success.to(dtype=self.dtype)
        else: 
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(qpos-qgoal,internal=False),dim=-1)
            success = (goal_dist < self.goal_threshold).to(dtype=self.dtype)*(1 - collision) 
        
        reward = 0.0

        # Return the sparse reward if using sparse_rewards
        if not self.reward_shaping:
            reward -= self.hyp_collision * self.collision
            reward += self.hyp_success * success
            return reward

        # otherwise continue to calculate the dense reward
        # reward for position term
        reward -= self.hyp_dist_to_goal * goal_dist
        # reward for effort
        reward -= self.hyp_effort * torch.linalg.norm(action,dim=-1)
        # Add collision if needed
        reward -= self.hyp_collision * self.collision
        # Add fail-safe if needed
        reward -= self.hyp_fail_safe * (1 - self.safe.to(dtype=self.dtype))
        # Add stuck if needed
        reward -= self.hyp_stuck * torch.tensor(self.fail_safe_count > self.stuck_threshold,dtype=self.dtype)
        # Add success if wanted
        reward += self.hyp_success * success

        return reward    

    def joint_limit_check(self):
        if self.check_joint_limit:
            t_peak_optimum = -self.qvel/self.ka # time to optimum of first half traj.
            qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*self.ka*t_peak_optimum**2).nan_to_num()
            qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * self.ka * T_PLAN**2
            qvel_peak = self.qvel + self.ka * T_PLAN

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * self.ka * T_PLAN * T_FULL
            qpos_possible_max_min = torch.cat((qpos_peak_optimum.unsqueeze(-2),qpos_peak.unsqueeze(-2),qpos_brake.unsqueeze(-2)),-2)[:,:,self._lim_flag]

            qpos_ub = (qpos_possible_max_min - self._actual_pos_lim[:,0]).reshape(self.n_envs,-1)
            qpos_lb = (self._actual_pos_lim[:,1] - qpos_possible_max_min).reshape(self.n_envs,-1)
            self.exceed_joint_limit = (abs(qvel_peak)>self._vel_lim).any(-1) + (qpos_ub>0).any(-1) + (qpos_lb>0).any(-1)

        self.exceed_joint_limit = torch.zeros(self.n_envs,dtype=bool)
    
    def collision_check(self,qs):
        unsafe = torch.zeros(self.n_envs,dtype=bool)
        if self.check_collision:
            R_q = self.rot(qs)
            if len(R_q.shape) == 5:
                time_steps = R_q.shape[0]
                R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,:,j]
                    link =zp.batchZonotope(self.__link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1,1))
                    link = R@link+P
                    for o in range(self.n_obs):
                        buff = torch.cat(((link.center - self.obs_zonos[o].center).unsqueeze(-2),link.generators,self.obs_zonos[o].generators.unsqueeze(0).repeat(time_steps,1,1,1)),-2)
                        _,b = zp.batchZonotope(buff).polytope()
                        unsafe += (b.min(dim=-1)[0]>1e-6).any(dim=0)
                
                        
            else:
                time_steps = 1
                R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,j]
                    link = R@self.__link_zonos[j]+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.polytope()
                        unsafe += b.min(dim=-1)[0] > 1e-6
        return unsafe   

    def render(self,FO_link=None,show=True,dpi=None,save_kwargs=None):
        if self.render_flag:
            if self.fig is None:
                if show:
                    plt.ion()
                if self.n_plots == 1:
                    self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8],dpi=dpi)
                    self.axs = [plt.axes(projection='3d')]
                else:
                    self.fig = plt.figure(figsize=[self.plot_grid_size[1]*6.4,self.plot_grid_size[0]*4.8],dpi=dpi)
                    self.axs = []
                    for y in range(self.plot_grid_size[1]):
                        for x in range(self.plot_grid_size[0]):
                            self.axs.append(self.fig.add_subplot(self.plot_grid_size[0],self.plot_grid_size[1],x + 1 + y*self.plot_grid_size[0],projection='3d'))
                            if not self.ticks:
                                plt.tick_params(which='both',bottom=False,top=False,left=False,right=False, labelbottom=False, labelleft=False)
                if save_kwargs is not None:
                    os.makedirs(save_kwargs['save_path'],exist_ok=True) 
                    if self.n_plots == 1:
                        fontsize = 10
                    else:
                        fontsize = 7 + self.plot_grid_size[0]
                    self.axs[self.plot_grid_size[0]-1].set_title(save_kwargs['text'],fontsize=fontsize,loc='right')
            
            self.render_flag = False
            self.obs_patches, self.link_goal_patches, self.FO_patches, self.link_patches= [], [], [], [] 
            # Collect patches of polyhedron representation of link in goal configuration
            link_goal_patches = []
            R_q = self.rot(self.qgoal[:self.n_plots]).unsqueeze(-3)
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device) 

            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_goal_patches.append((self.link_polyhedron[j]@R.transpose(-1,-2)+P.unsqueeze(-3)).cpu())

            for b, ax in enumerate(self.axs):
                link_goal_patch, obs_patch = [], [] 
                for j in range(self.n_links):
                    link_goal_patch.extend(link_goal_patches[j][b])
                for o in range(self.n_obs):
                    obs_patch.extend(self.obs_zonos[o][b].polyhedron_patch())
                self.link_goal_patches.append(ax.add_collection3d(Poly3DCollection(link_goal_patch,edgecolor='gray',facecolor='gray',alpha=0.15,linewidths=0.5)))
                self.obs_patches.append(ax.add_collection3d(Poly3DCollection(obs_patch,edgecolor='red',facecolor='red',alpha=0.2,linewidths=0.2)))
                self.FO_patches.append(ax.add_collection3d(Poly3DCollection([])))
                self.link_patches.append(ax.add_collection3d(Poly3DCollection([])))

        # Collect patches of polyhedron representation of forward reachable set
        if FO_link is not None and self.FO_render_freq != 0: 
            FO_patch = []
            FO_render_idx = (self.fail_safe_count[:self.n_plots] == 0).nonzero().reshape(-1)
            g_ka = self.PI/24
            if FO_render_idx.numel() != 0:
                FO_link_slc = []
                for j in range(self.n_links):
                    PROPERTY_ID.update(self.n_links)
                    FO_link_slc.append(FO_link[j][FO_render_idx].to(dtype=self.dtype,device=self.device).slice_all_dep((self.ka[FO_render_idx]/g_ka).unsqueeze(1).repeat(1,100,1)).reduce(4))
                    zp.reset()

                for idx, b in enumerate(FO_render_idx.tolist()):
                    self.FO_patches[b].remove()
                    FO_patches = []
                    for j in range(self.n_links):
                        for t in range(100):
                            if t % self.FO_render_freq == 0:
                                FO_patch = FO_link_slc[j][idx,t].polyhedron_patch().detach()
                                FO_patches.extend(FO_patch)
                    self.FO_patches[b] = self.axs[b].add_collection3d(Poly3DCollection(FO_patches,alpha=0.03,edgecolor='green',facecolor='green',linewidths=0.2))

        if self.interpolate:
            timesteps = int(T_PLAN/T_FULL*self.T_len)
            if show and save_kwargs is None:
                plot_freq = 1
                R_q = self.rot(self.qpos_to_peak[1:,:self.n_plots]).unsqueeze(-3) 
            elif show:
                plot_freq = timesteps//save_kwargs['frame_rate']
                R_q = self.rot(self.qpos_to_peak[1:,:self.n_plots]).unsqueeze(-3)
            else:
                plot_freq = 1
                t_idx = torch.arange(timesteps+1,device=self.device)%(timesteps//save_kwargs['frame_rate'] ) == 1
                R_q = self.rot(self.qpos_to_peak[t_idx,:self.n_plots]).unsqueeze(-3) 
                timesteps = len(R_q)
             
            link_trace_polyhedrons = torch.zeros(timesteps,self.n_plots,12*self.n_links,3,3,dtype=self.dtype,device='cpu')
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,1,3,dtype=self.dtype,device=self.device)
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[:,:,j]
                link_trace_polyhedrons[:,:,12*j:12*(j+1)] = (self.link_polyhedron[j]@R.transpose(-1,-2)+P.unsqueeze(-3)).cpu()
            
            for t in range(timesteps):
                for b, ax in enumerate(self.axs):
                    self.link_patches[b].remove()
                    self.link_patches[b] = ax.add_collection3d(Poly3DCollection(list(link_trace_polyhedrons[t,b]), edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5))    
                    ax.set_xlim([-self.full_radius,self.full_radius])
                    ax.set_ylim([-self.full_radius,self.full_radius])
                    ax.set_zlim([-self.full_radius,self.full_radius])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                if save_kwargs is not None and t%plot_freq == 0:
                    filename = "/frame_"+"{0:04d}".format(self._frame_steps)                    
                    self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                    self._frame_steps+=1

        else:
            R_q = self.rot(self.qpos[:self.n_plots]).unsqueeze(-3) 
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            link_trace_polyhedrons = torch.zeros(self.n_plots,12*self.n_links,3,3,dtype=self.dtype,device='cpu')
        
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_trace_polyhedrons[:,12*j:12*(j+1)] = (self.link_polyhedron[j]@R.transpose(-1,-2)+P.unsqueeze(-3)).cpu()

            for b, ax in enumerate(self.axs):
                self.link_patches[b].remove()
                self.link_patches[b] = ax.add_collection3d(Poly3DCollection(list(link_trace_polyhedrons[b]), edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5)) 
                ax.set_xlim([-self.full_radius,self.full_radius])
                ax.set_ylim([-self.full_radius,self.full_radius])
                ax.set_zlim([-self.full_radius,self.full_radius])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if save_kwargs is not None:
                filename = "/frame_"+"{0:04d}".format(self._frame_steps)
                self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                self._frame_steps+=1

        if self.done.any():
            reset_flag = self.done[:self.n_plots].nonzero().reshape(-1)
        
            R_q = self.rot(self.qgoal[reset_flag]).unsqueeze(-3) 
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            link_goal_polyhedrons = torch.zeros(reset_flag.numel(),12*self.n_links,3,3,dtype=self.dtype,device='cpu')

            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_goal_polyhedrons[:,12*j:12*(j+1)] = (self.link_polyhedron[j]@R.transpose(-1,-2)+P.unsqueeze(-3)).cpu()

            for idx, b in enumerate(reset_flag.tolist()):
                self.link_goal_patches[b].remove()
                self.obs_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
                ax = self.axs[b] 
                obs_patch = [] 
                for o in range(self.n_obs):
                    obs_patch.extend(self.obs_zonos[o][b].polyhedron_patch())                
                self.obs_patches[b] = ax.add_collection3d(Poly3DCollection(obs_patch,edgecolor='red',facecolor='red',alpha=0.2,linewidths=0.2))
                self.link_goal_patches[b] = ax.add_collection3d(Poly3DCollection(list(link_goal_polyhedrons[idx]),edgecolor='gray',facecolor='gray',alpha=0.15,linewidths=0.5))
                self.FO_patches[b] = ax.add_collection3d(Poly3DCollection([]))
                self.link_patches[b] = ax.add_collection3d(Poly3DCollection([])) 
    def close(self):
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.obs_patches[b].remove()
                self.link_goal_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
        self.render_flag = True
        self._frame_steps = 0
        plt.close()
        self.fig = None 

    def get_plot_grid_size(self,n_plots):
        if n_plots is None:
            n_plots = self.n_envs

        if n_plots in (1,2,3):
            self.plot_grid_size = (1, n_plots)
        elif n_plots < 9:
            self.plot_grid_size = (2, min(n_plots//2,4))
        else:
            self.plot_grid_size = (3,3)
        self.n_plots = self.plot_grid_size[0]*self.plot_grid_size[1]
        
    def rot(self,q=None):
        if q is None:
            q = self.qpos
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3,dtype=self.dtype,device=self.device) + torch.sin(q)*self.rot_skew_sym + (1-torch.cos(q))*self.rot_skew_sym@self.rot_skew_sym

    @property
    def action_spec(self):
        pass
    @property
    def action_dim(self):
        pass
    @property 
    def action_space(self):
        pass 
    @property 
    def observation_space(self):
        pass 
    @property 
    def obs_dim(self):
        pass




class Parallel_Locked_Arm_3D(Parallel_Arm_3D):
    def __init__(self,
            n_envs = 1, # number of environments
            robot='Kinova3', # robot model
            n_obs=1, # number of obstacles
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.05, # goal threshold
            hyp_effort = 1.0, # hyperpara
            hyp_dist_to_goal = 1.0,
            hyp_collision = 2500,
            hyp_success = 50,
            hyp_fail_safe = 1,
            hyp_stuck =2000,
            stuck_threshold = None,
            reward_shaping=True,
            gamma = 0.99, # discount factor on reward
            max_episode_steps = 300,
            n_plots = None,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            FO_render_freq = 0,
            ticks = False,
            scale = 1,
            max_combs = 200,
            dtype= torch.float,
            device = torch.device('cpu'),
            locked_idx = [],
            locked_qpos = [],
            ):
        self.unlocked_idx = []
        super().__init__(
            n_envs = n_envs, # number of environments
            robot=robot, # robot model
            n_obs=n_obs, # number of obstacles
            T_len=T_len, # number of discritization of time interval
            interpolate = interpolate, # flag for interpolation
            check_collision = check_collision, # flag for whehter check collision
            check_collision_FO = check_collision_FO, # flag for whether check collision for FO rendering
            collision_threshold = collision_threshold, # collision threshold
            goal_threshold = goal_threshold, # goal threshold
            hyp_effort = hyp_effort, # hyperpara
            hyp_dist_to_goal = hyp_dist_to_goal,
            hyp_collision = hyp_collision,
            hyp_success = hyp_success,
            hyp_fail_safe = hyp_fail_safe,
            hyp_stuck = hyp_stuck,
            stuck_threshold = stuck_threshold,
            reward_shaping=reward_shaping,
            gamma = gamma, # discount factor on reward
            max_episode_steps = max_episode_steps,
            n_plots = n_plots,
            FO_render_level = FO_render_level, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            FO_render_freq = FO_render_freq,
            ticks = ticks,
            scale = scale,
            max_combs = max_combs,
            dtype= dtype,
            device = device
            )
        self.locked_idx = torch.tensor(locked_idx,dtype=int,device=device)
        self.locked_qpos = torch.tensor(locked_qpos,dtype=dtype,device=device)
        self.dof = self.n_links - len(locked_idx)
    
        self.unlocked_idx = torch.ones(self.n_links,dtype=bool,device=device)
        self.unlocked_idx[self.locked_idx] = False
        self.unlocked_idx = self.unlocked_idx.nonzero().reshape(-1)

        locked_pos_lim = self.pos_lim.clone()
        locked_pos_lim[self.locked_idx,0] = locked_pos_lim[self.locked_idx,1] = self.locked_qpos
        self.pos_sampler = torch.distributions.Uniform(locked_pos_lim[:,1],locked_pos_lim[:,0],validate_args=False)

        self.pos_lim = self.pos_lim[self.unlocked_idx]
        self.vel_lim = self.vel_lim[self.unlocked_idx]
        self.tor_lim = self.tor_lim[self.unlocked_idx]
        self.lim_flag = self.lim_flag[self.unlocked_idx] # NOTE, wrap???
        self.joint_id = self.joint_id[self.unlocked_idx]
        self.reset()
    
    def wrap_cont_joint_to_pi(self,phases,internal=True):
        if internal:
            phases_new = torch.clone(phases)
            idx = self.unlocked_idx[~self.lim_flag]
            phases_new[:,idx] = (phases[:,idx] + torch.pi) % (2 * torch.pi) - torch.pi
        else:
            phases_new = torch.zeros(phases.shape[0], self.n_links).to(phases.device, phases.dtype)
            phases_new[:,self.unlocked_idx] = phases
            idx = self.unlocked_idx[~self.lim_flag]
            phases_new[:,idx] = (phases_new[:,idx] + torch.pi) % (2 * torch.pi) - torch.pi
            phases_new = phases_new[:,self.unlocked_idx]
        return phases_new

    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos.to(dtype=self.dtype,device=self.device)
        self.qpos[:,self.locked_idx] = self.locked_qpos.unsqueeze(0).repeat(self.n_envs,1)
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = qvel.to(dtype=self.dtype,device=self.device)
        self.qvel_int = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = qgoal.to(dtype=self.dtype,device=self.device)   
        self.qgoal[:,self.locked_idx] = self.locked_qpos.unsqueeze(0).repeat(self.n_envs,1)

        if self.interpolate:
            T_len_to_peak = int(T_PLAN/T_FULL*self.T_len)+1
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
            self.qpos_to_peak = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1,1)
            self.qvel_to_peak = torch.zeros(T_len_to_peak,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1,1)
            self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_envs,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)     
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)

        for pos in obs_pos:
            po = pos.to(dtype=self.dtype,device=self.device).unsqueeze(1)
            obs = zp.batchZonotope(torch.cat((po,self.obstacle_generators),1))
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.polytope(self.combs)
                if any(min(b.min(1).values) > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.polytope(self.combs)
                if any(min(b.min(1).values) > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'    
            self.obs_zonos.append(obs)
        self.fail_safe_count = torch.zeros(self.n_envs,dtype=int,device=self.device)
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.obs_patches[b].remove()
                self.link_goal_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
        self.render_flag = True

        self.done = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self.collision = torch.zeros(self.n_envs,dtype=bool,device=self.device)
        self._elapsed_steps = torch.zeros(self.n_envs,dtype=int,device=self.device)
        self.reward_com = torch.zeros(self.n_envs,dtype=self.dtype,device=self.device)
        return self.get_observations()

    def step(self,ka,flag=None):

        ka_all = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
        ka_all[:,self.unlocked_idx] = ka

        return super().step(ka_all,flag)
    
    def get_observations(self):
        observation = {'qpos':torch.clone(self.qpos[:,self.unlocked_idx]),'qvel':torch.clone(self.qvel[:,self.unlocked_idx]),'qgoal':torch.clone(self.qgoal[:,self.unlocked_idx])}
        
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.cat([self.obs_zonos[o].center.unsqueeze(1) for o in range(self.n_obs)],1)
            observation['obstacle_size'] = torch.cat([self.obs_zonos[o].generators[:,[0,1,2],[0,1,2]].unsqueeze(1) for o in range(self.n_obs)],1)
        return observation

if __name__ == '__main__':
    n_envs = 9
    # [2,4,5,6], [0,0,0,0]
    # [2,5], [0,0]
    env = Parallel_Locked_Arm_3D(n_envs = n_envs, n_obs=2,T_len=50,interpolate=True,n_plots=1,locked_idx = [2,5], locked_qpos = [0,0])
    for _ in range(3):
        for _ in range(10):
            env.step(torch.rand(n_envs,env.dof))
            env.render()
            import pdb;pdb.set_trace()
            #env.reset()