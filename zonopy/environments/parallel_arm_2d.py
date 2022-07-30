import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import os
import numpy as np

from zonopy.conSet import PROPERTY_ID

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Parallel_Arm_2D:
    def __init__(self,
            n_envs = 1, # number of environments
            n_links = 2, # number of links
            n_obs = 1, # number of obstacles
            T_len = 50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            # NOTE: validation
            check_collision_FO = False, # flag for whether check collision for FO rendering
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.05, # goal threshold
            hyp_effort = 1.0, # hyperpara
            hyp_dist_to_goal = 1.0,
            hyp_collision = 300,
            hyp_success = 50,
            hyp_fail_safe = 1,
            hyp_stuck = 250,
            stuck_threshold = None,
            reward_shaping=True,
            max_episode_steps = 100,
            n_plots = None,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            ticks = False,
            dtype= torch.float,
            device = torch.device('cpu')
            ):
        self.dtype = dtype
        self.device = device
        self.n_envs = n_envs

        self.dimension = 2
        self.dof = self.n_links = n_links
        self.joint_id = torch.arange(self.n_links,dtype=int,device=device)
        self.n_obs = n_obs

        link_Z = torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]],dtype=dtype,device=device)
        self.link_zonos = [zp.polyZonotope(link_Z,0)]*n_links 
        link_Z = link_Z.unsqueeze(0).repeat(n_envs,1,1)
        self.__link_zonos = [zp.batchZonotope(link_Z)]*n_links 
        self.link_polygon = [zono[0].project([0,1]).polygon() for zono in self.__link_zonos]
        self.P0 = [torch.tensor([0.0,0.0,0.0],dtype=dtype,device=device)]+[torch.tensor([1.0,0.0,0.0],dtype=dtype,device=device)]*(n_links-1)
        self.R0 = [torch.eye(3,dtype=dtype,device=device)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links,dtype=dtype,device=device)
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]],dtype=dtype,device=device)
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)

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
        self.obstacle_config = {'side_length':0.1*torch.eye(2,dtype=dtype,device=device).unsqueeze(0).repeat(n_envs,1,1), 'zero_pad': torch.zeros(n_envs,3,1,dtype=dtype,device=device)}
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
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
        self.discount = 1

        self.fig = None
        self.render_flag = True
        self._frame_steps = 0
        assert FO_render_level<4
        self.FO_render_level = FO_render_level
        self.ticks = ticks

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = torch.zeros(self.n_envs,dtype=int,device=device)
        
        self.get_plot_grid_size(n_plots)
        self.reset()
    

    def __reset(self,idx):
        n_envs = idx.sum()
        self.qpos[idx] = torch.rand(n_envs,self.n_links,dtype=self.dtype,device=self.device)*2*self.PI - self.PI
        self.qpos_int[idx] = self.qpos[idx]
        self.qvel[idx] = torch.zeros(n_envs,self.n_links,dtype=self.dtype,device=self.device)
        self.qpos_prev[idx] = self.qpos[idx]
        self.qvel_prev[idx] = self.qvel[idx]
        self.qgoal[idx] = torch.rand(n_envs,self.n_links,dtype=self.dtype,device=self.device)*2*self.PI - self.PI

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
        self.qpos = torch.rand(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)*2*self.PI - self.PI
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)
        self.qvel_int = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = torch.rand(self.n_envs,self.n_links,dtype=self.dtype,device=self.device)*2*self.PI - self.PI

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
            obs_Z = torch.zeros(self.n_envs,3,3,dtype=self.dtype,device=self.device)
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
                self.one_time_patches[b].remove()
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
            obs = torch.cat((torch.cat((po,self.obstacle_config['side_length']),1),self.obstacle_config['zero_pad']),-1)
            obs = zp.batchZonotope(obs)
            for j in range(self.n_links):            
                buff = link_init[j]-obs
                _,bi = buff.project([0,1]).polytope()
                if any(bi.min(1).values > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,bg = buff.project([0,1]).polytope()   
                if any(bg.min(1).values > -1e-5):
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
            self.obs_zonos.append(obs)

        self.fail_safe_count = torch.zeros(self.n_envs,dtype=int,device=self.device)
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.one_time_patches[b].remove()
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
        r,th = torch.rand(2,n_envs,dtype=self.dtype,device=self.device)
        #obs_pos = torch.rand(n_envs,2)*2*self.n_links-self.n_links
        obs_pos = (3/4*self.n_links*r*torch.vstack((torch.cos(2*self.PI*th),torch.sin(2*self.PI*th)))).T
        obs_Z = torch.cat((torch.cat((obs_pos.unsqueeze(1),self.obstacle_config['side_length'][:n_envs]),1),self.obstacle_config['zero_pad'][:n_envs]),-1)
        obs = zp.batchZonotope(obs_Z)
        safe_flag = torch.zeros(len(idx),dtype=bool,device=self.device)
        safe_flag[idx] = True
        for j in range(self.n_links):            
            buff = link_init[j][idx]-obs
            _,bi = buff.project([0,1]).polytope()
            buff = link_goal[j][idx]-obs
            _,bg = buff.project([0,1]).polytope()   
            safe_flag[idx] *= ((bi.min(1).values < -1e-5) * (bg.min(1).values < -1e-5)) # Ture for safe envs, -1e-6: more conservative, 1e-6: less conservative

        return obs_Z[safe_flag[idx]], safe_flag

    def step(self,ka,flag=None):
        if flag is None:
            self.step_flag = torch.zeros(self.n_envs,dtype=int,device=self.device)
        else:
            self.step_flag = flag.to(dtype=int,device=self.device).detach()
        self.safe = self.step_flag <= 0
        # -torch.pi<qvel+k*T_PLAN < torch.pi
        # (-torch.pi-qvel)/T_PLAN < k < (torch.pi-qvel)/T_PLAN
        ka = ka.detach()
        self.ka = ka.clamp((-self.PI-self.qvel)/T_PLAN,(self.PI-self.qvel)/T_PLAN) # velocity clamp
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
        self.reward_com *= self.discount
        self.reward_com += self.reward
        self.done = self.success + self.collision

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
                'step_flag':int(self.step_flag[idx])
                }
            if self.collision[idx]:
                collision_info = {
                    'qpos_init':self.qpos_int[idx],
                    'qvel_int':torch.zeros(self.n_links),
                    'obs_pos':[self.obs_zonos[o].center[idx,:2] for o in range(self.n_obs)],
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
            observation['obstacle_pos']= torch.cat([self.obs_zonos[o].center[:,:2].unsqueeze(1) for o in range(self.n_obs)],1)
            observation['obstacle_size'] = torch.cat([self.obs_zonos[o].generators[:,[0,1],[0,1]].unsqueeze(1) for o in range(self.n_obs)],1)
        return observation



    def get_reward(self, action, qpos=None, qgoal=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None or qgoal is None:
            qpos = self.qpos
            qgoal = self.qgoal
        
        self.goal_dist = torch.linalg.norm(wrap_to_pi(qpos-qgoal),dim=-1)
        self.success = self.goal_dist < self.goal_threshold 
        success = self.success.to(dtype=self.dtype)
        
        reward = 0.0

        # Return the sparse reward if using sparse_rewards
        if not self.reward_shaping:
            reward -= self.hyp_collision * self.collision
            reward += success - 1 + self.hyp_success * success
            return reward

        # otherwise continue to calculate the dense reward
        # reward for position term
        reward -= self.hyp_dist_to_goal * self.goal_dist
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
                        _,b = zp.batchZonotope(buff).project([0,1]).polytope()
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
                        _,b = buff.project([0,1]).polytope()
                        unsafe += b.min(dim=-1)[0] > 1e-6
        return unsafe   

    def render(self,FO_link=None,show=True,dpi=None,save_kwargs=None):
        if self.render_flag:
            if self.fig is None:
                if show:
                    plt.ion()
                if self.n_plots == 1:
                    self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8],dpi=dpi)
                    self.axs = np.array([self.fig.gca()])
                else:
                    self.fig, self.axs = plt.subplots(self.plot_grid_size[0],self.plot_grid_size[1],figsize=[self.plot_grid_size[1]*6.4/2,self.plot_grid_size[0]*4.8/2],dpi=dpi)
                if not self.ticks: 
                    plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                plt.setp(self.axs,xticks=[],xticklabels=[],yticks=[],yticklabels=[])
                if save_kwargs is not None:
                    os.makedirs(save_kwargs['save_path'],exist_ok=True)
                    if self.n_plots == 1:
                        fontsize = 10
                    else:
                        fontsize = 7 + self.plot_grid_size[0]
                    self.axs.reshape(self.plot_grid_size)[0,-1].set_title(save_kwargs['text'],fontsize=fontsize,loc='right')

            self.render_flag = False
            self.one_time_patches, self.FO_patches, self.link_patches= [], [], []

            R_q = self.rot(self.qgoal[:self.n_plots])
            link_goal_polygons = []
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[:,j]
                link_goal_polygons.append((self.link_polygon[j]@R.transpose(-1,-2)[:,:2,:2] + P[:,:2].unsqueeze(-2)).cpu())

            obs_patches = []
            for o in range(self.n_obs):
                obs_patches.append(self.obs_zonos[o][:self.n_plots].polygon(nan=False).cpu().numpy())
                #patches.Polygon(p,alpha=alpha,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth)
  
            for b, ax in enumerate(self.axs.flat):
                one_time_patch = []
                for j in range(self.n_links):
                    one_time_patch.append(patches.Polygon(link_goal_polygons[j][b],alpha = .5, facecolor='gray',edgecolor='gray',linewidth=.2))
                for o in range(self.n_obs):
                    one_time_patch.append(patches.Polygon(obs_patches[o][b],alpha = .5, facecolor='red',edgecolor='red',linewidth=.2))
                self.one_time_patches.append(PatchCollection(one_time_patch, match_original=True))
                ax.add_collection(self.one_time_patches[b])
                self.FO_patches.append(ax.add_collection(PatchCollection([])))
                self.link_patches.append(ax.add_collection(PatchCollection([])))
        
        if FO_link is not None: 
            FO_patch = []
            FO_render_idx = (self.fail_safe_count[:self.n_plots] == 0).nonzero().reshape(-1) # NOTE: deal with _reset()
            if FO_render_idx.numel() != 0:
                g_ka = self.PI/24 
                FO_link_polygons = []
                for j in range(self.n_links):
                    FO_patch = []
                    PROPERTY_ID.update(self.n_links)
                    FO_link_slc = FO_link[j][FO_render_idx].to(dtype=self.dtype,device=self.device).slice_all_dep((self.ka[FO_render_idx]/g_ka).unsqueeze(1).repeat(1,100,1))
                    zp.reset()
                    FO_link_polygons.append(FO_link_slc.polygon().detach())
            
                for idx, b in enumerate(FO_render_idx.tolist()):
                    self.FO_patches[b].remove()
                    FO_patch = []
                    if self.FO_render_level == 3:
                        for j in range(self.n_links):
                            FO_patch.extend([patches.Polygon(polygon,alpha=0.1,edgecolor='green',facecolor='none',linewidth=.2) for polygon in FO_link_polygons[j][idx]])
                    elif self.FO_render_level == 2:
                        for j in range(self.n_links):
                            FO_patch.append(patches.Polygon(FO_link_polygons[j][idx].reshape(-1,2),alpha=0.3,edgecolor='none',facecolor='green',linewidth=.2))  
                    '''
                    elif self.FO_render_level == 1:
                        FO_link_polygons_temp = []
                        for j in range(self.n_links):
                            FO_link_polygons_temp.append(FO_link_polygons[j][idx].reshape(-1,2))
                        FO_patch.append(patches.Polygon(torch.vstack(FO_link_polygons_temp),alpha=0.3,edgecolor='none',facecolor='green',linewidth=.2))
                    '''
                    self.FO_patches[b] = PatchCollection(FO_patch, match_original=True)
                    self.axs.flat[b].add_collection(self.FO_patches[b])
  
        if self.interpolate:
            
            timesteps = int(T_PLAN/T_FULL*self.T_len) # NOTE
            if show and save_kwargs is None:
                plot_freq = 1
                R_q = self.rot(self.qpos_to_peak[1:,:self.n_plots])
            elif show:
                plot_freq = timesteps//save_kwargs['frame_rate']
                R_q = self.rot(self.qpos_to_peak[1:,:self.n_plots])    
            else:
                plot_freq = 1
                t_idx = torch.arange(timesteps+1,device=self.device)%(timesteps//save_kwargs['frame_rate'] ) == 1
                R_q = self.rot(self.qpos_to_peak[t_idx,:self.n_plots])
                timesteps = len(R_q)
            link_trace_polygons = torch.zeros(timesteps,self.n_plots,5*self.n_links,2,dtype=self.dtype,device='cpu')
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,1,3,dtype=self.dtype,device=self.device)
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,:,j]
                link_trace_polygons[:,:,5*j:5*(j+1)] = (self.link_polygon[j]@R.transpose(-1,-2)[:,:,:2,:2] + P[:,:,:2].unsqueeze(-2)).cpu()

            for t in range(timesteps):
                for b, ax in enumerate(self.axs.flat):
                    self.link_patches[b].remove()
                    link_trace_patches = [patches.Polygon(link_trace_polygons[t,b,5*j:5*(j+1)],alpha = .5, facecolor='blue',edgecolor='blue',linewidth=.2) for j in range(self.n_links)]
                    self.link_patches[b] = PatchCollection(link_trace_patches, match_original=True)
                    ax.add_collection(self.link_patches[b])
                
                    ax_scale = 1.2
                    axis_lim = ax_scale*self.n_links
                    ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                if save_kwargs is not None and t%plot_freq == 0:
                    filename = "/frame_"+"{0:04d}".format(self._frame_steps)                    
                    self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                    self._frame_steps+=1
        else:
            R_q = self.rot(self.qpos[:self.n_plots])
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            link_trace_polygons = []
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_trace_polygons.append((self.link_polygon[j]@R.transpose(-1,-2)[:,:2,:2] + P[:,:2].unsqueeze(-2)).cpu())              
            for b, ax in enumerate(self.axs.flat):
                self.link_patches[b].remove()
                link_trace_patches = [patches.Polygon(link_trace_polygons[j][b],alpha = .5, facecolor='blue',edgecolor='blue',linewidth=.2) for j in range(self.n_links)]
                self.link_patches[b] = PatchCollection(link_trace_patches, match_original=True)
                ax.add_collection(self.link_patches[b])
                ax_scale = 1.2
                axis_lim = ax_scale*self.n_links
                ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if save_kwargs is not None:
                filename = "/frame_"+"{0:04d}".format(self._frame_steps)                    
                self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                self._frame_steps+=1

        if self.done.any():
            reset_flag = self.done[:self.n_plots].nonzero().reshape(-1)
        
            R_q = self.rot(self.qgoal[reset_flag])
            link_goal_polygons = []
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_goal_polygons.append((self.link_polygon[j]@R.transpose(-1,-2)[:,:2,:2] + P[:,:2].unsqueeze(-2)).cpu())   

            for idx, b in enumerate(reset_flag.tolist()):
                self.one_time_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
                ax = self.axs.flat[b]
                one_time_patch = []
                one_time_patch.extend([self.obs_zonos[o][b].polygon_patch(edgecolor='red',facecolor='red') for o in range(self.n_obs)])
                one_time_patch.extend([patches.Polygon(link_goal_polygons[j][idx],alpha = .5, facecolor='gray',edgecolor='gray',linewidth=.2) for j in range(self.n_links)])
                self.one_time_patches[b] = PatchCollection(one_time_patch, match_original=True)
                ax.add_collection(self.one_time_patches[b])
                self.FO_patches[b] = ax.add_collection(PatchCollection([]))
                self.link_patches[b] = ax.add_collection(PatchCollection([]))  
    
    def close(self):
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.one_time_patches[b].remove()
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

if __name__ == '__main__':
    import time
    from zonopy.environments.arm_2d import Arm_2D
    n_envs = 3
    env = Parallel_Arm_2D(n_envs=n_envs,interpolate=True,n_plots=2)
    env1 = Arm_2D()
    '''
    ts = time.time()
    for _ in range(n_envs):
        env1.reset()
    print(f'serial reset: {time.time()-ts}')
    ts = time.time()
    env.reset()
    print(f'parallel reset : {time.time()-ts}')
    '''
    for i in range(20):
        observations, rewards, dones, infos = env.step(torch.ones(env.n_envs,env.n_links))
        env.render()
    
    import pdb;pdb.set_trace()