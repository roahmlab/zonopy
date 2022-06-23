import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
#from .utils import locate_figure

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Arm_2D:
    def __init__(self,
            n_links=2, # number of links
            n_obs=1, # number of obstacles
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.05, # goal threshold
            hyp_effort = 1.0, # hyperpara
            hyp_dist_to_goal = 1.0,
            hyp_collision = -200,
            hyp_success = 50,
            reward_shaping=True,
            max_episode_steps = 100
            ):

        self.dimension = 2
        self.n_links = n_links
        self.n_obs = n_obs
        self.link_zonos = [zp.polyZonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]]),0)]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.interpolate = interpolate
        self.PI = torch.tensor(torch.pi)


        if interpolate:
            self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len)+1:] - T_PLAN
        
        

        self.obs_buffer_length = torch.tensor([0.001,0.001])
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
        self.collision_threshold = collision_threshold
        
        self.goal_threshold = goal_threshold
        self.hyp_effort = hyp_effort
        self.hyp_dist_to_goal = hyp_dist_to_goal
        self.hyp_collision = hyp_collision
        self.hyp_success = hyp_success
        self.reward_shaping = reward_shaping

        self.fig = None
        self.render_flag = True

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.reset()
    def reset(self):
        self.qpos = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_links)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.fail_safe_count = 0
        if self.interpolate:
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
            self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links)            

        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3), torch.zeros(3)       
        Rg, Pg = torch.eye(3), torch.zeros(3)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link = (Ri@self.link_zonos[j]+Pi).to_zonotope()
            link_init.append(link)
            link = (Rg@self.link_zonos[j]+Pg).to_zonotope()
            link_goal.append(link)

        for _ in range(self.n_obs):
            while True:
                r,th = torch.rand(2)
                #obs_pos = torch.rand(2)*2*self.n_links-self.n_links
                obs_pos = 3/4*self.n_links*r*torch.tensor([torch.cos(2*torch.pi*th),torch.sin(2*torch.pi*th)])
                obs = torch.hstack((torch.vstack((obs_pos,0.1*torch.eye(2))),torch.zeros(3,1)))
                obs = zp.zonotope(obs)
                safe_flag = True
                for j in range(self.n_links):
                    buff = link_init[j]-obs
                    _,b = buff.project([0,1]).polytope()
                    if min(b) > 1e-6:
                        safe_flag = False
                        break
                    buff = link_goal[j]-obs
                    _,b = buff.project([0,1]).polytope()
                    if min(b) > 1e-6:
                        safe_flag = False
                        break

                if safe_flag:
                    self.obs_zonos.append(obs)
                    break

        self.fail_safe_count = 0
        if self.render_flag == False:
            self.one_time_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0
        
        return self.get_observations()


    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = qvel
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = qgoal
        self.fail_safe_count = 0
        if self.interpolate:
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
            self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links)            
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3), torch.zeros(3)       
        Rg, Pg = torch.eye(3), torch.zeros(3)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link = (Ri@self.link_zonos[j]+Pi).to_zonotope()
            link_init.append(link)
            link = (Rg@self.link_zonos[j]+Pg).to_zonotope()
            link_goal.append(link)

        for pos in obs_pos:
            obs = torch.hstack((torch.vstack((pos,0.1*torch.eye(2))),torch.zeros(3,1)))
            obs = zp.zonotope(obs)
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.project([0,1]).polytope()
                if min(b) > 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.project([0,1]).polytope()
                if min(b) > 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
            self.obs_zonos.append(obs)

        self.fail_safe_count = 0
        if self.render_flag == False:
            self.one_time_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0

        return self.get_observations()

    def step(self,ka,flag=0):
        self.step_flag = flag
        self.safe = flag <= 0
        # -torch.pi<qvel+k*T_PLAN < torch.pi
        # (-torch.pi-qvel)/T_PLAN < k < (torch.pi-qvel)/T_PLAN
        self.ka = ka.clamp((-torch.pi-self.qvel)/T_PLAN,(torch.pi-self.qvel)/T_PLAN) # velocity clamp
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        if self.interpolate:
            if self.safe:
                self.fail_safe_count = 0
                
                # to peak
                self.qpos_to_peak = wrap_to_pi(self.qpos + torch.outer(self.t_to_peak,self.qvel) + .5*torch.outer(self.t_to_peak**2,self.ka))
                self.qvel_to_peak = self.qvel + torch.outer(self.t_to_peak,self.ka)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                #to stop
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_to_brake = wrap_to_pi(self.qpos + torch.outer(self.t_to_brake,self.qvel) + .5*torch.outer(self.t_to_brake**2,bracking_accel))
                self.qvel_to_brake = self.qvel + torch.outer(self.t_to_brake,bracking_accel)
                
                self.collision = self.collision_check(torch.vstack((self.qpos_to_peak,self.qpos_to_brake)))
            else:
                self.fail_safe_count +=1
                self.qpos_to_peak = torch.clone(self.qpos_to_brake)
                self.qvel_to_peak = torch.clone(self.qvel_to_brake)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
                self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)
        else:
            if self.safe:
                self.fail_safe_count = 0
                self.qpos += wrap_to_pi(self.qvel*T_PLAN + 0.5*self.ka*T_PLAN**2)
                self.qvel += self.ka*T_PLAN
                self.qpos_brake = wrap_to_pi(self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN))
                self.qvel_brake = torch.zeros(self.n_links)

                self.collision = self.collision_check(torch.vstack((self.qpos,self.qpos_brake)))
            else:
                self.fail_safe_count +=1
                self.qpos = torch.clone(self.qpos_brake)
                self.qvel = torch.clone(self.qvel_brake) 
        
        '''
        goal_distance = torch.linalg.norm(wrap_to_pi(self.qpos_to_peak-self.qgoal),dim=1)
        self.done = goal_distance.min() < 0.05
        if self.done:
            self.until_goal = goal_distance.argmin()
        '''
        self._elapsed_steps += 1
        self.reward = self.get_reward(ka) # NOTE: should it be ka or self.ka ??
        self.done = self.success or self.collision
        observations = self.get_observations()
        info = self.get_info()
        return observations, self.reward, self.done, info

    def get_info(self):
        info ={'is_success':self.success,'collision':self.collision,'safe_flag':self.safe,'step_flag':self.step_flag}
        if self.collision:
            collision_info = {
                'qpos_collision':self.qpos_collision,
                'qpos_init':self.qpos_int,
                'qvel_int':torch.zeros(self.n_links),
                'obs_pos':[self.obs_zonos[o].center[:2] for o in range(self.n_obs)],
                'qgoal':self.qgoal
            }
            info['collision_info'] = collision_info
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not self.done
            self.done = True            
        info['episode'] = {"r":self.reward,"l":self._elapsed_steps}
        return info

    def get_observations(self):
        observation = {'qpos':self.qpos,'qvel':self.qvel,'qgoal':self.qgoal}
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.vstack([self.obs_zonos[o].center[:2] for o in range(self.n_obs)])
            observation['obstacle_size'] = torch.vstack([torch.diag(self.obs_zonos[o].generators) for o in range(self.n_obs)])
        return observation

    def collision_check(self,qs):

        if self.check_collision:
            
            R_q = self.rot(qs)
            if len(R_q.shape) == 4:
                time_steps = len(R_q)
                R, P = torch.eye(3), torch.zeros(3)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,j]
                    link =zp.batchZonotope(self.link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1))
                    link = R@link+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.project([0,1]).polytope()
                        unsafe = b.min(dim=-1)[0]>1e-6
                        if any(unsafe):
                            self.qpos_collision = qs[unsafe]
                            return True

            else:
                time_steps = 1
                R, P = torch.eye(3), torch.zeros(3)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q
                    link = (R@self.link_zonos[j]+P).to_zonotope()
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.project([0,1]).polytope()
                        if min(b) > 1e-6:
                            self.qpos_collision = qs
                            return True
  
        return False

    def get_reward(self, action, qpos=None, qgoal=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None or qgoal is None:
            qpos = self.qpos
            qgoal = self.qgoal

        self.goal_dist = torch.linalg.norm(wrap_to_pi(qpos-qgoal))
        self.success = self.goal_dist < self.goal_threshold 
        success = self.success.to(dtype=torch.get_default_dtype())
        
        reward = 0.0

        # Return the sparse reward if using sparse_rewards
        if not self.reward_shaping:
            reward += self.hyp_collision * torch.tensor(self.collision,dtype=torch.get_default_dtype())
            reward += success - 1 + self.hyp_success * success
            return reward

        # otherwise continue to calculate the dense reward
        # reward for position term
        reward -= self.hyp_dist_to_goal * self.goal_dist
        # reward for effort
        reward -= self.hyp_effort * torch.linalg.norm(action)
        # Add collision if needed
        reward += self.hyp_collision * torch.tensor(self.collision,dtype=torch.get_default_dtype())
        # Add success if wanted
        reward += self.hyp_success * success
        return reward       


    def render(self,FO_link=None):
        
        if self.render_flag:
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8])
                #self.fig.canvas.manager.window.move(100,400)
                self.ax = self.fig.gca()

            self.render_flag = False
            self.one_time_patches = self.ax.add_collection(PatchCollection([]))
            self.FO_patches = self.ax.add_collection(PatchCollection([]))
            self.link_patches = self.ax.add_collection(PatchCollection([]))
            one_time_patches = []
            for o in range(self.n_obs):
                one_time_patches.append(self.obs_zonos[o].polygon_patch(edgecolor='red',facecolor='red'))
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3), torch.zeros(3)            
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='gray',facecolor='gray')
                one_time_patches.append(link_patch)
            self.one_time_patches = PatchCollection(one_time_patches, match_original=True)
            self.ax.add_collection(self.one_time_patches)

        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count != 1:
                g_ka = torch.maximum(self.PI/24,abs(self.qvel_prev/3)) # NOTE: is it correct?
                self.FO_patches.remove()
                for j in range(self.n_links):
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)) 
                    if self.check_collision_FO:
                        c_link_slc = FO_link[j].center_slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1))
                        for o,obs in enumerate(self.obs_zonos):
                            obs_Z = obs.Z[:,:self.dimension].unsqueeze(0).repeat(100,1,1)
                            A, b = zp.batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope()
                            cons, _ = torch.max((A@c_link_slc.unsqueeze(-1)).squeeze(-1) - b,-1)
                            for t in range(100):                            
                                if cons[t] < 1e-6:
                                    color = 'red'
                                else:
                                    color = 'green'
                                FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor=color)
                                FO_patches.append(FO_patch)
                    else:
                        for t in range(100): 
                            FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor='green')
                            FO_patches.append(FO_patch)
                self.FO_patches = PatchCollection(FO_patches, match_original=True)
                self.ax.add_collection(self.FO_patches)            

        if self.interpolate:
            R_q = self.rot(self.qpos_to_peak)
            time_steps = int(T_PLAN/T_FULL*self.T_len)
            '''
            if not self.done:
                time_steps = int(T_PLAN/T_FULL*self.T_len)
            else:
                time_steps = self.until_goal
            '''
            for t in range(time_steps):
                R, P = torch.eye(3), torch.zeros(3)
                link_patches = []
                self.link_patches.remove()
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[t,j]
                    link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='blue',facecolor='blue')
                    link_patches.append(link_patch)            
                self.link_patches = PatchCollection(link_patches, match_original=True)
                self.ax.add_collection(self.link_patches)
                ax_scale = 1.2
                axis_lim = ax_scale*self.n_links
                plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        else:
            R_q = self.rot()
            R, P = torch.eye(3), torch.zeros(3)
            link_patches = []
            self.link_patches.remove()
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='blue',facecolor='blue')
                link_patches.append(link_patch)
            self.link_patches = PatchCollection(link_patches, match_original=True)
            self.ax.add_collection(self.link_patches)
            ax_scale = 1.2
            axis_lim = ax_scale*self.n_links
            plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def rot(self,q=None):
        if q is None:
            q = self.qpos
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w
    
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

'''
class Batch_Arm_2D:
    def __init__(self,n_links=2,n_obs=1,n_batches=4):
        self.n_batches = n_batches
        self.n_links = n_links
        self.n_obs = n_obs
        self.link_zonos = [zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]])).to_polyZonotope()]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.reset()
        self.get_plot_grid_size()

    def reset(self):
        self.qpos = torch.rand(self.n_batches,self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zeros(self.n_batches,self.n_links)
        self.qgoal = torch.rand(self.n_batches,self.n_links)*2*torch.pi - torch.pi
        self.break_qpos = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
        self.break_qvel = torch.zeros(self.n_batches,self.n_links)

        self.obs_zonos = []
        for _ in range(self.n_obs):
            obs = torch.tensor([[0.0,0,0],[0.1,0,0],[0,0.1,0]]).repeat(self.n_batches,1,1)
            obs[:,0,:2] = torch.rand(self.n_batches,2)*2*self.n_links-self.n_links
            obs = zp.batchZonotope(obs)
            #if SAFE:
            self.obs_zonos.append(obs)
        self.render_flag = True
        
    def step(self,ka):
        self.qpos += self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
        self.qvel += ka*T_PLAN
        self.break_qpos = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
        self.break_qvel = 0

    def render(self,FO_link=None):
        if self.render_flag:
            plt.ion()
            self.fig, self.axs = plt.subplots(self.plot_grid_size[0],self.plot_grid_size[1],figsize=[self.plot_grid_size[1]*6.4/2,self.plot_grid_size[0]*4.8/2])
            self.render_flag = False
            for b, ax in enumerate(self.axs.flat):
                for o in range(self.n_obs):
                    self.obs_zonos[o][b].plot(ax=ax, edgecolor='red',facecolor='red')
            self.link_patches = [[[] for _ in range(self.n_links)] for _ in range(b+1)]
        
        R_q = self.rot 
        
        for b, ax in enumerate(self.axs.flat):
            R, P = torch.eye(3), torch.zeros(3)  
            #if len(self.link_patches[0]) > 0:
            #ax.clear()
            for j in range(self.n_links):
                if not isinstance(self.link_patches[b][j],list):
                    self.link_patches[b][j].remove()
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[b,j]
                self.link_patches[b][j] = (R@self.link_zonos[j]+P).to_zonotope().plot(ax=ax, edgecolor='blue',facecolor='blue')
            ax_scale = 1.2
            axis_lim = ax_scale*self.n_links
            ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()        
        
    def get_plot_grid_size(self):
        if self.n_batches in (1,2,3):
            self.plot_grid_size = (1, self.n_batches)
        elif self.n_batches < 9:
            self.plot_grid_size = (2, min(self.n_batches//2,3))
        else:
            self.plot_grid_size = (3,3)
            
    @property
    def rot(self):
        w = torch.tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = self.qpos.reshape(self.n_batches,self.n_links,1,1)
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

'''    

if __name__ == '__main__':

    env = Arm_2D(n_obs=2)
    #from zonopy.optimize.armtd import ARMTD_planner
    #planner = ARMTD_planner(env)
    for _ in range(20):
        for _ in range(4):
            #ka, flag = planner.plan(env.qpos,env.qvel,env.qgoal,env.obs_zonos,torch.zeros(2))
            observations, reward, done, info = env.step(torch.rand(2))
            env.render()
            if done:
                env.reset()
                break
            
    '''

    env = Batch_Arm_2D()
    for _ in range(50):
        env.step(torch.rand(env.n_batches,2))
        env.render()
    '''    

