import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
#from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Arm_3D:
    def __init__(self,
            n_links=7, # number of links
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
            hyp_fail_safe = - 1,
            reward_shaping=True,
            max_episode_steps = 100,
            scale = 10
            ):
        self.dimension = 3
        self.n_links = n_links
        self.n_obs = n_obs
        self.scale = scale

        #### load
        params, _ = zp.load_sinlge_robot_arm_params('Kinova3')
        self.link_zonos = params['link_zonos'] # NOTE: zonotope, should it be poly zonotope?
        self.link_zonos = [(self.scale*self.link_zonos[j]).to_polyZonotope() for j in range(n_links)]
        self.P0 = [self.scale*P for P in params['P']]
        self.R0 = params['R']
        self.joint_axes = torch.vstack(params['joint_axes'])
        
        self.pos_lim = torch.tensor(params['pos_lim'])
        self.vel_lim = torch.tensor(params['vel_lim'])
        self.tor_lim = torch.tensor(params['tor_lim'])
        self.lim_flag = torch.tensor(params['lim_flag'],dtype=bool)
        self.pos_sampler = torch.distributions.Uniform(self.pos_lim[:,1],self.pos_lim[:,0])
        self.full_radius = self.scale*0.8
        #self.full_radius = sum([(abs(self.P0[j])).max() for j in range(self.n_links)])        
        #### load

        self.fig_scale = 1
        self.interpolate = interpolate
        self.PI = torch.tensor(torch.pi)
        if interpolate:
            self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len):] - T_PLAN

        self.obs_buffer_length = torch.tensor([0.001,0.001])
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
        self.collision_threshold = collision_threshold
        
        self.goal_threshold = goal_threshold
        self.hyp_effort = hyp_effort
        self.hyp_dist_to_goal = hyp_dist_to_goal
        self.hyp_collision = hyp_collision
        self.hyp_success = hyp_success
        self.hyp_fail_safe = hyp_fail_safe
        self.reward_shaping = reward_shaping
        self.discount = 1

        self.fig = None
        self.render_flag = True

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.FO_freq = 10
        self.reset()
    def reset(self):
        self.qpos = self.pos_sampler.sample()
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_links)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = self.pos_sampler.sample()
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
                obs_pos = torch.rand(3)*2*8-8

                # NOTE
                #rho, th, psi 
                obs = zp.zonotope(torch.vstack((obs_pos,torch.eye(3))))
                
                
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
            self.obs_patches.remove()
            self.link_goal_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0
        
        self.reward_com = 0

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
                
                self.collision = self.collision_check(torch.vstack((self.qpos_to_peak,self.qpos_to_brake[1:])))
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
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_brake = wrap_to_pi(self.qpos + self.qvel*(T_FULL-T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2)
                self.qvel_brake = torch.zeros(self.n_links)

                self.collision = self.collision_check(torch.vstack((self.qpos,self.qpos_brake)))
            else:
                self.fail_safe_count +=1
                self.qpos = torch.clone(self.qpos_brake)
                self.qvel = torch.clone(self.qvel_brake) 

        self._elapsed_steps += 1
        self.reward = self.get_reward(ka) # NOTE: should it be ka or self.ka ??
        self.reward_com *= self.discount
        self.reward_com += self.reward
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
                'obs_pos':[self.obs_zonos[o].center for o in range(self.n_obs)],
                'qgoal':self.qgoal
            }
            info['collision_info'] = collision_info
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not self.done
            self.done = True            
        info['episode'] = {"r":self.reward_com,"l":self._elapsed_steps}
        return info

    def get_observations(self):
        observation = {'qpos':self.qpos,'qvel':self.qvel,'qgoal':self.qgoal}
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.vstack([self.obs_zonos[o].center for o in range(self.n_obs)])
            observation['obstacle_size'] = torch.vstack([torch.diag(self.obs_zonos[o].generators) for o in range(self.n_obs)])
        return observation

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
        # Add fail-safe if needed
        reward += self.hyp_fail_safe * (1-bool(self.safe))
        # Add success if wanted
        reward += self.hyp_success * success

        return float(reward)   

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
                        _,b = buff.polytope()
                        unsafe = b.min(dim=-1)[0]>1e-6
                        if any(unsafe):
                            self.qpos_collision = qs[unsafe]
                            return True

            else:
                time_steps = 1
                R, P = torch.eye(3), torch.zeros(3)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[j]
                    link = (R@self.link_zonos[j]+P).to_zonotope()
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.polytope()
                        if min(b) > 1e-6:
                            self.qpos_collision = qs
                            return True
  
        return False

    def render(self,FO_link=None):
        if self.render_flag:
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8])
                self.ax = a3.Axes3D(self.fig)
            
            self.render_flag = False
            self.FO_patches = self.ax.add_collection3d(Poly3DCollection([]))
            self.link_patches = self.ax.add_collection3d(Poly3DCollection([]))
            
            obs_patches = []
            for o in range(self.n_obs):
                obs_patches.extend(self.obs_zonos[o].polyhedron_patch())
            self.obs_patches = self.ax.add_collection3d(Poly3DCollection(obs_patches,edgecolor='red',facecolor='red',alpha=0.2,linewidths=0.2))

            goal_patches = []
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3), torch.zeros(3)            
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polyhedron_patch()
                goal_patches.extend(link_patch)
            self.link_goal_patches = self.ax.add_collection3d(Poly3DCollection(goal_patches,edgecolor='gray',facecolor='gray',alpha=0.15,linewidths=0.5))
                
        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count != 1:
                g_ka = self.PI/24
                self.FO_patches.remove()
                for j in range(self.n_links): 
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)).reduce(4)
                    for t in range(100): 
                        if t % self.FO_freq == 0:
                            FO_patch = FO_link_slc[t].polyhedron_patch()
                            FO_patches.extend(FO_patch)
                self.FO_patches = self.ax.add_collection3d(Poly3DCollection(FO_patches,alpha=0.05,edgecolor='green',facecolor='green',linewidths=0.2)) 

        if self.interpolate:
            R_q = self.rot(self.qpos_to_peak)
            time_steps = int(T_PLAN/T_FULL*self.T_len)
            for t in range(time_steps):
                R, P = torch.eye(3), torch.zeros(3)
                link_patches = []
                self.link_patches.remove()
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[t,j]
                    link_patch = (R@self.link_zonos[j]+P).to_zonotope().polyhedron_patch()
                    link_patches.extend(link_patch)            
                self.link_patches = self.ax.add_collection(Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5))
                self.ax.set_xlim([-self.full_radius,self.full_radius])
                self.ax.set_ylim([-self.full_radius,self.full_radius])
                self.ax.set_zlim([-self.full_radius,self.full_radius])
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
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polyhedron_patch()
                link_patches.extend(link_patch)
            self.link_patches = self.ax.add_collection3d(Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5, linestyles=':'))
            self.ax.set_xlim([-self.full_radius,self.full_radius])
            self.ax.set_ylim([-self.full_radius,self.full_radius])
            self.ax.set_zlim([-self.full_radius,self.full_radius])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def rot(self,q=None):
        if q is None:
            q = self.qpos
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w


if __name__ == '__main__':

    env = Arm_3D(n_obs=3,T_len=20,interpolate=True)
    for _ in range(3):
        for _ in range(10):
            env.step(torch.rand(7))
            env.render()
            env.reset()