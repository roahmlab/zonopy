import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
#from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3

T_PLAN, T_FULL = 0.5, 1

class Arm_3D:
    def __init__(self,n_links=7,n_obs=1,T_len=50,intermediate = True):
        self.dimension = 3
        self.n_links = n_links
        self.n_obs = n_obs

        #### load
        params, _ = zp.load_sinlge_robot_arm_params('Kinova3')
        self.link_zonos = params['link_zonos'] # NOTE: zonotope, should it be poly zonotope?
        self.link_zonos = [self.link_zonos[j].to_polyZonotope() for j in range(n_links)]
        self.P0 = params['P']
        self.R0 = params['R']
        self.joint_axes = torch.vstack(params['joint_axes'])
        #### load

        self.fig_scale = 1
        self.intermediate = intermediate
        self.PI = torch.tensor(torch.pi)
        if intermediate:
            self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len)+1:] - T_PLAN

        self.reset()
    def reset(self):
        self.qpos = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zeros(self.n_links)
        self.qgoal = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.fail_safe_count = 0
        if self.intermediate:
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
                obs_pos = torch.rand(3)*2*0.8-0.8
                obs = zp.zonotope(torch.vstack((obs_pos,0.1*torch.eye(3))))
                safe_flag = True
                for j in range(self.n_links):
                    buff = link_init[j]-obs
                    A,b = buff.polytope() # NOTE: self.comb
                    if max(A@torch.zeros(3)-b) < 1e-6:
                        safe_flag = False
                        break
                    buff = link_goal[j]-obs
                    A,b = buff.polytope() # NOTE: self.comb
                    if max(A@torch.zeros(3)-b) < 1e-6:
                        safe_flag = False
                        break

                if safe_flag:
                    self.obs_zonos.append(obs)
                    break
        self.render_flag = True
        self.done = False

    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos
        self.qvel = qvel
        self.qgoal = qgoal
        self.fail_safe_count = 0
        if self.intermediate:
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
                A,b = buff.polytope()
                if max(A@torch.zeros(3)-b) < 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                A,b = buff.polytope()
                if max(A@torch.zeros(3)-b) < 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
            self.obs_zonos.append(obs)

        self.render_flag = True
        self.done = False

    def step(self,ka,safe=0):
        self.safe = safe == 0
        self.ka = ka
        if self.intermediate:
            if self.safe:
                self.fail_safe_count = 0
                # to peak
                self.qpos_to_peak = self.qpos + torch.outer(self.t_to_peak,self.qvel) + .5*torch.outer(self.t_to_peak**2,ka)
                self.qvel_to_peak = self.qvel + torch.outer(self.t_to_peak,ka)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                #to stop
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_to_brake = self.qpos + torch.outer(self.t_to_brake,self.qvel) + .5*torch.outer(self.t_to_brake**2,bracking_accel)
                self.qvel_to_brake = self.qvel + torch.outer(self.t_to_brake,bracking_accel)
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
                self.qpos += self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
                self.qvel += ka*T_PLAN
                self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
                self.qvel_brake = torch.zeros(self.n_links)
            else:
                self.fail_safe_count +=1
                self.qpos = torch.clone(self.qpos_brake)
                self.qvel = torch.clone(self.qvel_brake) 

        #goal_distance = torch.linalg.norm(self.qpos_to_peak-self.qgoal,dim=1)
        goal_distance = torch.linalg.norm(self.qpos-self.qgoal)
        self.done = goal_distance.min() < 0.15
        if self.done:
            self.until_goal = goal_distance.argmin()
        return self.done


    def render(self,FO_link=None):
        if self.render_flag:
            plt.ion()
            self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8])
            self.ax = a3.Axes3D(self.fig)
            self.render_flag = False
            self.FO_patches = self.ax.add_collection3d(Poly3DCollection([]))
            self.link_patches = self.ax.add_collection3d(Poly3DCollection([]))

            obs_patches = []
            for o in range(self.n_obs):
                obs_patches.extend(self.obs_zonos[o].polyhedron_patch())
            self.ax.add_collection3d(Poly3DCollection(obs_patches,edgecolor='red',facecolor='red',alpha=0.2,linewidths=0.2))
            
            goal_patches = []
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3), torch.zeros(3)            
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polyhedron_patch()
                goal_patches.extend(link_patch)
            self.ax.add_collection3d(Poly3DCollection(goal_patches,edgecolor='gray',facecolor='gray',alpha=0.15,linewidths=0.5))
                
        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count != 1:
                g_ka = torch.minimum(torch.maximum(self.PI/24,abs(self.qvel/3)),self.PI/3)
                self.FO_patches.remove()
                for j in range(self.n_links): 
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)).reduce(4)
                    for t in range(100): 
                        import pdb;pdb.set_trace()
                        FO_patch = FO_link_slc[t].polyhedron_patch()
                        FO_patches.extend(FO_patch)
                self.FO_patches = Poly3DCollection(FO_patches,alpha=0.05,edgecolor='green',facecolor='green',linewidths=0.2)
                self.ax.add_collection3d(self.FO_patches)            

        if self.intermediate:
            R_q = self.rot(self.qpos_to_peak)
            if not self.done:
                time_steps = int(T_PLAN/T_FULL*self.T_len)
            else:
                time_steps = self.until_goal
            for t in range(time_steps):
                R, P = torch.eye(3), torch.zeros(3)
                link_patches = []
                self.link_patches.remove()
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[t,j]
                    link_patch = (R@self.link_zonos[j]+P).to_zonotope().polyhedron_patch()
                    link_patches.extend(link_patch)            
                self.link_patches = Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5)
                self.ax.add_collection(self.link_patches)
                self.ax.set_xlim([-0.8,0.8])
                self.ax.set_ylim([-0.8,0.8])
                self.ax.set_zlim([-0.8,0.8])
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
            self.link_patches = Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5, linestyles=':')
            self.ax.add_collection3d(self.link_patches)
            self.ax.set_xlim([-0.8,0.8])
            self.ax.set_ylim([-0.8,0.8])
            self.ax.set_zlim([-0.8,0.8])
            #plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
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

    env = Arm_3D(n_obs=3,T_len=20,intermediate=True)
    for _ in range(100):
        env.step(torch.rand(7))
        env.render()