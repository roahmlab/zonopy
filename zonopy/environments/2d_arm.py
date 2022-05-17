import torch 
import zonopy as zp
import matplotlib.pyplot as plt 

T_PLAN, T_FULL = 0.5, 1

class Arm_2D:
    def __init__(self,n_links=2,n_obs=1):
        self.n_links = n_links
        self.n_obs = n_obs
        self.link_zonos = [zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]])).to_polyZonotope()]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.reset()
    def reset(self):
        self.qpos = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zeros(self.n_links)
        self.qgoal = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.obs_zonos = []
        for _ in range(self.n_obs):
            obs_pos = torch.rand(2)*2*self.n_links-self.n_links
            obs = torch.hstack((torch.vstack((obs_pos,0.1*torch.eye(2))),torch.zeros(3,1)))
            obs = zp.zonotope(obs)
            #if SAFE:
            self.obs_zonos.append(obs)
        self.render_flag = True
        
        

    def step(self,ka):
        self.qpos += self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
        self.qvel += ka*T_PLAN
        self.break_qpos = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
        self.break_qvel = torch.zeros(self.n_links)

    def render(self):
        if self.render_flag:
            plt.ion()
            self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8])
            self.ax = self.fig.gca()
            self.render_flag = False
            for o in range(self.n_obs):
                self.obs_zonos[o].plot(ax=self.ax, edgecolor='red',facecolor='red')
            self.link_patches = [[] for _ in range(self.n_links)]
    
        R_q = self.rot 
        R, P = torch.eye(3), torch.zeros(3)       
        for j in range(self.n_links):
            if not isinstance(self.link_patches[j],list):
                self.link_patches[j].remove()
            P = R@self.P0[j] + P 
            R = R@self.R0[j]@R_q[j]
            self.link_patches[j] = (R@self.link_zonos[j]+P).to_zonotope().plot(ax=self.ax, edgecolor='blue',facecolor='blue')

        ax_scale = 1.2
        axis_lim = ax_scale*self.n_links
        plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    @property
    def rot(self):
        w = torch.tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = self.qpos.reshape(self.n_links,1,1)
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w


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

    def render(self):
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

if __name__ == '__main__':

    env = Arm_2D()
    for _ in range(50):
        env.step(torch.rand(2))
        env.render()
    '''
    env = Batch_Arm_2D()
    for _ in range(50):
        env.step(torch.rand(env.n_batches,2))
        env.render()
    '''    