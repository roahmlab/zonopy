import torch 
import zonopy as zp

class ARM_2D:
    def __init__(self,n_links=2,n_obs=1):
        self.n_links =2 
        self.n_obs = 1
        self.link_zonos = [zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]])).to_polyZonotope()]*n_links 

    def reset(self):
        self.qpos = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zero(self.n_links)
        self.qgoal = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.obs_zonos = []
        for _ in range(self.n_obs):
            obs_pos = torch.rand(2)*2*self.n_links-self.n_links
            obs = torch.hstack((torch.vstack((obs_pos,0.1*torch.eye(2))),torch.zeros(3,1)))
            obs = zp.zonotope(obs)
            if SAFE:
                self.obs_zonos.append(obs)

    #def step(ka):


    #def render(ax):


if __name__ == '__main__':
    import numpy as np 
    import matplotlib.pyplot as plt 
    import matplotlib.animation as animation 


    x = np.linspace(0, 10*np.pi, 100)
    y = np.sin(x)
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')
    
    for phase in np.linspace(0, 10*np.pi, 100):
        line1.set_ydata(np.sin(0.5 * x + phase))
        fig.canvas.draw()
        fig.canvas.flush_events()

    '''
    fig = plt.figure()
    ax = plt.axes(xlim=(-2,2),ylim=(-2,2))
    Z = zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]]))
    line, = Z.plot(ax)
    ax.add_patch(patches.Polygon(p,alpha=.5,edgecolor='green',facecolor='blue',linewidth=0.1))


    line, = ax.plot([],[],lw=2)
    def init():
        line.set_data([],[])
        return line,
    def animate(i):
        x = np.linspace(0,2,1000)
        y = np.sin(2*np.pi*(x-0.01*i))
        line.set_data(x,y)
        return line,
    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=200,interval=20,blit=True)
    plt.show()

    '''

