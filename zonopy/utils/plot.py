import matplotlib.pyplot as plt
from zonopy.utils.math import cos, sin

# TODO: CHECK
def plot_JRSs(Qs,deg= 6,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None, axis_lim=None):
    n_time_steps = len(Qs)
    n_joints = len(Qs[0])
    if axis_lim is None:
        L = 1.1
        axis_lim = [-L,L,-L,L]

    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_joints):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                q = Qs[t][i]
                c_q = cos(q,deg)
                s_q = sin(q,deg)
                c_s_q = c_q.exactCartProd(s_q)
                Z = c_s_q.to_zonotope()
                Z.plot(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis(axis_lim)
        plt.show()
    return ax

# TODO: CHECK
def plot_polyzonos_xy(PZx,PZy,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None, axis_lim=None):
    assert len(PZx) == len(PZy)
    n_time_steps = len(PZx)
    
    n_y = len(PZy[0])
    if isinstance(PZx,list):
        X = None
    elif len(PZx[0]) == 1:
        X = [0]*n_y
    elif len(PZx[0]) == n_y:
        X = list(range(n_y))
    else:
        assert False

    if axis_lim is None:
        L = 1.1
        #axis_lim = [-n_joints*L,n_joints*L,-n_joints*L,n_joints*L]

    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_y):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                if X is None:
                    PZ = PZx[t].exactCartProd(PZy[t][i])
                else:
                    PZ = PZx[t][X[i]].exactCartProd(PZy[t][i])
                Z = PZ.to_zonotope()
                Z.plot(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.axis(axis_lim)
        plt.show()
    return ax

# TODO: CHECK
def plot_polyzonos(PZs,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None, axis_lim=None):
    n_time_steps = len(PZs)
    n_joints = len(PZs[0])
    if axis_lim is None:
        L = 1.1
        axis_lim = [-n_joints*L,n_joints*L,-n_joints*L,n_joints*L]

    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_joints):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                Z = PZs[t][i].to_zonotope()
                Z.plot(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis(axis_lim)
        plt.show()
    return ax

# TODO: CHECK
def plot_zonos(Zs,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None,axis_lim=None):
    n_time_steps = len(Zs)
    n_joints = len(Zs[0])
    if axis_lim is None:
        L = 1.1
        axis_lim = [-n_joints*L,n_joints*L,-n_joints*L,n_joints*L]
    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_joints):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                Z = Zs[t][i]
                Z.plot(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis(axis_lim)
        plt.show()
    return ax