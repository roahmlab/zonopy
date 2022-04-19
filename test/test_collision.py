import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import matplotlib.pyplot as plt


qpos =  torch.Tensor([0,0])
qvel =  torch.Tensor([torch.pi,torch.pi/2])
params = {'joint_axes':[torch.Tensor([0,0,1])]*2, 
        'R': [torch.eye(3)]*2,
        'P': [torch.Tensor([0,0,0]), torch.Tensor([1,0,0])],
        'n_joints':2}
link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0],[0,0,0.01],[0,0,0]])).to_polyZonotope()]*2
_, R =zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=1,make_gens_independent =True)
n_time_steps = len(R)
n_joints = params['n_joints']

FO_link,FO_link_slc = [], [[] for _ in range(n_time_steps)]
k_id = zp.conSet.PROPERTY_ID['k'] # list
for t in range(n_time_steps):
    FO_link_temp,_,_ = forward_occupancy(R[t],link_zonos,params)
    FO_link.append(FO_link_temp)
    for i in range(n_joints):
        FO_link_slc[t].append(FO_link_temp[i].slice_dep(k_id,[1,1]))


# obstacle
obs = zp.zonotope([[-1.5, 0.2, 0],[0, 0, 0.2]])





#ax = zp.plot_polyzonos(FO_link_trig,plot_freq=1,edgecolor='blue',hold_on=True)

#

fig = plt.figure()
ax = fig.gca()
obs.plot(ax,facecolor='blue',edgecolor='blue', linewidth=1)
zp.plot_polyzonos(FO_link,plot_freq=1,ax=ax,hold_on=True)
for t in range(n_time_steps):
    for i in range(n_joints):
        if t%1 == 0:
            Z = FO_link_slc[t][i].to_zonotope()
            buff = Z.project()-obs
            A,b,_ = buff.polytope()
            if max(A@torch.zeros(2)-b)<1e-6:
                color = 'red'
                f_color = 'red'
            else:
                f_color = 'none'
                color = 'red'
            Z.plot(ax,facecolor=f_color, edgecolor=color, linewidth=.2)


plt.title('Collision detection')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2.2, 2.2, -2.2, 2.2])
plt.show()