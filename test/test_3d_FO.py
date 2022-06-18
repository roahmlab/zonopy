import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3


def rot(q,joint_axes):
    #joint_axes
    w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
    w = (w@joint_axes.T).transpose(0,-1)
    q = q.reshape(q.shape+(1,1))
    return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

scale = 100

params, _ = zp.load_sinlge_robot_arm_params('Kinova3')
n_joints = params['n_joints']
link_zonos = params['link_zonos'] # NOTE: zonotope, should it be poly zonotope?
link_zonos = [(scale*link_zonos[j]).to_polyZonotope() for j in range(n_joints)]
P0 = [scale*P for P in params['P']]
params['P'] = P0
R0 = params['R']
joint_axes = torch.vstack(params['joint_axes'])

qpos = torch.ones(n_joints)*torch.pi/2*0+(torch.rand(n_joints)*2*torch.pi - torch.pi)*0
qvel = torch.rand(n_joints)*2*torch.pi - torch.pi


fig = plt.figure()
ax = a3.Axes3D(fig)


R_q = rot(qpos,joint_axes)


RR = []
PP= []
R, P = torch.eye(3), torch.zeros(3)
link_patches = []
for j in range(n_joints):
    P = R@P0[j] + P
    R = R@R0[j]@R_q[j]
    PP.append(P)
    RR.append(R)
    link_patch = (R@link_zonos[j]+P).to_zonotope().polyhedron_patch()
    link_patches.extend(link_patch)            
link_patches = Poly3DCollection(link_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5)
ax.add_collection(link_patches)
ax.set_xlim([-.8*scale,.8*scale])
ax.set_ylim([-.8*scale,.8*scale])
ax.set_zlim([-.8*scale,.8*scale])


lamb = torch.zeros(n_joints)

_, R_trig = zp.load_batch_JRS_trig(qpos,qvel)
FO_link,_, _ = forward_occupancy(R_trig,link_zonos,params)



FO_patches = []
for j in range(n_joints): 
    for t in range(100): 
        FO_link_temp = FO_link[j].to_batchZonotope().reduce(4)        
        if t%20 == 0:
            FO_patch = FO_link_temp[t].polyhedron_patch()
            FO_patches.extend(FO_patch)
FO_patches = Poly3DCollection(FO_patches,alpha=0.03,edgecolor='green',facecolor='green',linewidths=0.2)
ax.add_collection3d(FO_patches)  

FO_patches = []
for j in range(n_joints): 
    for t in range(100): 
        FO_link_slc = FO_link[j].slice_all_dep(lamb.unsqueeze(0).repeat(100,1)).reduce(4)        
        if t%20 == 0:
            FO_patch = FO_link_slc[t].polyhedron_patch()
            FO_patches.extend(FO_patch)
FO_patches = Poly3DCollection(FO_patches,alpha=0.03,edgecolor='red',facecolor='red',linewidths=0.2)
ax.add_collection3d(FO_patches)  
plt.show()


