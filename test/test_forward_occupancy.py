

import torch
import zonopy
import matplotlib.pyplot as plt
from zonopy.forward_kinematics.FO import forward_occupancy,forward_occupancy_quat, forward_occupancy_fancy

qpos =  torch.tensor([torch.pi/2,0])
qvel =  torch.tensor([1.2,1.2])
joint_axes = [torch.tensor([0,0,1])]*2
P = [torch.tensor([0,0,0],dtype=torch.float32), torch.tensor([1,0,0],dtype=torch.float32)]
link_zonos = [zonopy.zonotope(torch.tensor([[0.5,0.5,0],[0,0,0.01],[0,0,0]])).to_polyZonotope()]*2


FO_link,_,P_motor = forward_occupancy_fancy(qpos,qvel,joint_axes,P,link_zonos)

ax = zonopy.plot_dict_polyzono(FO_link,title='test',hold_on=True)
#ax = zonopy.utils.plot_dict_pz(P_motor,title='test',hold_on=True,ax=ax)

FO_link,_,P_motor = forward_occupancy(qpos,qvel,joint_axes,P,link_zonos)

ax = zonopy.plot_dict_polyzono(FO_link,title='test',hold_on=True,ax=ax,edgecolor='blue')

max_key = max(FO_link.keys())
n_joints = max_key[0]+1
n_time_steps = max_key[1]+1

ka = torch.tensor([0,0])
t_total = 1
t_plan = 0.5
for t in range(n_time_steps):
    qpi = torch.clone(qpos)
    qvi = torch.clone(qvel)
    t_curr = t/n_time_steps*t_total
    if t_curr < t_plan:
        qp = qpi + qvi * t_curr + 0.5*ka * t_curr**2
    else:
        qp_peak = qpi + qvi * t_plan + 0.5*ka*t_plan**2
        qv_peak = qvi + ka*t_plan
        qp = qp_peak + qv_peak*(t_curr-t_plan) + 0.5*(0-qv_peak)/(t_total-t_plan)*(t_curr-t_plan)**2
    for i in range(n_joints):        
        p_motor = torch.zeros(3,dtype=torch.float32)
        r_motor = torch.eye(3,dtype=torch.float32)
        for j in range(i+1):
            c_qp = torch.cos(qp[j])
            s_qp = torch.sin(qp[j])
            Rot = torch.tensor([[c_qp,-s_qp,0],[s_qp,c_qp,0],[0,0,1]])
            p_motor = r_motor@P[j]+p_motor
            r_motor = r_motor@Rot
        Z = r_motor@link_zonos[i]+p_motor
        if t%10 == 0:
            Z.to_zonotope().plot2d(ax,edgecolor='red',facecolor='none')


plt.title('FRS of link zonotope')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2,2,-2,2])
plt.show()