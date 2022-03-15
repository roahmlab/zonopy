import torch
import zonopy
import matplotlib.pyplot as plt
from zonopy.forward_kinematics.FK import forward_kinematics
qpos =  torch.tensor([0,0])
qvel =  torch.tensor([1.2,-1.2])
joint_axes = [torch.tensor([0,0,1])]*2
P = [torch.tensor([0,0,0],dtype=torch.float32), torch.tensor([1,0,0],dtype=torch.float32)]
_,P_motor = forward_kinematics(qpos,qvel,joint_axes,P)

max_key = max(P_motor.keys())
n_joints = max_key[0]+1
n_time_steps = max_key[1]+1

fig = plt.figure()    
ax = fig.gca() 

for i in range(n_joints):
    for t in range(n_time_steps):
        Z = P_motor[(i,t)].to_zonotope()
        if t % 1 == 0:
            Z.plot2d(ax,facecolor='none')
            #import pdb; pdb.set_trace()
            #

plt.title('FRS of joint')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2,2,-2,2])
plt.show()