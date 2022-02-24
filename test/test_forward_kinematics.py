import torch
import zonopy
import matplotlib.pyplot as plt
from zonopy.forward_kinematics.FK import forward_kinematics
qpos =  torch.tensor([0,0])
qvel =  torch.tensor([1.2,1.2])
joint_axes = [torch.tensor([0,0,1])]*2
P = [torch.tensor([1,0,0],dtype=torch.float32)]*2
_,EE = forward_kinematics(qpos,qvel,joint_axes,P)

max_key = max(EE.keys())
n_joints = max_key[0]+1
n_time_steps = max_key[1]+1

fig = plt.figure()    
ax = fig.gca() 

for i in range(n_joints):
    for t in range(n_time_steps):
        Z = EE[(i,t)].to_zonotope()
        Z.plot2d(ax,facecolor='none')

plt.title('FRS of end-effector')
plt.xlabel('x')
plt.ylabel('y')
plt.axis([-2,2,-2,2])
plt.show()