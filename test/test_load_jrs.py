import torch
import zonopy
import matplotlib.pyplot as plt

qpos =  torch.tensor([0])
qvel =  torch.tensor([0.05])
JRS_poly = zonopy.load_JRS(qpos,qvel)

max_key = max(JRS_poly.keys())
n_time_steps = max_key[0]+1
n_joints = max_key[1]+1

fig = plt.figure()    
ax = fig.gca() 


for t in range(n_time_steps):
    Z = JRS_poly[(0,t)].to_zonotope()
    Z.plot2d(ax)

