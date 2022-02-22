import torch
import zonopy
import matplotlib.pyplot as plt

qpos =  torch.tensor([0])
qvel =  torch.tensor([1.2])
JRS_poly = zonopy.load_JRS(qpos,qvel)

max_key = max(JRS_poly.keys())
n_joints = max_key[0]+1
n_time_steps = max_key[1]+1

fig = plt.figure()    
ax = fig.gca() 


for t in range(n_time_steps):
    Z = JRS_poly[(0,t)].to_zonotope()
    Z.plot2d(ax,facecolor='none')

plt.axis([0,1,0,1])
plt.show()

'''
JRS = zonopy.load_JRS(torch.tensor([1]),torch.tensor([0.2]))
fig = plt.figure()    
ax = fig.gca() 
for t in range(100):
    Z = JRS[(0,t)].to_zonotope()
    Z.plot2d(ax,facecolor='none')
    
plt.axis([0.35,0.55,0.84,0.93])
plt.show()
'''



