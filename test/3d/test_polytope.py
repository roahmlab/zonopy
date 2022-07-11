import torch 
import zonopy as zp


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3



scale = 5


z1 = zp.zonotope([[-1.99999,-1.99999,-1.99999],[1,0,0],[0,1,0],[0,0,1]])
z2 = zp.zonotope([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
z3 = zp.zonotope([[2,2,2],[1,0,0],[0,1,0],[0,0,1]])



fig = plt.figure()
ax = a3.Axes3D(fig)
          
z1_patch = Poly3DCollection(z1.polyhedron_patch(), edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.2)
ax.add_collection(z1_patch)
z2_patch = Poly3DCollection(z2.polyhedron_patch(), edgecolor='orange',facecolor='orange',alpha=0.2,linewidths=0.2)
ax.add_collection(z2_patch)
z3_patch = Poly3DCollection(z3.polyhedron_patch(), edgecolor='purple',facecolor='purple',alpha=0.2,linewidths=0.2)
ax.add_collection(z3_patch)

_,b12 = (z1-z2).polytope()
_,b23 = (z2-z3).polytope()

if b12.min() > 1e-6:
    print('z1 and z2 are in collision')
else:
    print('z1 and z2 are NOT in collision')
if b23.min() > 1e-6:
    print('z2 and z3 are in collision')
else:
    print('z2 and z3 are NOT in collision')


ax.set_xlim([-1*scale,1*scale])
ax.set_ylim([-1*scale,1*scale])
ax.set_zlim([-1*scale,1*scale])
plt.show()
