import torch
import zonopy as zp
import matplotlib.pyplot as plt

# Q,R = zp.gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=False)
# import pdb; pdb.set_trace()
import numpy as np
q = np.array([0, np.pi])
dq = np.array([np.pi, np.pi/2])
# BERNSTEIN
# param_range = np.ones(1) * np.pi/36
param_range = np.minimum(np.maximum(np.pi/24, abs(dq/3)), np.pi/3)
param_range = np.vstack([-param_range, param_range])
# PIECEWISE
tdiscretization = 0.01
i_s = np.arange(int(1/tdiscretization))
expMat = torch.eye(1, dtype=int)
gens = torch.ones(100) * tdiscretization/2
# Some tricks to make it into a batched poly zono
centers = torch.as_tensor(tdiscretization*i_s+tdiscretization/2, dtype=torch.get_default_dtype())
z = torch.vstack([centers, gens]).unsqueeze(2).transpose(0,1)
times = zp.batchPolyZonotope(z, 1, expMat, id=1).compress(2)
# Make sure to update the id map (fixes slicing bug)!
gen = zp.trajectories.PiecewiseArmTrajectory(q,
                                             dq,
                                             q,
                                             np.array(zp.polyZonotope([[0],[1]],1,id=0)),
                                             krange=param_range)
Q = gen.getReference(times)
from zonopy.conSet.polynomial_zonotope.utils import remove_dependence_and_compress
Q = remove_dependence_and_compress(Q[0][0], np.array(0))
import zonopy.transformations.rotation as rot
Q = rot.cos_sin_cartProd(Q, 1)


q = torch.tensor([0, torch.pi])
dq = torch.tensor([torch.pi, torch.pi/2])
PZ_JRS,_ = zp.load_JRS_trig(q,dq)
BPZ_JRS,_ = zp.load_batch_JRS_trig(q, dq)
ax = zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='blue',hold_on=True)
zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='orange',hold_on=True, ax=ax)
# zp.plot_polyzonos(PZ_JRS,plot_freq=1,ax=ax)

Q = [[Q[i]] for i in range(Q.batch_shape[0])]
# ax = zp.plot_JRSs(Q,plot_freq=1,edgecolor='blue',hold_on=True)
zp.plot_polyzonos(Q,plot_freq=1, ax=ax)
# zp.plot_JRSs(Q,deg=1,plot_freq=1,ax=ax)