import torch
import zonopy as zp
import matplotlib.pyplot as plt

q = torch.tensor([0])
dq = torch.tensor([torch.pi])
Q,R = zp.gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=False)
import pdb; pdb.set_trace()


PZ_JRS,_ = zp.load_JRS_trig(q,dq)
ax = zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='blue',hold_on=True)
#zp.plot_polyzonos(PZ_JRS,plot_freq=1,ax=ax)

zp.plot_JRSs(Q,deg=1,plot_freq=1,ax=ax)