import torch
import zonopy as zp
import matplotlib.pyplot as plt

qpos =  torch.tensor([0])
qvel =  torch.tensor([1.2])
PZ_JRS = zp.load_JRS_trig(qpos,qvel)
zp.plot_polyzonos(PZ_JRS)