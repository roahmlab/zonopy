import torch
from zonopy.environments.arm_2d import Arm_2D
from zonopy.optimize.armtd import ARMTD_planner
import time
n_links = 2

#torch.manual_seed(3)


env = Arm_2D(n_links=n_links,n_obs=1)
env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])

planner = ARMTD_planner(env)
for _ in range(100):
    ka, flag = planner.plan(env,torch.zeros(n_links))
    #observations, reward, done, info = env.step(torch.tensor(ka,dtype=torch.get_default_dtype()),flag)
    observations, reward, done, info = env.step(torch.rand(n_links))
    env.render(planner.FO_link)

    if info['collision']:
        import pdb;pdb.set_trace()

    if done:
        import pdb;pdb.set_trace()
        break
import pdb;pdb.set_trace()