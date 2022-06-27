import torch 
from zonopy.environments.arm_2d import Arm_2D
from zonopy.layer.single_rts_star import gen_grad_RTS_2D_Layer
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
eps = 1e-6
n_links = 2
E = eps*torch.eye(n_links)
torch.set_default_dtype(torch.float64)
### 2. NLP gradient 
render = True
env = Arm_2D(n_links=n_links,n_obs=1)
observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([1,0.6])])

t_armtd = 0
params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
joint_axes = [j for j in env.joint_axes]
RTS = gen_grad_RTS_2D_Layer(env.link_zonos,joint_axes,env.n_links,env.n_obs,params)

observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys() ])
#k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
lam_hat = torch.tensor([[.3,.8]])
lam, FO_link, flag,Info = RTS(lam_hat,observ_temp.reshape(1,-1)) 

assert flag>=0, 'fail safe'

if render:
    observation, reward, done, info = env.step(lam[0]*torch.pi/24,flag[0])
    FO_link = [fo[0] for fo in FO_link]
    env.render(FO_link)

print(f'action: {lam_hat}')

diff1 = torch.zeros(n_links,n_links)
diff2 = torch.zeros(n_links,n_links)

for i in range(n_links):
    lam_hat1 = lam_hat + E[:,i]
    lam1, _, flag1,_ = RTS(lam_hat1,observ_temp.reshape(1,-1)) 

    lam_hat2 = lam_hat - E[:,i]
    lam2, _, flag2,_ = RTS(lam_hat2,observ_temp.reshape(1,-1)) 


    diff1[:,i] = (lam1-lam)/eps
    diff2[:,i] = (lam1-lam2)/(2*eps)
    #import pdb;pdb.set_trace()


print(f'diff 1: {diff1}')
print(f'diff 2: {diff2}')

import pdb;pdb.set_trace()



print(f'action: {lam}')
print(f'flag: {flag}')





