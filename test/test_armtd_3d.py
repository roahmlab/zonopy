import torch

from zonopy.environments.arm_2d import Arm_2D
from zonopy.environments.arm_3d import Arm_3D

from zonopy.optimize.armtd_2d import ARMTD_2D_planner
from zonopy.optimize.armtd_3d import ARMTD_3D_planner

import pickle 



test_flag = False 
n_test = 50
n_timesteps = 100
N_OBS = torch.randint(10,25,(n_test,))
if test_flag:
    collision_info = []
    for i in range(n_test):
        env = Arm_3D(n_obs=int(N_OBS[i]))
        planner = ARMTD_3D_planner(env,device='cuda:0')
        for _ in range(n_timesteps):
            ka, flag = planner.plan(env,torch.zeros(env.n_links))
            observations, reward, done, info = env.step(ka.cpu(),flag)
            #env.render(planner.FO_link)
            if info['collision']:
                collision_info.append(info['collision_info'])

            #if done:
            #    break
            #import pdb;pdb.set_trace()
        print(f'{i+1}-th test, collision so far: {len(collision_info)}')
        with open('results/test_armtd_3d.pickle', 'wb') as handle:
            pickle.dump(collision_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
else:
    with open('results/test_armtd_3d.pickle', 'rb') as handle:
        collision_info = pickle.load(handle)
    if len(collision_info) == 0:
        print('all safe')
        exit()

    n_col_init = 0
    debug_test = []
    qpos_init_prev = torch.ones(7)*1e20
    for i in range(len(collision_info)):
        if (qpos_init_prev != collision_info[i]['qpos_init']).all():
            debug_test.append(i)
            n_col_init += 1
            qpos_init_prev = collision_info[i]['qpos_init']
            n_obs = len(collision_info[i]['obs_pos'])
            print(f'num of obs from collision test: {n_obs}')


    i = debug_test[0]
    env = Arm_3D(n_obs=len(collision_info[i]['obs_pos']))
    env.set_initial(qpos = collision_info[i]['qpos_init'],qvel= torch.zeros(env.n_links), qgoal = collision_info[i]['qgoal'],obs_pos=collision_info[i]['obs_pos'])
    #env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    
    planner = ARMTD_3D_planner(env,device='cpu')
    for t in range(n_timesteps):
        print(f'time step: {t}')
        ka, flag = planner.plan(env,torch.zeros(env.n_links),t>=4)
        observations, reward, done, info = env.step(ka.cpu(),flag)
        #env.render(planner.FO_link)
        if info['collision']:
            import pdb;pdb.set_trace()
