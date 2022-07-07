import torch

from zonopy.environments.arm_2d import Arm_2D
from zonopy.environments.arm_3d import Arm_3D

from zonopy.optimize.armtd_2d import ARMTD_2D_planner
from zonopy.optimize.armtd_3d import ARMTD_3D_planner

import pickle 



test_flag = True 
n_test = 50
N_OBS = torch.randint(10,25,(n_test,))
if test_flag:
    collision_info = []
    for i in range(n_test):        
        env = Arm_3D(n_obs=int(N_OBS[i]))
        planner = ARMTD_3D_planner(env,device='cpu')
        for _ in range(100):
            ka, flag = planner.plan(env,torch.zeros(env.n_links))
            observations, reward, done, info = env.step(torch.tensor(ka,dtype=torch.get_default_dtype()),flag)
            #env.render(planner.FO_link)
            if info['collision']:
                collision_info.append(info['collision_info'])

            #if done:
            #    break
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
    qpos_init_prev = torch.tensor([10,10])
    for i in range(len(collision_info)):
        if (qpos_init_prev != collision_info[i]['qpos_init']).all():
            debug_test.append(i)
            n_col_init += 1
            qpos_init_prev = collision_info[i]['qpos_init']
            print(len(collision_info[i]['obs_pos']))


    i = debug_test[0]
    env = Arm_2D(n_obs=len(collision_info[i]['obs_pos']))
    env.set_initial(qpos = collision_info[i]['qpos_init'],qvel= torch.zeros(env.n_links), qgoal = collision_info[i]['qgoal'],obs_pos=collision_info[i]['obs_pos'])
    #env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    
    planner = ARMTD_2D_planner(env)
    for _ in range(200):
        ka, flag = planner.plan(env,torch.zeros(env.n_links))
        observations, reward, done, info = env.step(torch.tensor(ka,dtype=torch.get_default_dtype()),flag)
        env.render(planner.FO_link)
        if info['collision']:
            import pdb;pdb.set_trace()
