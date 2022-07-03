'''
Reference: robosuite gymwrapper
'''
#from zonopy.environments.wrappers.wrapper import Wrapper
from zonopy.environments.wrappers.gym_wrapper import GymWrapper
from zonopy.environments.wrappers.utils import dict_torch2np
import numpy as np
import torch
from gym import spaces
from gym.core import Env


class PrallelGymWrapper(GymWrapper):
    def __init__(self, env, keys=None):
        super().__init__(env=env)

    def _setup_observation_space(self):
        obs = self.env.get_observations()
        self.modality_dims = {key: tuple(obs[key].shape[1:]) for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.shape[1]
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict):
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                ob_lst.append(obs_dict[key].numpy().astype(float).reshape(self.n_envs,-1))
        return np.hstack(ob_lst)

    def step(self, action, flag=None):
        if isinstance(flag,np.ndarray):
            flag = torch.tensor(flag, dtype=int)
        ob_dicts, rewards, dones, infos = self.env.step(torch.tensor(action,dtype=env.dtype),flag)
        for b in range(self.n_envs):
            infos[b]['action_taken'] = action[b]
            for key in infos[b].keys():
                if isinstance(infos[b][key],torch.Tensor):
                    infos[b][key] = infos[b][key].numpy().astype(float)
                elif isinstance(infos[b][key],torch.Tensor) and isinstance(infos[b][key][0],torch.Tensor):
                    infos[b][key] = [el.numpy().astype(float) for el in infos[b][key]]
        return self._flatten_obs(ob_dicts), rewards.numpy(), dones.numpy(), infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.reward(action = torch.tensor(info['action_taken'],dtype=torch.get_default_dtype())).numpy()