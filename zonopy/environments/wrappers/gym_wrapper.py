'''
Reference: robosuite gymwrapper
'''
from zonopy.environments.wrappers.wrapper import Wrapper
from zonopy.environments.wrappers.utils import dict_torch2np
import numpy as np
import torch
from gym import spaces
from gym.core import Env


class GymWrapper(Wrapper, Env):
    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Get reward range
        self.reward_range = (0.0, 1.0)

        if keys is None:
            keys = []
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.get_observations()
        self.modality_dims = {key: tuple(obs[key].shape) for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict):
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                ob_lst.append(obs_dict[key].numpy().astype(float).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(torch.tensor(action,dtype=torch.get_default_dtype()))
        info['action_taken'] = action
        return self._flatten_obs(ob_dict), float(reward), done, dict_torch2np(info)

    def seed(self, seed=None):
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
                torch.maual_seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float(self.env.reward(action = torch.tensor(info['action_taken'],dtype=torch.get_default_dtype())))