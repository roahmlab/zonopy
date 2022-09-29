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
    def __init__(self, env, keys=[]):
        # Run super method
        super().__init__(env=env)
        # Get reward range
        self.reward_range = (0.0, 1.0)

        self.keys = keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        self._setup_observation_space()
        low, high = self.action_spec
        self.action_space = spaces.Box(low=low, high=high)
 
    def _setup_observation_space(self):
        obs = self.env.get_observations()
        self.modality_dims = {key: tuple(obs[key].shape) for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

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

    def step(self, action, *args, **kwargs):
        ob_dict, reward, done, info = self.env.step(torch.as_tensor(action,dtype=self.env.dtype), *args, **kwargs)
        info['action_taken'] = action
        return self._flatten_obs(ob_dict), reward, done, dict_torch2np(info)

    def seed(self, seed=None):
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
                torch.maual_seed(seed)
                
            except:
                TypeError("Seed must be an integer type!")
        self.action_space.seed(seed)
        return Env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.get_reward(action = torch.as_tensor(info['action_taken'],dtype=self.env.dtype))
    
    def close(self):
        self.env.close()
    
    


class ParallelGymWrapper(GymWrapper):
    def __init__(self, env, keys=[]):
        self.num_envs = env.n_envs
        super().__init__(env=env,keys=keys)

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
    '''
    def step(self, action, flag=None):
        if isinstance(flag,np.ndarray):
            flag = torch.tensor(flag, dtype=int,device=env.device)
        import pdb;pdb.set_trace()
        ob_dicts, rewards, dones, infos = self.env.step(torch.tensor(action,dtype=env.dtype))
        ob_dicts, rewards, dones, infos = self.env.step(torch.tensor(action,dtype=env.dtype),flag)
    '''
    def step(self, action, *args, **kwargs):
        if len(args) > 0:
            flag = torch.tensor(args[0], dtype=int,device=self.env.device)
        else:
            flag = None
        ob_dicts, rewards, dones, infos = self.env.step(torch.as_tensor(action,dtype=self.env.dtype), flag)
        for b in range(self.n_envs):
            infos[b]['action_taken'] = action[b]
            for key in infos[b].keys():
                if isinstance(infos[b][key],torch.Tensor):
                    infos[b][key] = infos[b][key].numpy().astype(float)
                elif isinstance(infos[b][key],torch.Tensor) and isinstance(infos[b][key][0],torch.Tensor):
                    infos[b][key] = [el.numpy().astype(float) for el in infos[b][key]]
        return self._flatten_obs(ob_dicts), rewards.numpy(), dones.numpy(), infos

    def compute_reward(self, achieved_goal, desired_goal, infos):
        action_taken = []
        for info in infos:
            action_taken.append(info['action_taken'])
        action_taken = torch.as_tensor(np.vstack(action_taken),dtype=self.env.dtype)
        return self.env.get_reward(action = torch.as_tensor(info['action_taken'],dtype=self.env.dtype)).numpy()

    
    def get_attr(self,attr_name,indices=None):
        indices = self._get_indices(indices) 
        attr = getattr(self.envs,attr_name)
        if isinstance(attr,torch.Tensor) and attr.shape[0] == self.num_envs:
            return [attr[i] for i in indices]
        else:
            return [attr for _ in indices] 
    
    def set_attr(self,attr_name,value,indices=None):
        indices = self._get_indices(indices) 
        attr = getattr(self.envs,attr_name)
        if isinstance(attr,torch.Tensor) and attr.shape[0] == self.num_envs:
            for i in indices:
                attr[i] = value[i]
        else:
            attr = value 
        setattr(self.envs,attr_name,attr)

    def env_method(self,method_name,*method_args,incides=None,**method_kwargs):
        indices = self.get_indices(indices) 
        output = getattr(self.envs,method_name)(*method_args, **method_kwargs)

        if isinstance(output,torch.Tensor) and output.shape[0] == self.num_envs:
            return [output[i] for i in indices]
        else:
            return [output for _ in indices]

    def _get_indices(self,indices):
        if indices is None:
            indices - range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices] 
        return indices

if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    env = Arm_2D()
    env = GymWrapper(env,keys=['qpos','qvel','qgoal','obstacle_pos','obstacle_size'])
    import pdb;pdb.set_trace()