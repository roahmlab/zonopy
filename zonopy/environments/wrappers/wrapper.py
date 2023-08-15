# TODO VALIDATE

'''
Reference: robosuite gymwrapper
'''
import numpy as np
import torch
from zonopy.environments.wrappers.utils import dict_torch2np
class Wrapper:
    """
    Base class for all wrappers in robosuite.
    Args:
        env (MujocoEnv): The environment to wrap.
    """

    def __init__(self, env):
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                env = env.env
            else:
                break

    def step(self, action, *args, **kwargs):
        """
        By default, run the normal environment step() function
        Args:
            action (np.array): action to take in environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(torch.tensor(action,dtype=torch.get_default_dtype()), *args, **kwargs)
        return dict_torch2np(ob_dict), float(reward), done, dict_torch2np(info)

    def reset(self):
        """
        By default, run the normal environment reset() function
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return dict_torch2np(ob_dict)

    def render(self, *args, **kwargs):
        """
        By default, run the normal environment render() function
        Args:
            **kwargs (dict): Any args to pass to environment render function
        """
        return self.env.render(*args,**kwargs)

    def observation_spec(self):
        """
        By default, grabs the normal environment observation_spec

        """
        return dict_torch2np(self.env.get_observations())

    @property
    def action_spec(self):
        """
        By default, grabs the normal environment action_spec
        Returns:
            2-tuple:
                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        return (-np.pi/24*np.ones(self.env.dof),np.pi/24*np.ones(self.env.dof))

    @property
    def action_dim(self):
        """
        By default, grabs the normal environment action_dim
        Returns:
            int: Action space dimension
        """
        return self.env.dof

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment
        Returns:
            env (MujocoEnv): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr
            

