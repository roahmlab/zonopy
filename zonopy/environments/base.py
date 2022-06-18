from typing import Callable
from zonopy.environments.arm_2d import Arm_2D
from zonopy.environments.wrappers.dict_gym_wrapper import DictGymWrapper
from zonopy.environments.wrappers.gym_wrapper import GymWrapper
from gym.wrappers import TimeLimit
import gym
from stable_baselines3.common.utils import set_random_seed


def make_env(n_links: int = 2, n_obs: int = 0, ep_len: int = 100, use_her: bool = True, rank: int = 0, seed: int = 0, render: bool = False) -> Callable:
    def _init() -> gym.Env:

        keys = ['qpos', 'qvel']
        goal_state_key = 'qgoal'
        acheived_state_key = 'qpos'

        if n_obs > 0:
            keys.extend(['obstacle_pos','obstacle_size'])
        env = Arm_2D(n_links=n_links,n_obs=n_obs)

        if use_her:
            env = DictGymWrapper(env, goal_state_key, acheived_state_key, keys=keys)
        else:
            keys.append(goal_state_key)
            env = GymWrapper(env, keys = keys)
        env = TimeLimit(env, max_episode_steps=ep_len)
        env.seed(seed+rank)
        return env
    set_random_seed(seed)
    return _init