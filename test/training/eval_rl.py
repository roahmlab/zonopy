import numpy as np
import sys

from typing import Callable
from zonopy.environments.arm_2d import Arm_2D
from zonopy.environments.wrappers.dict_gym_wrapper import DictGymWrapper
from zonopy.environments.wrappers.gym_wrapper import GymWrapper

from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import gym
from gym import spaces
from gym.wrappers import TimeLimit



def make_env(n_links: int = 2, n_obs: int = 0, ep_len: int = 100, use_her: bool = True, rank: int = 0, seed: int = 0, render: bool = False) -> Callable:
    def _init() -> gym.Env:

        keys = ['qpos', 'qvel']
        goal_state_key = 'qgoal'
        acheived_state_key = 'qpos'

        if n_obs > 0:
            keys.extend(['obstacle_pos','obstacle_size'])
            interpolate = True
            check_collision = True
        else:
            interpolate = False
            check_collision = False

        env = Arm_2D(n_links=n_links,n_obs=n_obs, interpolate = interpolate, check_collision = check_collision)

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





if __name__ == '__main__':
    # The noise objects for TD3
    try:
        n_links = int(sys.argv[1])
        n_obs = int(sys.argv[2])
        num_workers = int(sys.argv[3])
        gpu_id = str(int(sys.argv[4]))
        gpu_id = 'cuda:' + gpu_id
        ep_len = int(sys.argv[5])
        timesteps = int(sys.argv[6])
        train_freq = int(sys.argv[7])
        use_her = bool(int(sys.argv[8]))
        action_noise = float(sys.argv[9])
        learning_rate = float(sys.argv[10])
    except:
        print("Please provide {num_links} {num_obs} {num_workers} {gpu_id} {ep_len} {train_timesteps} {train_freq} {use_her=0,1} {action_noise} and {learn_rate} for training!")
        exit()
    
    env = SubprocVecEnv([make_env(n_links,n_obs,ep_len,use_her,i) for i in range(num_workers)])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise * np.ones(n_actions))

    trial_name = "arm2d_reach_rl"
    trial_name += '-'.join((str(n) for n in sys.argv[1:]))

    # Setup training
    if use_her:
        model = TD3("MultiInputPolicy",
                    env,
                    learning_rate=learning_rate,
                    action_noise = action_noise,
                    verbose = 1,
                    replay_buffer_class=HerReplayBuffer,
                    learning_starts = 5000,
                    batch_size=256,
                    # Parameters for HER
                    replay_buffer_kwargs=dict(
                        # n_sampled_goal=4,
                        # goal_selection_strategy=goal_selection_strategy,
                        # online_sampling=False,
                        max_episode_length=ep_len,
                        # device = gpu_id,
                    ),
                    train_freq = (train_freq, "step"),     # To enable multiprocessing, we set it to train every 500 steps instead of every episode.
                    device = gpu_id,
                    tensorboard_log = f"./results/{trial_name}")
    else:
        model = TD3("MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    action_noise = action_noise,
                    verbose = 1,
                    train_freq = (num_workers, "step"),     # To enable multiprocessing, we set it to train every 500 steps instead of every episode.
                    device = gpu_id,
                    tensorboard_log = f"./results/{trial_name}")

    keys = ['qpos', 'qvel']
    goal_state_key = 'qgoal'
    acheived_state_key = 'qpos'

    if n_obs > 0:
        keys.extend(['obstacle_pos','obstacle_size'])
        interpolate = True
        check_collision = True
    else:
        interpolate = True
        check_collision = False

    env = Arm_2D(n_links=n_links,n_obs=n_obs, interpolate = interpolate, check_collision = check_collision)

    if use_her:
        env = DictGymWrapper(env, goal_state_key, acheived_state_key, keys=keys)
    else:
        keys.append(goal_state_key)
        env = GymWrapper(env, keys = keys)

    # Evaluation
    model.set_parameters("logs/best_model_obs1")
    for i in range(40):
        obs = env.reset()
        done = False 
        env.render()
        for j in range(70):
            action,_states =model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break 



