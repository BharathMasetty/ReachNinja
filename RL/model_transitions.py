from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import numpy as np

def create_model(name, env):

    if name == 'A2C':
        model = A2C('MlpPolicy',
                 env,
                 learning_rate=0.00005,
                 gamma=0.99,
                 n_steps=50,
                 verbose=1,
                 device='cuda')
    
    elif name == 'PPO':
        model =  PPO('MlpPolicy',
                 env,
                 learning_rate=0.00005,
                 n_steps=50,
                 gamma=0.99,
                 verbose=1,
                 device='cuda')
    
    elif name == 'DQN':
        model = DQN('MlpPolicy',
                 env,
                 learning_rate=0.000005,
                 gamma=0.99,
                 verbose=1,
                 batch_size=50,
                 learning_starts = 5,
                 train_freq = 1,
                 device='cuda')

    elif name == 'DDPG':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
        model =  DDPG('MlpPolicy',
                 env,
                 learning_rate=0.0001,
                 gamma=0.99,
                 verbose=1,
                 action_noise=action_noise,
                 device='cuda',
                 buffer_size=100000,
                 seed=0)

    elif name == 'SAC':
        model =  SAC('MlpPolicy',
                 env,
                 learning_rate=0.0001,
                 gamma=0.99,
                 verbose=1,
                 device='cuda')

    elif name == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
        model =  TD3('MlpPolicy',
                 env,
                 learning_rate=0.0001,
                 gamma=0.99,
                 verbose=1,
                 action_noise=action_noise,
                 device='cuda',
                 buffer_size=100000,
                 seed=0,
                 batch_size=100,
                 train_freq=500,
                 optimize_memory_usage=True)

    return model



def transfer_model(name, prev_policy_path, env, envName):
    
    if name == 'A2C': return A2C.load(prev_policy_path, env=env, device='cuda')
    elif name == 'PPO': return PPO.load(prev_policy_path, env=env, device='cuda')
    elif name == 'DQN': return DQN.load(prev_policy_path, env=env, device='cuda')
    elif name == 'SAC': return SAC.load(prev_policy_path, env=env, device='cuda')
    elif name == 'DDPG': return DDPG.load(prev_policy_path, env=env, device='cuda')
    elif name == 'TD3': return TD3.load(prev_policy_path, env=env, device='cuda')


