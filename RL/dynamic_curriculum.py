import gym
import argparse
import numpy as np
from ninja import Ninja
import os
from baseline import SaveIntermediateModelCallBack
from ninja_cmdp_world import CMDPEnv
from gym import error, spaces, utils
from gym.utils import seeding
from ninjaParams import cmdp_source_tasks,target_task, final_cl
from evaluate_metrics import get_hcmdp_state
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# def h_cmdp():
env = CMDPEnv()
log_dir='../results/ICRA/HCMDPEval2/'
os.makedirs(log_dir, exist_ok=True)
env.initialize(target_task=target_task, sources=final_cl, log=True, log_path=log_dir+'/logs/')
env = Monitor(env, log_dir)

# Training teacher agent
model = DQN("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [16]}, device='cpu', learning_rate=0.001, train_freq=1,
			learning_starts=10000, batch_size=50, gradient_steps=5)
model.load_replay_buffer('../results/ICRA/HCMDPBuffer/HCMDPBuffer.pkl')
callback = SaveIntermediateModelCallBack(save_freq=10, log_dir=log_dir, name='intermediate_models')
model.train(batch_size=50, gradient_steps=12000)
model.learn(total_timesteps=10000, log_interval=4, callback=callback)
model.save(log_dir+'TeacherModel')
model.save_replay_buffer(log_dir)
print("Training on Buffer done, starting evaluation now")


# env = CMDPEnv(log=True, log_path=log_dir+'/logs/')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=False)


"""
teacher_agent = DQN(policy='MlpPolicy',
					env=env, 
					learning_rate=0.01,
					verbose=0,
					learning_starts=5,
					batch_size=5,
					train_freq=2,
					target_update_interval=5,
					buffer_size=100000,
					device='cuda')
"""
# n_steps = 100
# teacher_agent.load_replay_buffer(log_dir+'sample_teacher')
# teacher_agent.learn(total_timesteps=int(n_steps), log_interval=1)
# teacher_agent.save(log_dir+'sample_teacher')
# teacher_agent.save_replay_buffer(log_dir+'sample_teacher')
