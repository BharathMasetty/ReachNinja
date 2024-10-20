from __future__ import division
import gym
import argparse
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ninjaParams import cmdp_source_tasks,target_task, baseline_task
from evaluate_metrics import get_hcmdp_state
from ninja import Ninja
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy

"""
This class defines a student agent in h-cmdp/cmdp setup

TODO: Crete seperate CMDP and H-CMDP student classes from this class
"""
class student():
	def __init__(self, path="../results/GAIL/Refresh/Tuning/GAIL_5.zip"):
		
		params = PPO.load(path).policy.parameters_to_vector()
		env = Ninja(target_task, actionNoise=3)
		self.model = PPO('MlpPolicy',
						env,
						learning_rate= 0.001,
						n_steps=1000,
						policy_kwargs={"net_arch": [32]},
						gamma=0.99,
						verbose=1,
						device='cpu')

		self.model.policy.load_from_vector(params)
		# self.model = PPO.load(path)
		self.model.verbose=0
		print("Pre-Trained Model Loaded!")

	def train(self, n_steps):
		# print(f"Student Learning for {n_steps} steps")
		self.model.env.reset()
		self.model.learn(total_timesteps=int(n_steps))

	def set_env(self, env):
		self.model.set_env(env)

	def get_weights(self, save=False, path=None, name=None):
		state = get_hcmdp_state(self.model, n_episodes=5, save=save, metrics_path=path, name=name)
		# print([M[0][0], M[1][0], M[2][0], M[3][0], M[4]])
		## only mean of metrics and mean reward
		# print(state)
		return np.array(state)
	def save(self, path):
		self.model.save(path)


if __name__ == '__main__':

	test_agent = student()
	env = Ninja(target_task)
	env.reset()
	test_agent.set_env(env)
	# test_agent.train(10000)
	test_agent.get_weights()
