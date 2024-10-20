import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import ray
import pickle
import datetime
from gym import error, spaces, utils
from ninjaParams import cmdp_source_tasks, target_task, final_cl
from ninja import Ninja
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from model_transitions import create_model, transfer_model


"""
This class defines a single transition for a cmdp/hcmdp traning step.
Use this class to save transitions and build approriate replay buffer later on.

TODO: Define transition with reward later on.
"""


parser = argparse.ArgumentParser()
parser.add_argument('-id', action='store', type=int, default=0, dest='id')
# parser.add_argument('-lr', action='store', type=float, default=0.001, dest='lr')
args = parser.parse_args()



class Transition():
	def __init__(self, model_pre, action, model_post, is_done=False):
		self.model_pre = model_pre
		self.model_post = model_post
		self.action = action
		self.is_done = is_done

def make_ninja(task):
	def _init():
		return Ninja(task, action_type='cont')
	return _init

"""
This class is a call back for saving the transition every timestep and 
evaluating the performace on the target task to reset the training
"""

class CurriculumBufferGenerator():

	def __init__(self, eval_freq:int, log_dir:str, lr:float, n_steps:int, name:str, initial_model):
		self.eval_freq = eval_freq
		self.log_dir = log_dir
		self.transitions = {}
		self.source_tasks  = final_cl
		print('Number of source tasks :', len(self.source_tasks)) 
		self.action_space = spaces.Discrete(len(final_cl))
		self.initial_model = initial_model
		self.lr = lr
		self.name = name
		self.n_steps = n_steps
		self._init_callback()

	def reset_model(self):
		self.model = PPO('MlpPolicy',
						Ninja(target_task, actionNoise=3),
						learning_rate=self.lr,
						n_steps=self.n_steps,
						policy_kwargs={"net_arch": [32]},
						gamma=0.99,
						verbose=0,
						device='cpu')
		print("Model reset call")
		self.model.policy.load_from_vector(self.initial_model)

	def _init_callback(self) -> None:
		os.makedirs(self.log_dir, exist_ok=True)
		self.reset_model()
		self.episode_idx = 0
		self.timestep = 0
		self.max_episodes = 5 # 35-default
		self.max_num_timesteps = 12 # 65-default
		self.beta = 100000
		self.max_target_reward = 1000
		self.done = False
		self.init_envs()

	def init_envs(self):
		self.envs = []
		for param in self.source_tasks:
 			self.envs.append(Ninja(param, actionNoise=3))	

	def eval_model(self) -> float:
		mean_reward, std_reward = evaluate_policy(self.model, Ninja(target_task, actionNoise=3), n_eval_episodes=5, deterministic=True)
		return mean_reward

	def reset(self):
		self.save_transitions()
		self.reset_model()
		self.timestep = 0
		self.episode_idx += 1
		self.done = False
		self.transitions = {}

	def save_transitions(self):
		save_path = self.log_dir+name+'_episode_'+str(self.episode_idx)
		f = open(save_path+'.pkl', 'wb')
		pickle.dump(self.transitions, f)
		f.close()

	def step(self) -> None:
		action = self.action_space.sample()
		# env = SubprocVecEnv([make_ninja(baseline_task) for i in range(4)])
		current_state = self.model.policy.parameters_to_vector()
		self.model.set_env(self.envs[action])
		self.model.learn(total_timesteps=self.beta)
		print('Model name: ', self.name,' episode number: ', self.episode_idx,' step number: ', self.timestep, ' source_task: ', action)
		next_state = self.model.policy.parameters_to_vector()
		self.done = (self.timestep+1 == self.max_num_timesteps)
		self.transitions[self.timestep] = Transition(current_state, action, next_state, self.done)
		self.timestep += 1
		
		if self.timestep % self.eval_freq == 0:
			mean_reward = self.eval_model()
			print('current mean reward on target task: ', mean_reward)
			if mean_reward >= 0.95*self.max_target_reward: 
				print("Model reached threshold performace, resetting to inital parameters")
				self.reset_model()

		if self.done:
			self.reset()

	def run(self, start_episode_idx=0):
		self.episode_idx = start_episode_idx
		if start_episode_idx >= self.max_episodes:
			print("Max training episodes reached for this model, try something else!")
			return
		while self.episode_idx <= self.max_episodes:
			self.step()

		return True


if __name__ == '__main__':

	eval_freq = 10 # 10
	lr = 0.001
	n_steps = 1000
	source = '../results/ICRA/HCMDPBuffer/intermediate_models.pkl'
	base_path = "../results/ICRA/HCMDPBuffer/Trasitions/"
	with open(source, 'rb') as f:
	    models = pickle.load(f)
	total_models = len(models)
	print("Total Initial Models : ", len(models))
	model_ids = int(args.id)
	model = models[model_ids]
	name = str(model_ids)

	log_dir = base_path + name + "/"
	buffer_gen = CurriculumBufferGenerator(eval_freq=eval_freq,
	 								       log_dir=log_dir,
	 									   lr = lr,
	 									   n_steps=n_steps,
	 									   name=name,
 										   initial_model=model)
	
	buffer_gen.run()
