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
from tqdm import tqdm
from gym import error, spaces, utils
from ninjaParams import cmdp_source_tasks, target_task
from ninja import Ninja
from ninja_cmdp_world import CMDPEnv
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
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from model_transitions import create_model, transfer_model
from buffer_generation import Transition
from evaluate_metrics import get_hcmdp_state


parser = argparse.ArgumentParser()
parser.add_argument('-id', action='store', type=int, default=0, dest='id')
parser.add_argument('-dir', action='store', type=float, default=0.001, dest='dir')
args = parser.parse_args()


class hcmdp_transition():
	def __init__(self, state, action, next_state, is_done):
		self.state = state
		self.next_state = next_state
		self.action = action
		self.is_done = is_done
		self.calculate_reward()

	def calculate_reward(self):
		# option 1: difference in rewards
		self.reward = next_state[8]
		# print(self.reward)
		# option 2: 
		# self.reward = 10000 - next_state[8]
		# if next_state[8] >= 9000:
		# 	self.reward += 10000

		# option 3:
		# Mix of above two


def generate_h_cmdp_buffer(path='../results/ICRA/HCMDPBuffer/Buffer/', mode='full_state'):
	episodes = []
	for f in os.listdir(path):
		for ep in os.listdir(path+f):
			episodes.append(path+f+'/'+ep)

	print('Total episodes: ', len(episodes))
	
	

	if mode == 'full_state':

		observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1000, -1000.0]), 
											high=np.array([target_task.maxAction, target_task.maxAction, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 20000.0, 20000.0]), 
											dtype=np.float32)
	elif mode == 'mean_state':

		observation_space = spaces.Box(low=np.array([0.0, -10.0, 0.0, 0.0, -1000]), 
											high=np.array([target_task.maxAction, 100.0, 100.0, 100.0, 20000.0]), 
											dtype=np.float32)

	elif mode == 'no_reward':

		observation_space = spaces.Box(low=np.array([0.0, 0.0, -10.0, 0.0,0.0, 0.0, 0.0, 0.0]), 
											high=np.array([target_task.maxAction, target_task.maxAction, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]), 
											dtype=np.float32)

	elif mode == 'no_reward_mean':

		observation_space = spaces.Box(low=np.array([0.0, -10.0, 0.0, 0.0]), 
											high=np.array([target_task.maxAction, 100.0, 100.0, 100.0]), 
											dtype=np.float32)
	elif mode =='score':
		observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
											high=np.array([100.0, 100.0]), 
											dtype=np.float32)


	action_space = spaces.Discrete(len(cmdp_source_tasks))
	buffer_size = len(episodes)*100
	replay_buffer = ReplayBuffer(buffer_size=buffer_size, 
								 observation_space = observation_space,
								 action_space=action_space)

	# transitions = []
	for ep in episodes:
		print(ep)
		with open(ep, 'rb') as f:
			data = pickle.load(f)

		# print(len(data))
			
		for transition in data:
			# print(transition)
			if mode == 'full_state':
				obs = transition.state
				next_obs = transition.next_state
			elif mode == 'mean_state':
				obs = transition.state[[0,2,4,6,8]]
				next_obs = transition.next_state[[0,2,4,6,8]]
			elif mode == 'no_reward':
				obs = transition.state[:8]
				next_obs = transition.next_state[:8]
			elif mode == 'no_reward_mean':
				obs = transition.state[[0,2,4,6]]
				next_obs = transition.next_state[[0,2,4,6]]
			elif mode == 'score':
				obs = transition.state[[6,7]]
				next_obs = transition.next_state[[6,7]]

			reward = transition.next_state[8] - transition.state[8]
			replay_buffer.add(obs, next_obs, transition.action, reward, transition.is_done)


	print(replay_buffer.observations.shape)
	print(replay_buffer.actions.shape)
	print(replay_buffer.next_observations.shape)
	print(replay_buffer.rewards.shape)
	print(replay_buffer.dones.shape)	

	save_to_pkl("../results/ICRA/HCMDPBuffer/", replay_buffer)

	return True

	
if __name__ == '__main__':

	# path='../results/NeurIPS/Buffer/h_cmdp_buffer/'
	# generate_h_cmdp_buffer(mode='mean_state')
	# generate_h_cmdp_buffer(mode='no_reward')
	generate_h_cmdp_buffer()
	# generate_h_cmdp_buffer()
	# replay_buffer = load_from_pkl(path+'hcmdp_buffer_main')
	#env = CMDPEnv()
	#env.initialize()
	#model = DQN("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [32]}, device='cpu', learning_rate=0.01)
	#model.load_replay_buffer('../results/NeurIPS/h_cmdp/hcmdp_buffer_main')
	#model.train(batch_size=100, gradient_steps=10000)
	#model.save('../results/NeurIPS/h_cmdp/model')
	
	"""
	env = gym.make("CartPole-v0")
	model = DQN("MlpPolicy", env, verbose=1, learning_starts=0)
	# model.load_replay_buffer("sample_buffer")
	model.learn(total_timesteps=100000, log_interval=4)
	# model.train(batch_size=32, gradient_steps=12500)
	model.save_replay_buffer('sample_buffer')
	# for i in tqdm(range(12500)):
	# model.train(batch_size=100, gradient_steps=1000)
	

	# del model
	# model = DQN.load("dqn_cartpole")
	obs = env.reset()
	ret = 0
	while True:
	    action, _states = model.predict(obs, deterministic=True)
	    obs, reward, done, info = env.step(action)
	    ret+=reward
	    env.render()
	    if done:
	    	print('failed', ret)
	    	ret = 0
	    	obs = env.reset()
	"""

"""

if __name__ == '__main__':

	
	path = '../results/ICRA/HCMDPBuffer/Transitions/'+str(args.id)+'/'
	

	model = PPO('MlpPolicy',
				Ninja(target_task, actionNoise=3.0),
				learning_rate=0.001,
				n_steps=1000,
				policy_kwargs={"net_arch": [32]},
				gamma=0.99,
				verbose=0,
				device='cpu')

	base_path = '../results/ICRA/HCMDPBuffer/Buffer/'+str(args.id)+'/'
	os.makedirs(base_path, exist_ok=True)

	for episode in os.listdir(path):
		
		save_path = base_path+episode
		with open(path+episode, 'rb') as f:
			transitions = pickle.load(f)

		print('episode ', episode, ' contains ', len(transitions), ' transitions')
		h_cmdp_transitions = []
		last_params = model.policy.parameters_to_vector()
		last_state = np.empty(10)
		for i in tqdm(range(len(transitions))):
			transition = transitions[i]
			if (transition.model_pre == last_params).all():
				# print("Last state is current state")
				state = last_state
			else:
				model.policy.load_from_vector(transition.model_pre)
				# print("Last state is not current state")
				state = get_hcmdp_state(model)
			
			model.policy.load_from_vector(transition.model_post)
			next_state = get_hcmdp_state(model, log_path='../results/ICRA/HCMDPBuffer/Evaluation_logs/'+str(args.id)+'/')
			current_transition = hcmdp_transition(state, transition.action, next_state, transition.is_done) 
			h_cmdp_transitions.append(current_transition)

			last_state = next_state
			last_params = transition.model_post
			# break

		
		f = open(save_path, 'wb')
		pickle.dump(h_cmdp_transitions, f)
		f.close()
		# break



"""





