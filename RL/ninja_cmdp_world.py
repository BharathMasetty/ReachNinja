from __future__ import division
import gym
import os
import argparse
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ninjaParams import cmdp_source_tasks,target_task,final_cl
from evaluate_metrics import get_hcmdp_state
from ninja import Ninja
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from agent import student

class CMDPEnv(gym.Env):

	def initialize(self, target_task=target_task, sources=final_cl, is_hcmdp=True, log=False, log_path='../results/NeurIPS/h_cmdp/logs/', name='Episodes', mode='full_state'):
		"""
		target_task: parameters for creating the target_task
		agent_ctor: Constructor handle for the learning agent
		sources: List of parameters of the source task
		"""
		self.action_space = spaces.Discrete(len(final_cl))
		self.source_tasks = final_cl
		self.target_params = target_task
		self.max_num_steps = 60 # maximum length of the curriculum
		self.beta = 20000
		self.goal_cmdp_state = []
		self.state_action_log = np.empty((self.max_num_steps, 12))
		self.log_path = log_path
		self.log = log
		self.name = name
		self.episode_num = 0
		os.makedirs(log_path, exist_ok=True)
		self.mode=mode
		if mode == 'full_state':

			self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -10.0, 0.0,0.0, 0.0, 0.0, 0.0, -1000, -1000.0]), 
												high=np.array([target_task.maxAction, target_task.maxAction, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 20000.0, 20000.0]), 
												dtype=np.float32)
			
		elif mode == 'mean_state':

			self.observation_space = spaces.Box(low=np.array([0.0, -10, 0.0, 0.0, -1000]), 
												high=np.array([target_task.maxAction, 100.0, 100.0, 100.0, 20000.0]), 
												dtype=np.float32)


		elif mode == 'no_reward':

			self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -10.0, 0.0,0.0, 0.0, 0.0, 0.0]), 
												high=np.array([target_task.maxAction, target_task.maxAction, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]), 
												dtype=np.float32)

		elif mode == 'no_reward_mean':

			self.observation_space = spaces.Box(low=np.array([0.0, -10.0, 0.0, 0.0]), 
												high=np.array([target_task.maxAction, 100.0, 100.0, 100.0]), 
												dtype=np.float32)
		elif mode =='score':
			self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
												high=np.array([100.0, 100.0]), 
												dtype=np.float32)

		# self.state_action_log = np.empty((self.max_num_steps, self.observation_space.shape[0]+2))
		#[Mean_speed, percentage_max_score, score_change_per_hit, ratio_of_blue_to_black]

		
	def emit_observation(self):
		os.makedirs(self.log_path+'tmp/', exist_ok=True)
		full_state=self.student.get_weights(path=self.log_path+'tmp/')
		if self.mode=='full_state':
			new_state = full_state
		elif self.mode == 'mean_state':
			new_state = full_state[[0,2,4,6,8]]
		elif self.mode == 'no_reward':
			new_state = full_state[:8]
		elif self.mode == 'no_reward_mean':
			new_state = full_state[[0,2,4,6]]
		elif self.mode == 'score':
			new_state = full_state[[6,7]]
		return new_state, full_state

	def step(self, action):
		self.reward = 0
		
		# print(self.state_action_log[self.curr_step])
		
		cmdp_action = self.source_tasks[action]
		self.student.set_env(Ninja(cmdp_action, actionNoise=3))
		self.student.train(n_steps=self.beta)

		new_state, next_new_full_state = self.emit_observation()
		print(new_state)
		self.reward = next_new_full_state[8]

		self.state_action_log[self.curr_step, :] = np.append(self.full_state, [action, self.reward])
		self.state = new_state
		self.full_state = next_new_full_state
		# self.prev_performace = self.state[8]
		self.curr_step += 1

		if self.check_termination() or self.curr_step == self.max_num_steps:
			self.done = True 

		if self.done and self.log :
			# save episodic metrics somewhere
			np.savetxt(self.log_path+self.name+"_"+str(self.episode_num)+'.txt', self.state_action_log)
			# print("saving curriculum transitions")
			# save episodic model somewhere
			self.student.save(self.log_path+self.name+"_"+str(self.episode_num))
			print("saving student after curriculum")
			self.episode_num += 1
		print("---------------------------------------------------------------------")
		print(f"CMDP STEP: ep: {self.episode_num} , step: {self.curr_step}, state: {self.state}, action: {action}, score on target: {self.full_state[8]}, reward: {self.reward}")
		return self.state, self.reward, self.done, {}

	def reset(self):
		self.done =  False
		self.curr_step = 0
		self.student = student()
		self.state, self.full_state = self.emit_observation()
		# self.prev_performace = self.state[8]
		return self.state
	
	def check_termination(self):
			# Load Metrics for optimal model
			# compare it with current state
			# Just comparing rewards now.
		if self.full_state[8] >= 500:
			#self.convergance_status = True
			self.reward += 1000
			return True
		return False

	def set_continuous_action_space(self):
		pass



if __name__ == '__main__':
	env = CMDPEnv()
	env.initialize(target_task=target_task, sources=cmdp_source_tasks, log=False, mode='no_reward_mean')
	env.reset()
	state, full_state = env.emit_observation()
	print(state, full_state)
	"""
	done = False
	while not env.done:
		action = env.action_space.sample()
		env.step(action)
		break
		#print(state, reward, done)
	"""