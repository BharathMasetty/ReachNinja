import gym
import os
import math
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from ninjaParams import SourceTasks, target_task, baseline_task, unconstrained_task
from ninja import Ninja
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from scipy import interpolate
import argparse
import copy

"""
Returns the hcmdp state as a vector  
"""


def get_hcmdp_state(model, n_episodes=5, save=False, metrics_path='../results/PreTrainedModel/', name='metrics',
					log_path = '../results/NeurIPS/Evaluation_logs/test/', actionNoise=3):

	#env 
	os.makedirs(log_path, exist_ok=True)
	for f in os.listdir(log_path):
		os.remove(log_path+f)

	env = Ninja(target_task, log='True', action_type='cont', save_path=log_path, actionNoise=actionNoise)
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)

	mean_speeds = []
	score_change_per_hits = [] 
	blue_to_black_ratios = [] 
	percentage_max_scores = []

	for ep in os.listdir(log_path):
		if not ep.endswith('.npz'): continue
		accuracy, confidance, precision, strategy = get_metric_values_per_episode(log_path+ep)
		percentage_max_scores.append(accuracy)
		blue_to_black_ratios.append(precision)
		mean_speeds.append(confidance)
		score_change_per_hits.append(strategy)

	state = np.empty(10)

	state[0:2] = [np.mean(mean_speeds),np.std(mean_speeds)]
	state[2:4] = [np.mean(score_change_per_hits),np.std(score_change_per_hits)]
	state[4:6] = [np.mean(blue_to_black_ratios),np.std(blue_to_black_ratios)]
	state[6:8] = [np.mean(percentage_max_scores),np.std(percentage_max_scores)]
	state[8:] = [mean_reward, std_reward]
	# print(state)
	if (save):
		# print(state)
		np.savez_compressed(
		   metrics_path+name, 
		   state=state)

	return state


def get_metric_values_per_episode(path):
	

	data = np.load(path)
	player_velocities = data['velocities']
	rewards = data['rewards']
	total_blue = data['num_blue_markers']
	total_black = data['num_black_markers']
	max_possible = data['max_score']
	min_possible = data['min_score']
	
	numBlue = 0
	numBlack = 0
	nonZeroRewards = []
	for r in rewards:
		if r > 0:
			numBlue+=1
			nonZeroRewards.append(r)
		elif r < 0:
			numBlack+=1
			nonZeroRewards.append(r)
			

	precision = numBlue*100/total_blue

	if len(nonZeroRewards) == 0:
		strategy = 0
	else:    
		strategy = np.mean(nonZeroRewards)*100/30
		

	accuracy = ((sum(rewards)-min_possible)/(max_possible-min_possible))*100
	confidance = np.mean(np.linalg.norm(player_velocities, axis=1))

	return accuracy, confidance, precision, strategy



class evaluate_human_metrics():
	
	def __init__(self, path='../results/LongPilot/D/Day1/Log_26_14-09-2020_22-34-34.log'):
		self.Data = open(path)
		self.lines = self.Data.readlines()

		self.center = np.array([640.0, 480.0])*0.5
		self.posNorm = np.array([640.0, 480.0])
		self.v_max = 3000.0
		self.vstep = 1000.0
		self.velNorm = np.array([3000.0, 3000.0])
		self.acceleration = np.array([0., -100.0])
		
		labels = []
		for word in self.lines[1].split(','):
			labels.append(word)
		print(labels)
		self.t_idx =  labels.index('Time')
		self.px = labels.index('Player:X')
		self.py = labels.index('Player:Y')
		self.vx = labels.index('PlayerVel:X')
		self.vy = labels.index('PlayerVel:Y')
		self.ret = labels.index('Player:Score')
		self.nm = labels.index('Marker #s')
		self.scores_idx = labels.index('Player:Score')

	def read_positions_per_file(self, lines):
		
		# data = open(file_name)
		# lines = data.readlines()
		positions = np.zeros((len(lines[4:]),25))
		scores = []
		# print(lines[2])
		# start from 2 for long pilot data
		for i in range(len(lines[4:])):
			line = lines[i+4]
			position = np.zeros(25)
			elements = []
			for word in line.split(','):
				elements.append(word)
				# print(elements[-1])
			# Getting time stamp
			position[-1] = float(elements[self.t_idx])

			# player position
			position[0:2] = np.array([float(elements[self.px]), float(elements[self.py])])

			# marker position
			num_markers = int(elements[self.nm])
			blue_idx = 0
			black_idx = 0
			for m in range(num_markers-1):
				# 16 for m_turk
				m_id = int(elements[9*m+16])
				# 17 for m_turk
				m_type = elements[9*m+17]
				#  19 20 for m_turk
				x = float(elements[9*m+19])
				y = float(elements[9*m+20])
				
				if m_type == 'Regular':
					# 19 and 20 for m_turk
					position[2+2*blue_idx:4+2*blue_idx] = np.array([x,y])
					blue_idx += 1
				elif m_type == 'Exploding':
					position[12+2*black_idx:14+2*black_idx] = np.array([x,y])
					black_idx += 1
				
				position[22] = blue_idx
				position[23] = black_idx

			positions[i] = position
			scores.append(float(elements[self.scores_idx]))

		# filter positions
		invalid_indices = []
		for i in range(1, positions.shape[0]-1):
			if max(abs(positions[i+1, :2] - positions[i, :2])) > 30.0:
				invalid_indices.append(i)
				invalid_indices.append(i+1)

		positions = np.delete(positions, invalid_indices, 0)
		scores = np.delete(np.array(scores), invalid_indices)
			
		# Interpolation
		y = np.round(positions[:, :22], 2)
		timesteps = np.round(positions[:,-1], 2)
		f = interpolate.PchipInterpolator(positions[:, -1], y, extrapolate=True)
		start = min(timesteps)
		end = max(timesteps)
		N = int((end-start)/0.01)
		t = np.linspace(min(timesteps), max(timesteps), N)
		new_positions = np.empty((len(t), 25))
		new_positions[:, :22] = f(t)
		new_positions[:, -1] = t
		new_scores = np.zeros(len(t))
		j = 0
		for i in range(len(t)):
			curr_time = t[i]
			ref_time = timesteps[j]
			if curr_time > ref_time:
				j+=1
			new_scores[i] = scores[j]
			new_positions[i, 22] = positions[j, 22]
			new_positions[i, 23] = positions[j, 23]

		return new_positions, new_scores

	def get_metrics_from_human_log(self, file_name):

		Data = open(file_name)
		lines = Data.readlines()

		positions, scores = self.read_positions_per_file(lines)
		velocities = np.empty((positions.shape[0]-1,22))
		for i in range(positions.shape[0]-1):
			velocities[i] = np.clip((positions[i+1, :22] - positions[i, :22])/(positions[i+1, -1] - positions[i, -1]), -3000, 3000)

		mean_speed = np.mean(np.linalg.norm(velocities[:, :2], axis=1))

		labels = []
		for word in lines[1].split(','):
			labels.append(word)
		numMarkers = labels.index('Marker #s')

		numBlue = 0
		numBlack = 0
		numBlueHit = 0
		numBlackHit=0
		last_score = 0
		max_possible_score = 0
		min_possible_score = 0
		last_m_id = 0
		max_radius = 10
		min_radius = 5
		max_vel = 1.
		velocity_scale = 400

		for s in scores:
			if s > last_score:
				numBlueHit +=1
			if s < last_score:
				numBlackHit += 1
			last_score = s


		for line in lines[2:]:

			elements = []
			for word in line.split(','):
				elements.append(word)
				
			num_markers = int(elements[numMarkers])
			for m in range(num_markers-1):
					m_id = int(elements[9*m+15])
					if m_id > last_m_id:
						last_m_id = m_id
						m_type = elements[9*m+16]
						if m_type == 'Exploding':
							numBlack += 1
							min_possible_score += -10
						elif m_type == 'Regular':
							numBlue += 1
							## update max_possible_score
							marker_radius = float(elements[9*m+17])
							marker_vel = np.linalg.norm([float(elements[9*m+20]), float(elements[9*m+21])])
							possible_score = 10 + 10*(max_radius-marker_radius)/(max_radius-min_radius) + 10*marker_vel/(velocity_scale*max_vel)
							max_possible_score += possible_score

		score_change_per_hit = last_score/(numBlackHit+numBlueHit)
		if numBlackHit == 0:
			numBlackHit = 1
		blue_to_black_ratio = numBlueHit/numBlackHit
		percentage_max_score = (last_score-min_possible_score)/(max_possible_score-min_possible_score) * 100
	#     print(mean_speed, score_change_per_hit, blue_to_black_ratio, percentage_max_score)
		return mean_speed, score_change_per_hit, blue_to_black_ratio, percentage_max_score


	def evaluate_human_state(self, source='../results/post_training_logs/Humans/', dest='../results/post_training_logs/Humans/', name='metrics'):
		
		mean_speeds = []
		score_change_per_hits = [] 
		blue_to_black_ratios = [] 
		percentage_max_scores = []

		for f in os.listdir(source):
			print(source+f)
			ms, scph, bbr, pms = self.get_metrics_from_human_log(source+f)
			mean_speeds.append(ms)
			score_change_per_hits.append(scph)
			blue_to_black_ratios.append(bbr)
			percentage_max_scores.append(pms)

		M1 = [np.mean(mean_speeds), np.std(mean_speeds)]
		M2 = [np.mean(score_change_per_hits), np.std(score_change_per_hits)]
		M3 = [np.mean(blue_to_black_ratios), np.std(blue_to_black_ratios)]
		M4 = [np.mean(percentage_max_scores), np.std(percentage_max_scores)]

		np.savez_compressed(
			   dest+name, 
			   mean_speeds=M1,
			   score_change_per_hits=M2,
			   blue_to_black_ratios=M3,
			   percentage_max_scores=M4)

		print(M1, M2, M3, M4)

		return M1, M2, M3, M4


def key(name):
	ret = 0
	for i in name:
		ret+= ord(i)
	return ret

if __name__ == '__main__':

	dest='../results/ICRA/GAILMetrics/Tuning/'
	os.makedirs(dest, exist_ok=True)
	source = '../results/GAIL/Refresh/Tuning/'
	# f = 'GAIL_9.zip'
	# model = PPO.load(source+f)
	# pps = get_cmdp_state(model, save=True,  metrics_path=dest, name=f, n_episodes=10)
	for f in os.listdir(source):
		print(f)
		model = PPO.load(source+f)
		pps = get_hcmdp_state(model, save=True,  metrics_path=dest, name=f, n_episodes=10)

	# i=0
	# models = sorted(os.listdir(source), key=key)
	# models = ['GAIL_PPO_NoMagnetic4.zip', 'GAIL_PPO_NoMagnetic22.zip','GAIL_PPO_NoMagnetic25.zip']
	# models = models.sort()
	# l = [5]
	# for i in l:
	#     model_name = 'GAIL_PPO_NoMagnetic'+str(i)+'.zip'
	#     print(f'Evaluating : {source+model_name}')
	#     model = PPO.load(source+model_name)
	#     pps = get_cmdp_state(model, save=True,  metrics_path=dest, name=str(i))
	
	# dest = '../results/post_training_logs/Humans/'
	# source = '../results/LongPilot/'
	# names=['PreTraining', 'PostTraining'] 
	# for n in names:
	#     metrics = evaluate_human_metrics()
	#     metrics.evaluate_human_state(source=source+n+'/', dest=dest,name=n)
