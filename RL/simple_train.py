import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import pickle
from ninjaParams import cmdp_source_tasks, target_task, test_task, marker_source_tasks, final_cl, unconstrained_task, unconstrained_basetask
from ninjaParams import unconstrained_a, unconstrained_b, actionStepParams, ReachNinjaParams
from ninja import Ninja
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from baseline import SaveOnBestTrainingRewardCallback, SaveIntermediateModelCallBack
from evaluate_intermediate_models import evaluate_models

parser = argparse.ArgumentParser()
parser.add_argument('-task', action='store', type=int, default=0, dest='task')
parser.add_argument('-noise', action='store', type=int, default=0, dest='noise')
args = parser.parse_args()

if __name__ == "__main__":
	
	# for step in [100,200,300,400,500]:
	log_dir = '../results/ICRA/GAILCL/'+str(args.task)+'/'
	# log_dir = '../results/ICRA/finalCL/Baseline/'
	os.makedirs(log_dir, exist_ok=True)
	
	# args.actionNoise = 4
	# # Base Task
	# source_task = unconstrained_b
	# # print(source_task.actionStep)
	# # source_task.actionStep = args.task
	# # print(source_task.actionNoise)
	# # source_task.visualization = True
	
	# env = Ninja(target_task, actionNoise=3)
	# print(source_task.magneticCoeff, env.actionNoise)
	# env = Monitor(env, log_dir)
	# model =  PPO('MlpPolicy',
	# 		 env,
	# 		 learning_rate= 0.001,
	# 		 n_steps=1000,
	# 		 policy_kwargs={"net_arch": [32]},
	# 		 gamma=0.99,
	# 		 verbose=1,
	# 		 device='cpu')
	# # callback = SaveOnBestTrainingRewardCallback(check_freq=5e4, log_dir=log_dir)
	# callback = SaveIntermediateModelCallBack(save_freq=1e4, log_dir=log_dir, name='intermediate_models')
	# model.load('../results/GAIL/Refresh/Tuning/GAIL_5.zip')
	# model.learn(total_timesteps=int(1.2e6), callback=callback)
	# model.save(log_dir+'final_model')
	# evaluate_models(log_dir, noise=3)
	
	
	
	# # CL - 1
	# static_cl_1 = []
	# static_cl_1.append(cmdp_source_tasks[6])
	# static_cl_1.append(cmdp_source_tasks[0])
	# static_cl_1.append(cmdp_source_tasks[1])
	# static_cl_1.append(cmdp_source_tasks[2])
	# static_cl_1.append(cmdp_source_tasks[3])
	# static_cl_1.append(cmdp_source_tasks[4])
	# static_cl_1.append(cmdp_source_tasks[5])
	# static_cl_1.append(cmdp_source_tasks[-1])
	# # static_curriculum = [0, 1, 2, 3, 4, 5, 6, 7]
	# # t = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


	# # CL - 2
	# mixed_tasks = []
	# mixed_tasks.append(cmdp_source_tasks[6])
	# mixed_tasks.append(marker_source_tasks[0])
	# mixed_tasks.append(marker_source_tasks[1])
	# mixed_tasks.append(cmdp_source_tasks[0])
	# mixed_tasks.append(cmdp_source_tasks[1])
	# mixed_tasks.append(cmdp_source_tasks[-1])
	# static_curriculum = [0, 1, 2, 3, 4, 5]
	# t = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	
	
	# for task_id in range(1):

	static_curriculum = [0, 1, 2, 3, 4]
	t = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4]
	# noise = [0, , 2, 3, 3]
	
	env = Ninja(target_task, actionNoise=3)
	model = PPO('MlpPolicy',
			 env,
			 learning_rate= 0.001,
			 n_steps=1000,
			 policy_kwargs={"net_arch": [32]},
			 gamma=0.99,
			 verbose=1,
			 device='cpu')
	gail_model_policy = PPO.load('../results/GAIL/Refresh/Tuning/GAIL_5.zip').policy.parameters_to_vector()
	model.policy.load_from_vector(gail_model_policy)
	# model = PPO.load('../results/GAIL/Refresh/Tuning/GAIL_5.zip')
	# model.learning_rate = 0.001

	log_dir = '../results/ICRA/GAILCL/staticCLWithGAIL3/'+str(args.task)+'/'
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(log_dir+'CLStages/', exist_ok=True)
	np.savetxt(log_dir+'curriculum.out', static_curriculum)
	callback = SaveIntermediateModelCallBack(save_freq=1e4, log_dir=log_dir, name='intermediate_models')

	# model  = PPO.load(log_dir+'final_model')
	for i in range(len(final_cl)):
		print("Training on Source task: ", final_cl[i])
		params = final_cl[i]
		# params.visualization = True
		# params = static_cl_1[static_curriculum[i]]
		env = Ninja(params, actionNoise=3)
		# print(env.observation_space)
		print(params.magneticCoeff, env.actionNoise)
		# model = PPO.load(log_dir+'CLStages/6')
		model.set_env(env)
		model.learn(total_timesteps=int(t[i]*1e5), callback=callback)
		model.save(log_dir+'CLStages/'+str(i))
		# break
		
	model.save(log_dir+'final_model')	
	evaluate_models(log_dir, noise=3)
		
	