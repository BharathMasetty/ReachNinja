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
from ninjaParams import cmdp_source_tasks, target_task, unconstrained_task
from ninja import Ninja
from evaluate_metrics import get_hcmdp_state
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



def evaluate_models(path, name='intermediate_models.pkl', noise=0):

	model_path = path+name
	print("Loading models from: ", model_path)
	with open(model_path, 'rb') as f:
		models = pickle.load(f)
	states = np.empty((len(models), 10))
	env =  Ninja(unconstrained_task, actionNoise=noise)
	model = PPO('MlpPolicy',
					 env,
					 learning_rate=0.003,
					 n_steps=1000,
					 policy_kwargs={"net_arch": [32]},
					 gamma=0.99,
					 verbose=0,
					 device='cpu')

	for i in range(len(models)):
		# print("Evaluating model number: ", i)		
		model.policy.load_from_vector(models[i])
		states[i] = get_hcmdp_state(model, n_episodes=5, log_path =path+'eval/', actionNoise=noise)
		print("Performance on Target Task: ", states[i,8:], "For model : ", i)

	np.savez_compressed(
		path+'training_target_eval',
		states = states)



if __name__ == "__main__":


	path = '../results/ICRA/staticCL/CL_1/'
	evaluate_models(path)

	"""

	models_path = '../results/ICRA/actionStep/Target_task_100/Target_task.pkl'
	# ppo_source = '../results/NeurIPS/Buffer/initial_policies/ppo_initial_policies.pkl'


	with open(models_path, 'rb') as f:
	    ppo_models = pickle.load(f)


	# with open(ppo_source, 'rb') as f:
	#     ppo_models = pickle.load(f)
	

	# gail_states = np.empty((len(gail_models), 10))
	ppo_states = np.empty((len(ppo_models), 10))

	env =  Ninja(target_task)
	model = PPO('MlpPolicy',
					 env,
					 learning_rate=0.003,
					 n_steps=1000,
					 policy_kwargs={"net_arch": [32]},
					 gamma=0.99,
					 verbose=0,
					 device='cpu')

	# for i in range(len(gail_models)):
	# print("Evaluating gail model number: ", i)		
	# model.policy.load_from_vector(gail_models[i])
	# gail_states[i] = get_hcmdp_state(model, n_episodes=5)

	for i in range(len(ppo_models)):
		print("Evaluating ppo model number: ", i)		
		model.policy.load_from_vector(ppo_models[i])
		ppo_states[i] = get_hcmdp_state(model, n_episodes=5, log_path ='../results/ICRA/actionStep/Target_task_100/eval/' )


	np.savez_compressed(
		"../results/ICRA/actionStep/Target_task_100/training_performace_on_target",
		ppo_states = ppo_states)

 	
	# data = np.load("../results/NeurIPS/Buffer/initial_policies/Initial_policies_performace.npz")
	
	# gail_evals = data['gail_evals']
	# gail_stds = data['gail_stds']
	# ppo_evals = data['ppo_evals']
	# ppo_stds = data['ppo_stds']

	# # plot and save
	# plt.figure(figsize=[20, 15])
	# plt.plot(range(len(gail_evals)), gail_evals, label = 'GAIL+PPO')
	# plt.fill_between(range(len(gail_evals)), gail_evals-gail_stds, gail_evals+gail_stds, alpha=0.2)
	# plt.plot(range(len(ppo_evals)), ppo_evals, label = 'PPO')
	# plt.fill_between(range(len(ppo_evals)), ppo_evals-ppo_stds, ppo_evals+ppo_stds, alpha=0.2)
	# save_dir = '../results/NeurIPS/Plots/Intermediate/'
	# os.makedirs(save_dir, exist_ok=True)
	# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
	# plt.xlabel('Model Index')
	# plt.ylabel('Average Return')
	# plt.title("Performance of Models used in Bufffer Generation")
	# plt.savefig(save_dir+"Intermediate_models_on_target.jpg")
	# plt.show()
	"""