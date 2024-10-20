import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import pickle
from ninjaParams import cmdp_source_tasks, target_task, test_task
from ninja import Ninja
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# parser = argparse.ArgumentParser()
# parser.add_argument('-agent', action='store', type=int, default=0, dest='agent')
# parser.add_argument('-lr', action='store', type=float, default=0.001, dest='lr')
# args = parser.parse_args()


"""
Call back to save the best model during training
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):

	def __init__(self, check_freq: int, log_dir: str, verbose=1):
			super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
			self.check_freq = check_freq
			self.log_dir = log_dir
			self.save_path = os.path.join(log_dir, 'best_model')
			self.best_mean_reward = -np.inf

	def _init_callback(self) -> None:
			# Create folder if needed
			if self.save_path is not None:
					os.makedirs(self.save_path, exist_ok=True)

	def _on_step(self) -> bool:
			if self.n_calls % self.check_freq == 0:

				# Retrieve training reward
				x, y = ts2xy(load_results(self.log_dir), 'timesteps')
				if len(x) > 0:
						# Mean training reward over the last 100 episodes
						mean_reward = np.mean(y[-100:])
						if self.verbose > 0:
							print("Num timesteps: {}".format(self.num_timesteps))
							print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

						# New best model, you could save the agent here
						if mean_reward > self.best_mean_reward:
								self.best_mean_reward = mean_reward
								# Example for saving best model
								if self.verbose > 0:
									print("Saving new best model to {}".format(self.save_path))
								self.model.save(self.save_path)

			return True


"""
Call back to regularly save an intermidiate model while training.
"""
class SaveIntermediateModelCallBack(BaseCallback):

	def __init__(self, save_freq:int, log_dir: str, name:str, verbose=1):
			super(SaveIntermediateModelCallBack, self).__init__(verbose)
			self.save_freq=save_freq
			self.log_dir=log_dir
			self.save_path = os.path.join(log_dir, name)
			self.model_index = 0
			self.intermediate_modes = {}
			# self.intermediate_modes[self.model_index] = self.model.policy.parameters_to_vector()

	def _init_callback(self)->None:
			if self.save_path is not None:
				os.makedirs(self.save_path, exist_ok=True)

	def _on_step(self)->None:
			if self.n_calls % self.save_freq == 0:
				# add the new intermediate model to the dict and save the file
				self.intermediate_modes[self.model_index] = self.model.policy.parameters_to_vector()
				self.model_index += 1
				self._save_models()

	def _save_models(self):
			f = open(self.save_path+'.pkl', 'wb')
			pickle.dump(self.intermediate_modes, f)
			f.close()



"""
Function to train on the target task
Best paramters:
lr = 0.001
n_steps = 1000
net_arch = 32
total_timesteps = 1.25 * 10^6
"""


def run_baseline(name='1', lr=0.001, n_steps=1000):

	base_dir = '../results/ICRA/Baseline/PPO/'
	# save_dir = '../results/NeurIPS/Intermidiate/'
	

	for task in range(len(cmdp_source_tasks)): 
		print("Training on source task: ", task)
		source_task = cmdp_source_tasks[task]
		env = Ninja(source_task)
		log_dir = base_dir+str(task)+'/'
		os.makedirs(log_dir, exist_ok=True)
		env = Monitor(env, log_dir)
		# callback = SaveOnBestTrainingRewardCallback(check_freq=5e4, log_dir=log_dir)
		savecallback = SaveIntermediateModelCallBack(save_freq=2e4, log_dir=log_dir, name=str(task))
		model = PPO('MlpPolicy',
					 env,
					 learning_rate= lr,
					 n_steps=n_steps,
					 policy_kwargs={"net_arch": [32]},
					 gamma=0.99,
					 verbose=1,
					 device='cpu')
		model.learn(total_timesteps=int(1.25e6),
									callback=savecallback)
		model.save(log_dir)

	return True

"""
Function to training on target task with gail
lr = 0.003
n_steps = 1000
net_arch = 32
total_timesteps = 1.25 * 10^6
"""
# @ray.remote
def run_Gail(name='1', lr=0.003, n_steps=1000):

	base_dir = '../results/NeurIPS/SourceTaskTraining/GAIL+PPO/'
	# save_dir = '../results/NeurIPS/Intermidiate/'
	os.makedirs(base_dir, exist_ok=True)

	for task in range(len(cmdp_source_tasks)):
		
		print("Training on source task: ", task)
		source_task = cmdp_source_tasks[task]
		env = Ninja(source_task)
		log_dir = base_dir+str(task)+'/'
		os.makedirs(log_dir, exist_ok=True)
		env = Monitor(env, log_dir)

		path = "../results/NeurIPS/GAIL/GAIL_8.zip"
		params = PPO.load(path).policy.parameters_to_vector()
		savecallback = SaveIntermediateModelCallBack(save_freq=2e4, log_dir=log_dir, name='gail')
		model = PPO('MlpPolicy',
								env,
								learning_rate=lr,
								n_steps=n_steps,
								gamma=0.99,
								policy_kwargs={"net_arch": [32]},
								verbose=1,
								device='cpu')

		model.policy.load_from_vector(params)
		# callback = SaveOnBestTrainingRewardCallback(check_freq=5e4, log_dir=log_dir)
		model.learn(total_timesteps=int(1.25e6), callback=savecallback)
		model.save(log_dir)

	return True


def eval_target_performance(base_path):

	env = Ninja(target_task)
	agent = PPO('MlpPolicy', env, policy_kwargs={"net_arch": [32]})

	for i in range(7):
		print("Eavluating Performance of task: ", str(i))
		path = base_path+str(i)+'/'
		with open(path+'gail.pkl', 'rb') as f:
			models = pickle.load(f)

		rewards = np.empty((len(models), 2))
		print("No. of models : ", len(models))
		for j in tqdm(range(len(models))):
			agent.policy.load_from_vector(models[j])
			mean, std = evaluate_policy(agent, env, n_eval_episodes=3, deterministic=True)
			rewards[j, :] = np.array([mean, std])

		np.savez_compressed(path+str(i),
			rewards=rewards)



if __name__ == '__main__':
	

	# run_baseline(name=str(args.agent), lr=args.lr)
	# ray.init()
	# names  = []
	# for n in range(10,20):
	# 	names.append(str(n))
	# # names = ['15', '16', '17', '18', '19']
	# # lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
	# n_steps = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000]  
	# lr = 0.002
	# n_step = 1000
	
	# result_ids = []
	# for name, step in zip(names, n_steps):
	# 	result_ids.append(run_Gail.remote(name=name, lr=lr, n_steps=n_step))

	# results = ray.get(result_ids)

	# print("Done Gail Baselines")
	# # print("Starting Gail")
	# # run_Gail()

	# Saveing Intermediate Models
	# ray.init()
	# result_ids = [run_baseline.remote(), run_Gail.remote()]
	# results = ray.get(result_ids)
	# print("Saving intermediate models Done!")

	# run_baseline()
	# run_Gail()
	base_path = '../results/NeurIPS/SourceTaskTraining/GAIL+PPO/'
	eval_target_performance(base_path)


