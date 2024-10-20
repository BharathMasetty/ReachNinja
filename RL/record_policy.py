import gym
from ninjaParams import SourceTasks, target_task, baseline_task, test_task, marker_source_tasks
from ninja import Ninja
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from buffer_generation import Transition
import argparse
import pickle
import os

# log_dir = "../logs/baseline/DQN/best_model.zip"
# parser = argparse.ArgumentParser()
# parser.add_argument('-agent', action='store', type=int, default=0, dest='agent')
# args = parser.parse_args()

# log_dir = "../results/GAIL/low_speed/GAIL_PreTrain_"+str(args.agent)+".zip"
# log_dir = '../results/GAIL/Refresh/Tuning/GAIL_5.zip'
log_dir = '../results/GAIL/Refresh/Tuning/GAIL_5.zip'
# log_dir = '../results/ICRA/GAILCL/staticCLWithGAIL/0/final_model.zip'
save_path = '../results/post_training_logs/BC/'
# log_dir = "../results/"
baseline_task.visualization = True
# test_task.actionStep=200

# check_env(env)

# model = PPO('MlpPolicy',
# 			env,
# 			learning_rate= 0.003,
# 			n_steps=1000,
# 			policy_kwargs={"net_arch": [32]},
# 			gamma=0.99,
# 			verbose=1,
# 			device='cpu')

# path = '../results/NeurIPS/Buffer/initial_policies/ppo_initial_policies.pkl'
# with open(path, 'rb') as f:
# 	data = pickle.load(f)

# print(type(data))
# params = data[0]
# model.policy.load_from_vector(params)
# parameters = model.get_parameters()
# for key, value in parameters.items():
# 	print(key, value)

# base_dir = '../results/ICRA/noiseCL/StaticCurricula/Baseline_2/'
for i in range(1,6):
	env = Ninja(baseline_task, log=False, save_path = save_path, action_type='cont', actionNoise=0) 
	# log_dir = base_dir+'/'+'final_model.zip'
	model = PPO.load(log_dir)
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
	print(mean_reward, std_reward)

