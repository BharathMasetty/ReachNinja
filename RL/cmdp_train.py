import gym
import argparse
import os
import numpy as np
from ninja import Ninja
import pickle
from tqdm import tqdm
from ninja_cmdp_world import CMDPEnv
from gym import error, spaces, utils
from gym.utils import seeding
from ninjaParams import cmdp_source_tasks,target_task
from evaluate_metrics import get_hcmdp_state
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('-lr', action='store', type=float, default=0.001, dest='lr')
# parser.add_argument('-mode', action='store', type=str, default='full_state', dest='mode')
args = parser.parse_args()

def train_from_buffer(rbuffer, model, env):

	observations = rbuffer.observations
	actions = rbuffer.actions
	rewards = rbuffer.rewards
	next_observations = rbuffer.next_observations
	dones = rbuffer.dones
	n_samples  = rbuffer.buffer_size 
	print("Size of the replay_buffer: ", n_samples)
	print(observations.shape)
	initial_buffer_size = 100
	model.replay_buffer = rbuffer
	
	# replay_buffer = ReplayBuffer(n_samples, env.observation_space, env.action_space, device='cpu')
	## Adding initial Buffer
	for i in range(initial_buffer_size):
		# replay_buffer.add(observations[i], next_observations[i], actions[i], rewards[i], dones[i])
		model.num_timesteps += 1
	train_freq = 1
	batch_size = 100
	gradient_steps = 1
	
	## Start Training
	for i in tqdm(range(initial_buffer_size, n_samples)):
		# update buffer
		# replay_buffer.add(observations[i],next_observations[i], actions[i], rewards[i], dones[i])
		model.num_timesteps += 1
		model._on_step()

		if i%train_freq == 0:
			# train
			# model.replay_buffer = replay_buffer
			model.train(gradient_steps, batch_size)	

	return model


if __name__=='__main__':
	
	lr = args.lr
	print(lr)
	mode='no_reward'
	path = '../results/NeurIPS/h_cmdp/'
	log_dir = path+str(lr)+'/'
	os.makedirs(log_dir, exist_ok=True)
	
	# Load the replay Buffer
	with open('../results/NeurIPS/Buffer/replay_buffers/hcmdp_buffer_'+mode+'.pkl', 'rb') as f:
		rbuffer = pickle.load(f)
	

	print(rbuffer.observation_space)

	env = CMDPEnv()
	env.initialize(log=True, log_path=log_dir+'Logs/', name='Eval_'+mode+'_'+str(args.lr), mode=mode)
	print(env.observation_space)
	env = Monitor(env, log_dir)
	model = DQN("MlpPolicy", 
				env, 
				verbose=1, 
				device='cpu', 
				policy_kwargs={"net_arch": [8]},
				buffer_size=rbuffer.buffer_size, 
				learning_starts=100, 
				learning_rate=args.lr)
	
	# model = train_from_buffer(rbuffer, model, env)
	# model.save(log_dir+'trained_model_'+mode)
	# del model
	model = DQN.load(log_dir+'trained_model_'+mode)
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)

	
	