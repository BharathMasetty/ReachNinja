import os
import pathlib
import pickle
import tempfile
import stable_baselines3 as sb3
from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util
from ninjaParams import target_task, baseline_task
from ninja import Ninja
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

def make_ninja(task):
	def _init():
		return Ninja(task, action_type='cont', actionNoise=0)
	return _init


if __name__ == '__main__':

	# # Load pickled test demonstrations.
	with open("../results/Imitation_learning_low_speed/1", "rb") as f:
	    # This is a list of `imitation.data.types.Trajectory`, where
	    # every instance contains observations and actions for a single expert
	    # demonstration.
	    trajectories = pickle.load(f)

	transitions = rollout.flatten_trajectories(trajectories)
	print(len(transitions))

	env = DummyVecEnv([make_ninja(baseline_task) for i in range(1)])

	dest_folder = '../results/GAIL/Refresh/Tuning/'
	os.makedirs(dest_folder, exist_ok=True)
	lrs = [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0005, 0.0007, 0.0009, 0.001]

	tempdir = tempfile.TemporaryDirectory(prefix="Imitation_learning_for_ninja")
	tempdir_path = pathlib.Path(tempdir.name)
	logger.configure(tempdir_path / "GAIL/")

	for i in range(len(lrs)):
		
		print('Training on learning_rate: ', lrs[i])
		agent = sb3.PPO("MlpPolicy", 
						 env, 
						 verbose=0, 
						 n_steps=1000,
						 learning_rate=lrs[i],
						 policy_kwargs={"net_arch": [32]},
						 device = 'cpu')


		
		N_TOTAL_STEPS = int(len(transitions))
		N_BATCH = int(1e4)
		N_EPOCHS = int(N_TOTAL_STEPS/N_BATCH)
		gail_trainer = adversarial.GAIL(
		    env,
		    expert_data=transitions,
		    expert_batch_size=1000,
		    gen_algo=agent,
		)

		gail_trainer.train(total_timesteps=int(3e5))
		agent.save(dest_folder+'GAIL_'+str(i))

	# for i in range(N_EPOCHS):
	#     gail_trainer.train(total_timesteps=N_BATCH)
	#     agent.save(dest_folder+'GAIL_'+str(i))

	print('PPO trained using GAIL on Human Policy!')

