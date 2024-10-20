import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from hcmdp_buffer import hcmdp_transition
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import rc
import random
import pickle


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    means = np.convolve(values, weights, 'valid')
    stds = np.std(rolling_window(values, window), 1)
    return means, stds


def plot_action_results(log_folders, title='Training Returns With Varying Bounds on Action Space', colors=[], labels=[]):
    rc('text', usetex=True)
    font = {'family':'serif', 'serif': ['computer modern roman']}
    plt.rc('font',**font)

    fig = plt.figure(figsize=[10,5]) 
    ax = fig.add_subplot(1, 1, 1)
    returns = np.empty((int(1.25e6), 5))
    for log_folder, label in zip(log_folders, labels):
        for i in range(5):
            x, y = ts2xy(load_results(log_folder+'/'+str(i)+'/'))
            results[:, i] = y
        means = np.mean(results, axis=0)
        stds = np.std(results, axis=0)
        ax.plot(x, means, label=label)
        ax.fill_between(x, means - stds, means+ stds, alpha=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel('Training steps')
        plt.ylabel('Average return on target task')
        plt.title(title)
        plt.tight_layout()
        plt.grid(True)

    ax.legend(loc='upper left', frameon=True)
    ax.set(facecolor=(0.0, 0.0, 1.0, 0.05))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.xaxis.labelpad = 30

    plt.savefig('../results/ICRA/Plots/actionSizesNew.jpg')
    plt.show()

def plot_results(log_folders, title='', colors=[], labels=[]):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    rc('text', usetex=True)
    font = {'family':'serif', 'serif': ['computer modern roman']}
    plt.rc('font',**font)

    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=[25,14]) 
    ax = fig.add_subplot(1, 1, 1)
    for log_folder, label in zip(log_folders,  labels):
        print(log_folder, label)
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        # print(y.size)
        y, stds = moving_average(y, window=10)
        # Truncate x
        x = x[len(x) - len(y):]
        print(x[0])
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - stds, y + stds, alpha=0.2)
        ax.legend(title='\textbf{Action Space Bound}',loc='upper left', frameon=True)
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel('Training steps')
        plt.ylabel('Average return on target task')
        # plt.title(title)
        # plt.tight_layout()
        plt.grid(True)
        savedir = "../results/Thesis/ActionStePlot/"
        os.makedirs(savedir, exist_ok=True)
        np.savez_compressed(savedir+label[:3],
            x=x,
            y=y,
            stds=stds)

        # ax.set(facecolor=(0.0, 0.0, 1.0, 0.05))
        # plt.savefig(savedir+'ActionStepSize.jpg')
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    
    ax.set(facecolor=(0.0, 0.0, 1.0, 0.05))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.xaxis.labelpad = 30
    # plt.savefig('../results/ICRA/Plots/actionSizes.jpg')
    plt.show()

    """
    #include h_cmpd parts here
    path = '../results/NeurIPS/h_cmdp/logs/Episodes_0.txt'
    h_cmdp_log = np.loadtxt(path)
    rewards = np.array(h_cmdp_log[:, 8])
    std = np.array(h_cmdp_log[:, 9])
    timesteps = np.array(range(len(rewards)))*20000
    # ax.errorbar(timesteps, rewards, std, fmt = '^', label='H_CMDP')
    ax.plot(timesteps, rewards, label='H_CMDP')
    ax.fill_between(timesteps, rewards - std, rewards + std, alpha=0.2)

    path = '../results/NeurIPS/Buffer/h_cmdp_buffer/0/0_episode_0.pkl'
    with open(path,'rb') as f:
        data = pickle.load(f)

    rewards = []
    std = []

    for t in data:
        rewards.append(t.state[8])
        std.append(t.state[9])

    rewards = np.array(rewards)
    std = np.array(std)
    timesteps = np.array(range(len(rewards)))*20000

    # ax.errorbar(timesteps, rewards, std, fmt = 'o', label='Random Curriculum')

    # for t, r, err in zip(timesteps, rewards, stds):
    #     ax.errorbar([t], [r], [err])

    ax.plot(timesteps, rewards, label='Random Curriculum')
    ax.fill_between(timesteps, rewards - std, rewards + std, alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.savefig('../results/NeurIPS/Plots/H_CMDP/Initial_Results.jpg')
    plt.show()
    """

#log_dirs = ["../logs/baseline/a2c/", "../logs/baseline/PPO/", "../logs/baseline/DQN/"]
#colors=['b','g','r']
#labels=['A2C', 'PPO', 'DQN']

if __name__ == '__main__':
    colors = list(mcolors.CSS4_COLORS)
    random.shuffle(colors)
    dirs  = []
    labels = []
    path = "../results/NeurIPS/PPO_Tuning_Gail/"
    cl = ['blue', 'green']
    """
    # lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]     
    # lrs = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    n_steps = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000 ]

    for i in range(10):
        dirs.append(path+str(i+10)+'/')
        labels.append(str(n_steps[i]))
    """
    base_dir = '../results/ICRA/positionNoise/'
    labels = []
    for f in os.listdir(base_dir):
        labels.append(f)
        dirs.append(base_dir+f)
    # labels = ['150','200'] ?
    # dirs = []?
    # for i in labels:
        # dirs.append(base_dir+i+'/')
    print(dirs)

    dirs = ['../results/actionStep/100/','../results/actionStep/200/','../results/actionStep/300/','../results/actionStep/400/','../results/actionStep/500/']
    labels = ['100 pi/s^2', '200 pi/s^2', '300 pi/s^2', '400 pi/s^2', '500 pi/s^2']
    # plot_action_results(log_folders=dirs, labels=labels)
    plot_results(log_folders=dirs, labels=labels)
    


