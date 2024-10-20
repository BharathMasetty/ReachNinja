import json
import numpy as np

# dump

# exploding_perc = 0.33, max_unobs_time = 0.2, max_obs_time = 0.8, 
#                                 vel_max = 0.5, vel_min = 1, acc = 100, theta_max = -30, theta_min = -150,
#                                 min_obstacles = 1, max_obstacles = 4, damping = 0, mirror = False

params = {}

exploding_perc_levels = [0, 0.166, 0.33]
max_unobs_levels = [0, 0.08, 0.15]
max_vel_levels = [0, 0.25, 0.5]
min_vel_levels = [0, 0.5, 1]
acc_levels = [0, 50, 100]
min_theta_levels = [-91, -120, -150] 
max_theta_levels= [-90, -60, -30]
min_obstacles_levels = [1, 1, 2]
max_obstacles_levels = [2, 3, 4]
damping_val = 0
mirror_val = False

bl = [1,2,3]
pt = [34,35,36,37,38,39,40]

for i in bl:
    params[i] = dict(exploding_perc = 0.33, max_unobs_time = 0.15, max_obs_time = 1, 
                        vel_max = 0.5, vel_min = 1, acc = 100, theta_max = -30, theta_min = -150,
                        min_obstacles = 1, max_obstacles = 4, damping = 0, mirror = False)    

vel_ac = 0
obstacle = 1
exploding = 2
observ = 3
angle = 4

level_array = np.array([0,0,0,0,0])

expt_ctr = bl[-1]+1

for i in range(0,10):

    internal_grp = i%5
    if i > 0:
        level_array[internal_grp - 1] += 1

    for _ in range(0,3):

        params[expt_ctr] = dict(exploding_perc = exploding_perc_levels[level_array[exploding]], 
                    max_unobs_time = max_unobs_levels[level_array[observ]], 
                    max_obs_time = 1, 
                    vel_max = max_vel_levels[level_array[vel_ac]], 
                    vel_min = min_vel_levels[level_array[vel_ac]], 
                    acc = acc_levels[level_array[vel_ac]], 
                    theta_max = max_theta_levels[level_array[angle]], 
                    theta_min = min_theta_levels[level_array[angle]],
                    min_obstacles = min_obstacles_levels[level_array[obstacle]],
                    max_obstacles = max_obstacles_levels[level_array[obstacle]],
                    damping = 0,
                    mirror = False)    
        expt_ctr += 1


for i in pt:
    params[i] = dict(exploding_perc = 0.33, max_unobs_time = 0.15, max_obs_time = 1, 
                        vel_max = 0.5, vel_min = 1, acc = 100, theta_max = -30, theta_min = -150,
                        min_obstacles = 1, max_obstacles = 4, damping = 0, mirror = False)    



with open('GameParams.json','w') as f:
    json.dump(params, f, indent=2)






