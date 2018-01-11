import numpy as np
import gym
from envs.grid_nd.NDGrid import NDGrid
from envs.manipulator import Manipulator

dims = 3
lengths = [0.3] * dims
dtheta = np.pi / 6
max_difficulty = 10
num_envs_per_difficulty = 10000
data_path = './env_data/'
num_obs = 4
max_width = 0.5
min_width = 0.3
partial_obs = False
num_per_obstacle_map = 10

print("dtheta: %f" % dtheta)
print("num obstacles: %d" % num_obs)
print("Max obstacle width: %f" % max_width)
print("Lengths: %s" % lengths)

for i in range(1, max_difficulty+1):
    print("Generating environments for difficulty %d" % i)
    envs = []
    count = 1
    map_args = {
        'type': 'generator',
        'num_obs': num_obs,
        'max_width': max_width,
        'min_width': min_width,
        'difficulty': i,
        'train': False
    }
    for j in range(0, num_envs_per_difficulty):
        # env = NDGrid(2, 8, map_args)    
        if j%1000 == 0:
            print("Progress: %d" % j)   
        env = Manipulator(dims, dtheta, lengths, map_args, partial_obs=partial_obs)
        envs.append( (env.start_coords, env.goal_coords, 
            env.full_state[0], env.full_state[1], env.obstacles, env.shortest_path,
            env.action_history) )

        # check if we should keep using the same obstacle map, or generate a new one
        if count < num_per_obstacle_map:
            map_args['type'] = 'cached'
            map_args['cached_map'] = env.full_state[0]
            map_args['cached_obstacles'] = env.obstacles
        else:
            count = 0
            map_args['type'] = 'generator'

        count = count + 1

    envs = np.array(envs)
    np.save(data_path + str(i), envs)


# map_data = np.load(data_path +)

