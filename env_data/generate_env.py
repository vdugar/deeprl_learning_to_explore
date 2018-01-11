import numpy as np
import gym
import init_paths
from envs.grid_nd.NDGrid import NDGrid
from envs.manipulator import Manipulator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff', type=int, default=1,
                    help='difficulty')
    parser.add_argument('--path', type=str, default='./',
                help='difficulty')
    args = parser.parse_args()

    dims = 3
    lengths = [0.3] * dims
    dtheta = np.pi / 12
    max_difficulty = 10
    num_envs_per_difficulty = 2
    data_path = args.path
    num_obs = 4
    max_width = 0.1
    min_width = 0.1
    partial_obs = False
    num_per_obstacle_map = 10

    print("dtheta: %f" % dtheta)
    print("num obstacles: %d" % num_obs)
    print("Max obstacle width: %f" % max_width)
    print("Lengths: %s" % lengths)

    print("Generating environments for difficulty %d" % args.diff)
    envs = []
    count = 1
    map_args = {
        'type': 'generator',
        'num_obs': num_obs,
        'max_width': max_width,
        'min_width': min_width,
        'difficulty': args.diff,
        'train': False
    }
    for j in range(0, num_envs_per_difficulty):
        # env = NDGrid(2, 8, map_args)    
        print j
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
    np.save(data_path + str(args.diff), envs)


if __name__ == '__main__':
    main()

