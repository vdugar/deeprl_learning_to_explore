import numpy as np
import gym
from envs.grid_nd.NDGrid import NDGrid
from envs.manipulator import Manipulator

map_args = {
    'type': 'generator',
    'num_obs': 5,
    'max_width': .1,
    'difficulty': 4
}

dims = 3
lengths = [0.3] * dims
dtheta = np.pi / 10
env = Manipulator(dims, np.pi/10, lengths, map_args, partial_obs=False)

state, coords = env.reset()
print("Start Coords: ")
print(coords)
print("Goal Coords: ")
print(env.goal_coords)
print("Graph channel: ")
print(state[0])
print("Goal channel")
print(state[1])
print("************")

# for i in range(env.n):
#     state, reward, is_terminal, _ = env.step(i)

#     print("Action: ")
#     print(env.actions[i])
#     print("Coords: ")
#     print(state[1])
#     print("Graph channel: ")
#     print(state[0][0])
#     print("Goal channel")
#     print(state[0][1])
#     print("reward: %f" % reward)
#     print("is terminal: %r" % is_terminal)
#     print("************")

# env.state_coords = env.goal_coords - env.actions[env.n-1][1:]
# state, reward, is_terminal, _ = env.step(env.n-1)
# print("Coords: ")
# print(state[1])
# print("Graph channel: ")
# print(state[0][0])
# print("Goal channel")
# print(state[0][1])
# print("reward: %f" % reward)
# print("is terminal: %r" % is_terminal)
# print("************")