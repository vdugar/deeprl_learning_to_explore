import numpy as np
import gym
from grid_nd.NDGrid import NDGrid

env = NDGrid(2, 32, 1)

env.reset()
env.step(env.n/2+1)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords
env.step(env.n/2+4)
print env.state_coords