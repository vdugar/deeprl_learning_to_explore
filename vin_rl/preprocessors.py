"""Suggested Preprocessors."""

from core import Preprocessor
import numpy as np


class GridPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def process_state_for_network(self, state):
        grid_map_channels = np.array(state[0], dtype=np.float32)
        coords = np.array(state[1], dtype=np.float32)
        return grid_map_channels, coords

    def process_state_for_memory(self, state):
        return state

    def reset(self):
        pass

    def process_batch(self, samples):
        states = [x.state for x in samples]
        next_states = [x.next_state for x in samples if x.is_terminal is False]
        grids = [x[0] for x in states]
        coords = [x[1] for x in states]
        next_grids = [x[0] for x in next_states]
        next_coords = [x[1] for x in next_states]
        rewards = [x.reward for x in samples]
        mask = [x.is_terminal is False for x in samples]
        actions = [x.action for x in samples]
        return np.array(grids, dtype=np.float32), np.array(coords, dtype=np.float32), np.array(next_grids, dtype=np.float32), np.array(next_coords, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(mask, dtype=np.int), np.array(actions, dtype=np.int)
