import numpy as np
import torch

class Trajectory:
    """
    Class to store trajectory data.
    """

    def __init__(self, states, actions=None, rewards=None, next_states=None, done=None):
        self.states = torch.from_numpy(np.array(states)).float()
        self.actions = torch.from_numpy(np.array(actions))
        self.done = done

    def __len__(self):
        return len(self.states)
