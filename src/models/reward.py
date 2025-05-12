import torch.nn as nn
import torch
import numpy as np
from data.trajectory import Trajectory


class RewardModel(nn.Module):
    """
    Given an state, this network outputs the reward.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.reward_net(x)

    def reward_trajectory(self, trajectory: Trajectory):
        """
        Given a trajectory, this function computes the reward for each state in the trajectory and returns sum of rewards.
        """
        states = torch.from_numpy(np.array(trajectory.states)).float()
        rewards = self.reward_net(states)
        return rewards.mean()
