import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from data.trajectory import Trajectory


class RewardModel(nn.Module):
    """
    Given an state, this network outputs the reward.
    """

    def __init__(self, state_dim: int, action_dim: int | None = None, hidden_dim: int = 128, device = "cpu"):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, state, action):
        # assume `action` is already a one-hot or float tensor of shape (batch, action_dim)
        action_onehot = F.one_hot(action.view(-1), num_classes=self.action_dim).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.reward_net(x).squeeze(-1)

    def reward_trajectory(self, trajectory: Trajectory):
        """
        Given a trajectory, this function computes the reward for each state in the trajectory and returns sum of rewards.
        """
        states = trajectory.states.to(self.device)
        actions = trajectory.actions.to(self.device)
        rewards = self.forward(states, actions)
        return rewards.mean()
