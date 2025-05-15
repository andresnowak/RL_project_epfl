import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional

from data.trajectory import Trajectory


class RewardModel(nn.Module):
    """
    Given an state, this network outputs the reward.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_dims: List[int] = [64, 64],
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_actions = action_dim is not None

        # Validate dimensions
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        # Get activation functions
        activation_fn = nn.ReLU()
        output_activation_fn = nn.Tanh()

        # Calculate input dimension
        input_dim = state_dim + (action_dim if self.use_actions else 0)

        # Build network layers dynamically
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), activation_fn])
            current_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, 1))
        # if output_activation_fn:
        #     layers.append(output_activation_fn)

        self.reward_net = nn.Sequential(*layers).to(device)

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
        return rewards.sum()
