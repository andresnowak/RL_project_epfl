import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gymnasium as gym


class ActorNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, alpha=1e-3, fc_dims=256):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, alpha, fc_dims=256):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim, fc_dims), nn.ReLU(), nn.Linear(fc_dims, 1))

    def forward(self, state):
        value = self.critic(state)

        return value


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, alpha=1e-3, fc_dims=256):
        super(PPOPolicy, self).__init__()
        self.actor = ActorNetwork(action_dim, state_dim, alpha, fc_dims)
        self.critic = CriticNetwork(state_dim, alpha, fc_dims)

    def forward(self, state):
        # Unsure the state is a tensor
        assert isinstance(state, torch.Tensor), "State must be a tensor"

        dist = self.actor(state)
        value = self.critic(state)
        return dist, value


if __name__ == "__main__":
    # Example usage
    env = gym.make("CartPole-v1")
    actor = ActorNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0])
    actor.load_state_dict(torch.load("/Users/iris/Documents/master@epfl/ma2/rl/rl_github/rl_2/checkpoints_cartpole/best_actor_model_CP"))

    dist = actor(torch.tensor(env.reset()[0]))
    print(dist)
