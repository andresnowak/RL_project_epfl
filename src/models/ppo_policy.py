import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler
from models.base_policy import BasePolicy

# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#         self.batch_size = batch_size

#     def generate_batches(self):
#         n_states = len(self.states)
#         batches = BatchSampler(SubsetRandomSampler(range(n_states)), self.batch_size, drop_last=False)

#         return batches

#     def get_memory(self):
#         return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones)

#     def store_memory(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)

#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []


class DiscreteActor(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=256):
        """
        Args:
            action_dim (int): Dimension of the action space. For discrete actions, this is the number of actions.
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Dimension of the hidden layers. Default is 256.
        """
        super(DiscreteActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        """Return the action disttribution given the states
        For discrete actions, this is a categorical distribution.
        """
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits)

        return dist


class ContinuousActor(nn.Module):
    def __init__(self, action_dim, state_dim, action_bound, hidden_dim=256):
        """
        Args:
            action_dim (int): Dimension of the action space. For continuous actions, this is the number of actions.
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Dimension of the hidden layers. Default is 256.
        """
        super(ContinuousActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        """
        Since the action is continuous, we need to calculate the mean and standard deviation of the action distribution.
        The action distribution is a Gaussian distribution with mean and standard deviation.
        """
        x = self.actor(state)
        mu = F.tanh(self.mu(x)) * self.action_bound
        sigma = F.softplus(self.sigma(x))
        dist = torch.distributions.Normal(mu, sigma)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        """
        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Dimension of the hidden layers. Default is 256.
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """Return the value of the state"""
        return self.critic(state)


class PPOPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, action_bound=None, hidden_dim=256):
        """
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space. For discrete actions, this is the number of actions.
            hidden_dim (int): Dimension of the hidden layers. Default is 256.
        """
        # init base policy
        super(PPOPolicy, self).__init__()
        if action_bound is not None:
            self.actor = ContinuousActor(
                action_dim, state_dim, action_bound, hidden_dim
            )
            self.is_continuous_env = True
        else:
            self.actor = DiscreteActor(action_dim, state_dim, hidden_dim)
            self.is_continuous_env = False

        self.critic = Critic(state_dim, hidden_dim)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.sample()

            if self.is_continuous_env:
                log_prob = dist.log_prob(action).sum(-1)
                return action.squeeze(0).numpy(), log_prob.item()
            else:
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item()

    def evaluate(self, states, actions):
        states = torch.FloatTensor(states)
        if self.is_continuous_env:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.LongTensor(actions)

        dist = self.actor(states)
        state_values = self.critic(states).squeeze()

        if self.is_continuous_env:
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            log_probs = dist.log_prob(actions.squeeze(-1))
            entropy = dist.entropy()

        return log_probs, state_values, entropy
