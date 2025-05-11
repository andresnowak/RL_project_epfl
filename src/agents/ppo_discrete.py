from src.models.ppo_policy import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)
        # indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices)
        batches = BatchSampler(SubsetRandomSampler(range(n_states)), self.batch_size, drop_last=False)

        return batches

    def get_memory(self):
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPOAGENT:
    def __init__(self, env, batch_size, alpha, n_epochs):
        self.memory = PPOMemory(batch_size)
        self.ppo_policy = PPOPolicy(env.observation_space.shape[0], env.action_space.n, alpha=alpha, fc_dims=256)
        self.n_epochs = n_epochs

    def learn(self):
        n_learning = 0
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = self.memory.get_memory()
        values = vals_arr

        advantage = np.zeros(len(reward_arr))
        gae = 0
        for t in reversed(range(len(reward_arr) - 1)):
            delta = reward_arr[t] + self.gamma * values[t + 1] * (1 - int(dones_arr[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae
            advantage[t] = gae

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = torch.tensor(advantage).to(self.actor.device)

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                n_learning += 1
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        print("total learning iter: ", n_learning)
        self.memory.clear_memory()
