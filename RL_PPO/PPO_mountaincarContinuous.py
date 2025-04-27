import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
from itertools import count
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils import plot_learning_curve
from copy import deepcopy
from torch.distributions import Normal
import torch.nn.functional as F


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
        batches = BatchSampler(SubsetRandomSampler(range(n_states)), self.batch_size, drop_last=False)
        # n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)
        # indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices)
        # batches = [indices[i : i + self.batch_size] for i in batch_start]

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


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, action_bound, alpha=1e-3, fc_dims=256, chkpt_dir="RL_PPO/model/"):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(nn.Linear(*input_dims, fc_dims), nn.ReLU(), nn.Linear(fc_dims, fc_dims), nn.ReLU())
        self.mu = nn.Linear(fc_dims, *n_actions)
        self.sigma = nn.Linear(fc_dims, *n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.action_bound = action_bound

    def forward(self, state):
        x = self.actor(state)  # tensor([[1.]], grad_fn=<SoftmaxBackward0>)

        mu = F.tanh(self.mu(x)) * self.action_bound
        sigma = F.softplus(self.sigma(x))

        return mu, sigma

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model=None):
        if model is None:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(model))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc_dims=256, chkpt_dir="RL_PPO/model"):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(nn.Linear(*input_dims, fc_dims), nn.ReLU(), nn.Linear(fc_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPO:
    def __init__(self, n_actions, input_dims, action_bound, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.action_bound = action_bound

        self.actor = ActorNetwork(n_actions, input_dims, action_bound, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, actor_model=None):
        print("... loading models ...")
        self.actor.load_checkpoint(actor_model)
        # self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)

        mu, sigma = self.actor(state)
        value = self.critic(state)
        dist = Normal(mu, sigma)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        value = torch.squeeze(value).item()
        action = torch.clamp(action, -self.action_bound, self.action_bound)

        return action, probs, value

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

                mu, sigma = self.actor(states)
                dist = Normal(mu, sigma)
                entropy = dist.entropy().sum(dim=-1)
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

                total_loss = actor_loss + 0.5 * critic_loss  # - 0.01 * entropy.mean()
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        # print("total learning iter: ", n_learning)
        self.memory.clear_memory()


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    # N = 30
    batch_size = 32
    n_epochs = 10
    alpha = 1e-4  # lr
    agent = PPO(n_actions=env.action_space.shape, action_bound=float(env.action_space.high[0]), batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_episode = 100
    # min_rewards = -1000

    best_score = env.reward_range[0]
    score_history = []
    model_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    seed = 1
    # min_score = -1000
    # torch.manual_seed(seed)

    for i in range(n_episode):
        observation, _ = env.reset(seed=seed)
        done = False
        score = 0

        while 1:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, truncated, _ = env.step(action)

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            observation = observation_
            if done:
                print(f"episode {i} done... Score: {score}")
                agent.learn()
                learn_iters += 1
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                model_history.append(deepcopy(agent.actor.state_dict()))
                break

        print("episode", i, "score %.1f" % score, "avg score %.1f" % avg_score, "time_steps", n_steps, "learning_steps", learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "RL_PPO/learning_curve_MCContinuous.png")

    env.close()

    # Get prefered model and half quality one
    avg_score_history = np.zeros(len(score_history))
    for i in range(len(score_history)):
        avg_score_history[i] = np.mean(score_history[max(0, i - 100) : (i + 1)])
    avg_score_history = avg_score_history.tolist()
    best_idx = avg_score_history.index(max(avg_score_history))
    best_model = model_history[best_idx]

    half_idx = min(range(len(avg_score_history)), key=lambda i: abs((avg_score_history[i] - min(avg_score_history)) - (avg_score_history[best_idx] - min(avg_score_history)) / 2))
    half_model = model_history[half_idx]

    # save model
    print("save best model with score of", avg_score_history[best_idx])
    torch.save(best_model, "RL_PPO/model/best_actor_model_MCContinuous")
    print("save half model with score of", avg_score_history[half_idx])
    torch.save(half_model, "RL_PPO/model/half_actor_model_MCContinuous")

    # visualize the best model

    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    agent = PPO(n_actions=env.action_space.shape, input_dims=env.observation_space.shape, action_bound=float(env.action_space.high[0]))
    agent.load_models("RL_PPO/model/best_actor_model_MCContinuous")

    n_try = 1
    for _ in range(n_try):
        state, _ = env.reset()
        for _ in range(500):

            action, probs, value = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            state = state_
            env.render()
            if done or truncated:
                break

    env.close()
