import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
from itertools import count
from torch.utils.data import BatchSampler, SubsetRandomSampler
from src.models.utils import plot_learning_curve
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
        batches = BatchSampler(
            SubsetRandomSampler(range(n_states)), self.batch_size, drop_last=False
        )
        # n_states = len(self.states)
        # batch_start = np.arange(0, n_states, self.batch_size)
        # indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices)
        # batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def get_memory(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.probs, dtype=np.float32),
            np.array(self.vals, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )

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
    def __init__(
        self,
        n_actions,
        input_dims,
        action_bound,
        alpha=1e-3,
        fc_dims=256,
        device: str | torch.device = "cpu",
        chkpt_dir="RL_PPO/model/",
    ):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fc_dims, *n_actions)
        self.sigma = nn.Linear(fc_dims, *n_actions)

        self.action_bound = action_bound

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)
    
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize layers within the sequential 'features' part
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                # Use gain=sqrt(2) for layers followed by ReLU
                torch.nn.init.orthogonal_(layer.weight, gain=torch.nn.init.calculate_gain("relu"))
                torch.nn.init.constant_(layer.bias, 0)

        # Initialize output layers (mu and sigma) with specific gains
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        torch.nn.init.constant_(self.mu.bias, 0)
        torch.nn.init.orthogonal_(self.sigma.weight, gain=0.01)
        torch.nn.init.constant_(self.sigma.bias, 0)

    def forward(self, state):
        x = self.actor(state)  # tensor([[1.]], grad_fn=<SoftmaxBackward0>)

        mu = F.tanh(self.mu(x)) * self.action_bound
        sigma = F.softplus(self.sigma(x)) + 1e-8

        return mu, sigma

    def save_checkpoint(self, path=""):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model=None):
        if model is None:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(model))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        fc_dims=256,
        device: str | torch.device = "cpu",
        chkpt_dir="RL_PPO/model",
    ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc_dims), nn.ReLU(), nn.Linear(fc_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize layers within the sequential 'features' part
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                # Use gain=sqrt(2) for layers followed by ReLU
                torch.nn.init.orthogonal_(
                    layer.weight, gain=torch.nn.init.calculate_gain("relu")
                )
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self, path=""):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path=""):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPO:
    def __init__(
        self,
        env: gym.Env,
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2, # clip_range
        batch_size=64,
        n_epochs=10,
        ent_coef=0.0,
        vf_coef=0.5,
        device: str | torch.device = "auto"
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        n_actions = env.action_space.shape
        input_dims = env.observation_space.shape

        action_bound = float(env.action_space.high[0])

        self.action_bound = action_bound


        self.device = "cpu"
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_bound_low = torch.tensor(self.action_space.low, dtype=torch.float).to(self.device)
        self.action_bound_high = torch.tensor(self.action_space.high, dtype=torch.float).to(self.device)

        # setup model
        self.actor = ActorNetwork(n_actions, input_dims, action_bound, alpha, device=self.device)
        self.critic = CriticNetwork(input_dims, alpha, device=self.device)
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

    def log_probs(self, observation: torch.Tensor):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)

        mu, sigma = self.actor(state)
        value = self.critic(state)
        dist = Normal(mu, sigma)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).sum(dim=-1).item()
        value = torch.squeeze(value).item()
        action = torch.clamp(action, self.action_bound_low, self.action_bound_high)

        return action, probs, value

    def predict(self, observation: np.ndarray):
        self.actor.eval() # Set actor to evaluation mode

        obs_tensor = torch.tensor(observation, dtype=torch.float).to(self.device)
        # Add batch dimension if processing a single observation
        if obs_tensor.ndim == len(self.observation_space.shape):
             obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            mu, sigma = self.actor(obs_tensor)
   
            # Sample from the distribution
            dist = Normal(mu, sigma)
            action = dist.sample()

            # Clamp action to valid range
            action = torch.clamp(action, self.action_bound_low, self.action_bound_high)

        # Remove batch dimension if we added it
        if obs_tensor.ndim == len(self.observation_space.shape) + 1 and action.shape[0] == 1:
             action = action.squeeze(0)

        self.actor.train() # Set back to training mode

        return action.cpu().numpy(), None # Return action and None state

    def collect_rollout(self, rollout_steps=2048):
        obs = self.env.reset()[0]
        last_value = 0
    
        for _ in range(rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                mu, sigma = self.actor(obs_tensor)
                value = self.critic(obs_tensor)
   
                # Sample from the distribution
                dist = Normal(mu, sigma)
                action = dist.sample()

                # Clamp action to valid range
                action = torch.clamp(action, self.action_bound_low, self.action_bound_high)
                log_prob = dist.log_prob(action).sum(dim=-1)

            next_obs, reward, done, truncated, _ = self.env.step(action.cpu().numpy())
            done = done or truncated

            # Store in buffer
            self.memory.store_memory(obs, action.cpu().numpy(), log_prob.cpu().numpy(), value.item(), reward, done)

            if done:
                obs = self.env.reset()[0]
                last_value = 0
            else:
                obs = next_obs
                if _ == rollout_steps - 1:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                        last_value = self.critic(obs_tensor).item()


        # Return the last value for GAE bootstrapping
        return last_value

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = (
            self.memory.get_memory()
        )
        advantages = np.zeros(len(reward_arr), dtype=np.float32)
        gae = 0
        values = np.zeros(len(reward_arr) + 1)
        values[:-1] = vals_arr
        values[-1] = last_value # last value for bootstrapping

        for t in reversed(range(len(reward_arr))):
            delta = reward_arr[t] + gamma * values[t+1] * (1 - dones_arr[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones_arr[t])
            advantages[t] = gae

        returns = [a + v for a, v in zip(advantages, vals_arr)]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages).to(self.actor.device)

        return advantages, returns


    def _learn(self, last_value):
        n_learning = 0
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = (
            self.memory.get_memory()
        )
        values = vals_arr.astype(np.float32)

        advantage, returns_ = self.compute_gae(last_value, self.gamma, self.gae_lambda)

        values = torch.tensor(values).to(self.device)

        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()

            for batch in batches:
                n_learning += 1
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(
                    self.actor.device
                )
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                mu, sigma = self.actor(states)

                dist = Normal(mu, sigma)
                
                entropy = dist.entropy().sum(dim=-1)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(dim=-1)
                # prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
   
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                # critic_loss = (returns - critic_value) ** 2
                # critic_loss = critic_loss.mean()
                critic_loss = F.mse_loss(critic_value, returns)

                total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()
        
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
    
                total_loss.backward()
                # clip the gradients like in SB3
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
    
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        # print("total learning iter: ", n_learning)
        self.memory.clear_memory()

    def learn(self, num_timesteps: int = 10, alpha: float = 1e-4):
        score_history = []
        model_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0
        seed = 1

        buffer_size = 10_000
        for i in range(num_timesteps):
            last_value = self.collect_rollout(buffer_size)

            trajectory_scores = []
            episode_reward = 0
            trajectory_length = []
            length = 0

            for reward, done in zip(self.memory.rewards, self.memory.dones):
                episode_reward += reward
                length += 1

                if done:
                    trajectory_scores.append(episode_reward)
                    trajectory_length.append(length)
                    episode_reward = 0
                    length = 0

            self._learn(last_value)

            score_history.append(np.mean(trajectory_scores))
            avg_score = np.mean(trajectory_scores)

            model_history.append(deepcopy(self.actor.state_dict()))
    
            self.memory.clear_memory()

            n_steps += buffer_size

            print(
                "episode",
                i,
                "avg score %.1f" % avg_score,
                "time_steps",
                n_steps,
                "learning_steps",
                learn_iters,
                "average_length",
                np.mean(trajectory_length)
            )
        x = [i + 1 for i in range(len(score_history))]

        return score_history, model_history

