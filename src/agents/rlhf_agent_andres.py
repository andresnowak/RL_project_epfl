import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from models.ppo_policy import ActorNetwork, CriticNetwork
from models.reward import RewardModel
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gymnasium as gym
import copy

logger = logging.getLogger(__name__)

# setup logging to terminal
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

        return batches

    def get_memory(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
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


class PPORLHFAgent:
    def __init__(
        self,
        env: gym.Env,
        device,
        actor_model_path: str,
        critic_model_path: str,
        lr=0.001,
        gamma=0.99,
        clip_epsilon=0.2,
        beta=0.01,
        lam=0.95,
        seed=42,
    ):
        super().__init__()
        self.device = device
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n # Only for discrete env you use n, if not you use shape

        self.memory = PPOMemory(32)

        self.actor = ActorNetwork(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

        # Load pretrained actor from "half" model
        self.actor.load_state_dict(
            torch.load(Path(actor_model_path), map_location=device)
        )

        # Frozen reference actor
        self.actor_ref = copy.deepcopy(self.actor)
        self.actor_ref.eval()

        self.critic = CriticNetwork(self.state_dim, 0.5).to(self.device)
        # self.critic.load_state_dict(torch.load(Path(critic_model_path), map_location=device))

        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            hidden_dim=256,
            device=device,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr},
                {"params": self.critic.parameters(), "lr": lr},
                {"params": self.reward_net.parameters(), "lr": lr},
            ]
        )
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.lam = lam
        self.seed = seed

    def get_action(self, state: torch.Tensor):
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
        

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards) - 1)): # it should be up to the end, but we don't have the bootstrap advantage right now
            if dones[t]:
                last_advantage = 0
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1-dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values.unsqueeze(-1).numpy()
        return advantages, returns

    def train_reward_model(self, preferences, n_epochs, batch_size):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        batches = [
            preferences[i : i + batch_size]
            for i in range(0, len(preferences), batch_size)
        ]

        for epoch in tqdm(range(n_epochs), desc="Training Reward Model"):
            epoch_loss = 0
            for batch in batches:
                chosen_rewards = []
                rejected_rewards = []
                for chosen_trajectory, rejected_trajectory in batch:
                    chosen_rewards.append(
                        self.reward_net.reward_trajectory(chosen_trajectory)
                    )
                    rejected_rewards.append(
                        self.reward_net.reward_trajectory(rejected_trajectory)
                    )

                chosen_rewards = torch.stack(chosen_rewards)
                rejected_rewards = torch.stack(rejected_rewards)
                # print(chosen_rewards - rejected_rewards)
                # print(F.sigmoid(chosen_rewards - rejected_rewards))
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    def actor_loss(self, probs, og_probs_old, advantages):
        ratio = probs / og_probs_old
        ratio = ratio.detach()

        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()  # The actor loss (PPO normal)

        return policy_loss

    def critic_loss(self, values, old_values, returns):
        self.cliprange_value = 0.2
    
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = (
            0.5 * torch.sum(torch.max(vf_loss1, vf_loss2))
        )  

        return vf_loss

    def update_policy(
        self, states, actions, og_probs_old, advantages, returns, old_values
    ):
        dist = self.actor(states)
        probs = dist.log_prob(actions).exp()
        
        policy_loss = self.actor_loss(probs, og_probs_old, advantages)
        values = self.critic(states)

        critic_loss = self.critic_loss(values, old_values, returns)
        
        # Calculate KL divergence between current and reference policy
        with torch.no_grad():
            ref_dist = self.actor_ref(states)
        kl = torch.distributions.kl.kl_divergence(dist, ref_dist).mean()
        
        # Calculate entropy
        entropy = dist.entropy().mean()
        
        # Calculate value loss if using a critic (not shown in your code)
        # value_loss = F.mse_loss(values, batch_returns)
        
        # Total loss
        loss = (
            policy_loss + 0.5 * critic_loss + self.beta * kl - 0.00 * entropy
        )  # + value_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()

        return loss

    def train(self, env: gym.Env, num_episodes: int, epochs: int, preferences):
        self.train_reward_model(preferences, 100, 64)
        self.reward_net = self.reward_net.eval()
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(10_000):
                # Get action from policy
                action, log_prob, value = self.get_action(torch.tensor(state).to(self.device))

                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action)
                state_tensor = torch.tensor(next_state).to(self.device)
                reward = self.reward_net(state_tensor)

                # Store experience
                self.memory.store_memory(state, action, log_prob, value, reward.detach(), done)

                episode_reward += reward.item()
                state = next_state

                if done:
                    break
            
            state_arr, action_arr, old_prob_arr, old_vals_arr, reward_arr, dones_arr = self.memory.get_memory()
            states = torch.tensor(state_arr).to(self.device)
            actions = torch.tensor(action_arr).to(self.device)
            old_probs = torch.tensor(old_prob_arr).to(self.device)
            old_values = torch.tensor(old_vals_arr).to(self.device)
            rewards = torch.tensor(reward_arr).to(self.device)
            dones = torch.tensor(dones_arr, dtype=torch.float32).to(self.device)


            advantages, returns = self.compute_advantages(rewards, old_values, dones)

            advantages = torch.tensor(advantages).to(self.device)
            returns = torch.tensor(returns).to(self.device)

            for _ in range(10):
                total_loss = 0
                batches = self.memory.generate_batches()
                
                for batch in batches:

                    dist = self.actor(states)

                    new_probs = dist.log_prob(actions)

                    # Update policy after each episode
                    loss = self.update_policy(states[batch], actions[batch], old_probs[batch], advantages[batch], returns[batch], old_values[batch])

                    total_loss += loss.item()
            
                print("Total loss: ", total_loss / len(batches))

            self.memory.clear_memory()

            print(f"Episode {episode}, Reward: {episode_reward}")
