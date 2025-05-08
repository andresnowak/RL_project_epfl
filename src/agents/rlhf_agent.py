import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from models.ppo_policy import PPOPolicy
from models.reward import RewardModel

logger = logging.getLogger(__name__)

# setup logging to terminal
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class PPORLHFAgent:
    def __init__(
        self,
        env,
        device,
        is_continuous_env,
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
        self.is_continuous_env = is_continuous_env

        if is_continuous_env:
            self.action_dim = env.action_space.shape[0]
            self.action_bound = env.action_space.high[0]
        else:
            self.action_dim = env.action_space.n
            self.action_bound = None

        self.policy_net = PPOPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            action_bound=self.action_bound,
        ).to(device)

        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            hidden_dim=256,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy_net.parameters(), "lr": lr},
                {"params": self.reward_net.parameters(), "lr": lr},
            ]
        )
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.lam = lam
        self.seed = seed

    def train_reward_model(self, preferences, epochs, batch_size):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        batches = [
            preferences[i : i + batch_size]
            for i in range(0, len(preferences), batch_size)
        ]

        for epoch in tqdm(range(epochs), desc="Training Reward Model"):
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
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def compute_returns_and_advantages(
        self, rewards, dones, values, next_value, next_done
    ):
        """Compute returns using Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = (
                rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            )
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            )
        returns = advantages + values
        return returns, advantages

    def update_policy(
        self, states, actions, log_probs_old, advantages, returns, epochs
    ):
        states = torch.FloatTensor(states).to(self.device)
        if self.is_continuous_env:
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # for _ in tqdm(range(epochs), desc="Policy Update"):
        for _ in range(epochs):
            log_probs_new, state_values, dist_entropy = self.policy_net.evaluate(
                states, actions
            )

            # Compute ratio
            ratios = torch.exp(log_probs_new - log_probs_old)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            critic_loss = F.mse_loss(state_values, returns)

            # Total loss
            total_loss = (
                actor_loss + 0.5 * critic_loss - self.beta * dist_entropy.mean()
            )

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def evaluate(self, num_episodes=10, max_steps=300, render=False):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            for _ in range(max_steps):
                with torch.no_grad():
                    action, _ = self.policy_net.act(state)
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                if render:
                    self.env.render()
                if done:
                    break
            total_rewards.append(episode_reward)

        if render:
            self.env.close()
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)

        logger.info(
            f"Evaluation over {num_episodes} episodes: Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}"
        )
        return avg_reward, std_reward

    def train(self, preferences, epochs, batch_size=32, n_episodes=1000, max_steps=200):
        # First train the reward model
        logger.info("Training Reward Model...")
        self.train_reward_model(preferences, epochs, batch_size)

        # Train the policy network
        logger.info("Training Policy Network...")
        episode_rewards = []

        for episode in tqdm(range(n_episodes), desc="Training Policy Network"):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

            for _ in range(max_steps):
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, log_prob = self.policy_net.act(state_tensor)
                next_state, _, done, _, _ = self.env.step(action)

                # Important: Get reward from the reward model
                with torch.no_grad():
                    reward = self.reward_net(state_tensor.view(1, -1)).item()
                    value = self.policy_net.critic(state_tensor).item()

                # Store transition
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Compute returns and advantages
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(state).to(self.device)
                next_value = self.policy_net.critic(next_state_tensor).item()

            returns, advantages = self.compute_returns_and_advantages(
                rewards, dones, values, next_value, done
            )

            # Update policy
            self.update_policy(states, actions, log_probs, advantages, returns, epochs)

            episode_rewards.append(episode_reward)
            logger.info(
                f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward:.2f}"
            )
