import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from models.ppo_policy import ActorNetwork, CriticNetwork
from models.reward_andres import RewardModel
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gymnasium as gym
import copy
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)
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
        batch_indices = BatchSampler(
            SubsetRandomSampler(range(n_states)), self.batch_size, drop_last=False
        )
        return batch_indices

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
        lr=0.001,
        gamma=0.99,
        clip_epsilon=0.2,
        beta=0.01,
        lam=0.95,
        seed=42,
        max_steps_per_episode=1_000,
        n_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        super().__init__()
        self.device = device
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # Only for discrete env you use n
        self.memory = PPOMemory(32)

        # Initialize actor and move to device
        self.actor = ActorNetwork(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        ).to(device)

        # Load pretrained actor from "half" model
        self.actor.load_state_dict(
            torch.load(Path(actor_model_path), map_location=device)
        )

        # Frozen reference actor
        self.actor_ref = copy.deepcopy(self.actor)
        self.actor_ref.eval()

        # Initialize critic and move to device
        self.critic = CriticNetwork(self.state_dim, 0.5).to(self.device)

        # Initialize reward model
        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            device=device,
        ).to(device)

        # Initialize optimizers
        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        self.ppo_optimizer = torch.optim.Adam(
            [{"params": self.actor.parameters()}, {"params": self.critic.parameters()}],
            lr=lr,
        )

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.lam = lam
        self.seed = seed
        self.max_steps_per_episode = max_steps_per_episode
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.cliprange_value = 0.2

        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def get_action(self, state):
        """Get action, log prob, and value for a given state"""
        # Make sure state is a tensor on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)

        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return (action.cpu().item(), log_prob.cpu().item(), value.cpu().item())

    def compute_advantages(self, rewards, values, dones, last_value=0):
        """Compute advantages using GAE"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0

        # Append the bootstrap value
        values_extended = np.append(values, last_value)

        # Compute GAE
        for t in reversed(range(len(rewards))):
            # Delta: r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            delta = (
                rewards[t]
                + self.gamma * values_extended[t + 1] * (1 - dones[t])
                - values[t]
            )

            # GAE: delta + gamma * lambda * (1-done) * last_gae
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

            # Compute returns (for critic loss)
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def train_reward_model(self, preferences, n_epochs, batch_size):
        """Train the reward model on preference data"""
        self.reward_net.train()
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create batches
        batches = [
            preferences[i : i + batch_size]
            for i in range(0, len(preferences), batch_size)
        ]

        # Train for n_epochs
        for epoch in tqdm(range(n_epochs), desc="Training Reward Model"):
            epoch_loss = 0
            for batch in batches:
                chosen_rewards = []
                rejected_rewards = []

                # Process each preference pair
                for chosen_trajectory, rejected_trajectory in batch:
                    chosen_rewards.append(
                        self.reward_net.reward_trajectory(chosen_trajectory)
                    )
                    rejected_rewards.append(
                        self.reward_net.reward_trajectory(rejected_trajectory)
                    )

                # Stack rewards
                chosen_rewards = torch.stack(chosen_rewards)
                rejected_rewards = torch.stack(rejected_rewards)

                # Bradley-Terry preference loss
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

                # Optimization step
                self.reward_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.reward_net.parameters(), self.max_grad_norm
                )
                self.reward_optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    def actor_loss(self, states, actions, old_log_probs, advantages):
        """Compute PPO actor loss"""
        # Get current policy distribution
        dist = self.actor(states)
        log_probs = dist.log_prob(actions)

        # Compute ratio and clipped ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )

        # Actor loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute entropy for exploration
        entropy = dist.entropy().mean()

        # Compute KL divergence for reference policy constraint
        with torch.no_grad():
            ref_dist = self.actor_ref(states)
        kl_div = torch.distributions.kl.kl_divergence(dist, ref_dist).mean()

        # Total actor loss including entropy bonus and KL penalty
        total_loss = policy_loss - self.entropy_coef * entropy + self.beta * kl_div

        return total_loss, policy_loss, entropy, kl_div

    def critic_loss(self, states, old_values, returns):
        """Compute critic loss with value clipping"""
        # Get current value estimates
        values = self.critic(states)

        # Compute value loss with clipping
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.cliprange_value, self.cliprange_value
        )

        # Calculate losses
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2

        # Take maximum (worst case)
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        return value_loss

    def update_policy(
        self, states, actions, old_log_probs, old_values, advantages, returns
    ):
        """Update policy using PPO"""

        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute actor and critic losses
        actor_total_loss, policy_loss, entropy, kl_div = self.actor_loss(
            states, actions, old_log_probs, advantages
        )
        value_loss = self.critic_loss(states, old_values, returns)

        # Combined loss
        total_loss = actor_total_loss + self.value_coef * value_loss

        total_norm = 0.0
        for p in list(self.actor.parameters()) + list(self.critic.parameters()):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Optimization step
        self.ppo_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.ppo_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_div": kl_div.item(),
            "gradient_norm": total_norm,
        }

    def train(self, num_episodes: int, preferences=None):
        """Main training loop"""
        # Train reward model if preferences are provided
        if preferences:
            self.train_reward_model(preferences, n_epochs=200, batch_size=128)
            self.reward_net.eval()  # Set reward model to evaluation mode

        episode_counter = 0
        total_episodes = 0
        total_epoch_metrics = []

        # Main training loop
        for iteration in range(num_episodes):
            print(f"Starting iteration {iteration + 1}/{num_episodes}")

            # Collect experience using current policy
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            # Reset episode counter for this iteration
            episode_counter = 0

            # Collect data for a fixed number of steps (or multiple episodes)
            for step in range(self.max_steps_per_episode):
                # Get action from policy
                action, log_prob, value = self.get_action(state)

                # Take action in environment
                next_state, env_reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Get reward from reward model if available
                if self.reward_net is not None:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(
                            self.device
                        ).unsqueeze(0)
                        action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)

                        reward = self.reward_net(state_tensor, action_tensor).item()
                else:
                    reward = env_reward

                # Store experience
                self.memory.store_memory(state, action, log_prob, value, reward, done)

                episode_reward += reward
                episode_length += 1

                # If episode ended
                if done:
                    episode_counter += 1
                    total_episodes += 1
                    print(
                        f"Episode {total_episodes}, Length: {episode_length}, Reward: {episode_reward:.4f}"
                    )

                    # Reset environment
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    state = next_state

                # If we've collected enough steps, break
                if step >= self.max_steps_per_episode - 1:
                    break

            # Get data from memory
            states, actions, old_log_probs, old_values, rewards, dones = (
                self.memory.get_memory()
            )

            # Get value of last state for bootstrapping
            if done:
                last_value = 0
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(
                        self.device
                    )
                    last_value = self.critic(state_tensor).cpu().item()

            # Compute advantages and returns
            advantages, returns = self.compute_advantages(
                rewards, old_values, dones, last_value
            )

            # Convert numpy arrays to tensors
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(
                self.device
            )
            old_values = torch.tensor(old_values, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)


            # Perform multiple epochs of mini-batch updates
            for epoch in range(self.n_epochs):
                # Generate batches
                batch_indices = self.memory.generate_batches()

                # Initialize metrics for this epoch
                epoch_metrics = {
                    "total_loss": 0,
                    "policy_loss": 0,
                    "value_loss": 0,
                    "entropy": 0,
                    "kl_div": 0,
                    "gradient_norm": 0,
                }

                # Update policy for each batch
                for indices in batch_indices:
                    batch_states = states[indices]
                    batch_actions = actions[indices]
                    batch_old_log_probs = old_log_probs[indices]
                    batch_old_values = old_values[indices]
                    batch_advantages = advantages[indices]
                    batch_returns = returns[indices]

                    # Update policy
                    metrics = self.update_policy(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_old_values,
                        batch_advantages,
                        batch_returns,
                    )

                    # Accumulate metrics
                    for key in epoch_metrics:
                        epoch_metrics[key] += metrics[key]

                # Average metrics over batches
                num_batches = len(list(batch_indices))
                for key in epoch_metrics:
                    epoch_metrics[key] /= num_batches

                total_epoch_metrics.append(epoch_metrics)

                # Log metrics
                print(
                    f"Iteration {iteration + 1}, Epoch {epoch + 1}/{self.n_epochs}: "
                    f"Loss: {epoch_metrics['total_loss']:.4f}, "
                    f"Policy Loss: {epoch_metrics['policy_loss']:.4f}, "
                    f"Value Loss: {epoch_metrics['value_loss']:.4f}, "
                    f"Entropy: {epoch_metrics['entropy']:.4f}, "
                    f"KL Div: {epoch_metrics['kl_div']:.4f}, "
                    f"gradient_norm: {epoch_metrics['gradient_norm']}"
                )

            # Clear memory for next iteration
            self.memory.clear_memory()


        # Extract metric names (keys) from the first entry
        metrics = list(total_epoch_metrics[0].keys())
        num_metrics = len(metrics)

        # Create subplots
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [entry[metric] for entry in total_epoch_metrics]
            axes[i].plot(values, marker='o', linestyle='-')  # Plot with markers and lines
            axes[i].set_title(f"{metric} over Steps")
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(metric)

        plt.tight_layout()  # Adjust spacing
        plt.show()

        print(f"Training completed. Total episodes: {total_episodes}")
        return self.actor, self.critic, self.reward_net
