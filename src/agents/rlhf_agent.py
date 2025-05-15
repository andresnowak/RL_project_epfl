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
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class PPOTrajectoryMemory:
    def __init__(self, batch_size):
        self.trajectories = []  # Store complete trajectories
        self.batch_size = batch_size

    def generate_batches(self):
        n_trajectories = len(self.trajectories)
        batch_indices = BatchSampler(
            SubsetRandomSampler(range(n_trajectories)), self.batch_size, drop_last=False
        )
        return batch_indices

    def store_trajectory(self, trajectory):
        """Store a complete trajectory"""
        self.trajectories.append(trajectory)

    def get_memory(self):
        """Return all stored trajectories"""
        return self.trajectories

    def clear_memory(self):
        """Clear all stored trajectories"""
        self.trajectories = []


# This is a different trajectory file type
class Trajectory:
    """Class to store and process complete trajectories"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.total_reward = 0  # For the whole trajectory
        self.trajectory_return = 0  # Discounted return

    def add_step(self, state, action, log_prob, value, reward, done):
        """Add a step to the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_reward += reward

    def finalize(self, gamma, last_value=0):
        """Calculate returns and advantages for the trajectory"""
        # Convert lists to numpy arrays for easier processing
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.log_probs = np.array(self.log_probs)
        self.values = np.array(self.values)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)

        # Calculate trajectory return (discounted)
        returns = np.zeros_like(self.rewards)
        advantages = np.zeros_like(self.rewards)

        # Compute returns and advantages
        next_value = last_value
        next_advantage = 0

        for t in reversed(range(len(self.rewards))):
            # For returns
            returns[t] = self.rewards[t] + gamma * next_value * (1 - self.dones[t])
            next_value = returns[t]

            # For advantages (simple version, could be replaced with GAE)
            advantages[t] = returns[t] - self.values[t]

        self.returns = returns
        self.advantages = advantages
        self.trajectory_return = returns[0] if len(returns) > 0 else 0

        return self


class PPORLHFAgent:
    def __init__(
        self,
        env: gym.Env,
        device,
        actor_model_path: str,
        reward_model_hidden_size = [128, 256, 128],
        lr_reward=0.001,
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        beta=0.01,
        lam=0.95,
        seed=42,
        max_steps_per_episode=1_000,
        n_epochs=4,
        entropy_coef=0.001,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        super().__init__()
        self.device = device
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # Only for discrete env you use n
        self.memory = PPOTrajectoryMemory(
            batch_size=4
        )  # Using smaller batch size for trajectories

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
        for param in self.actor_ref.parameters():
            param.requires_grad = False  # Actual freezing

        # Initialize critic and move to device
        self.critic = CriticNetwork(self.state_dim).to(self.device)

        # Initialize trajectory reward model
        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=reward_model_hidden_size,
            device=device,
        ).to(device)

        # Initialize optimizers
        self.reward_optimizer = torch.optim.Adam(
            self.reward_net.parameters(), lr=lr_reward
        )
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=lr_critic
        )

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_reward = lr_reward
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

    def compute_gae(self, rewards, values, dones, last_value=0):
        """Compute advantages using GAE (for trajectory-based advantages)"""
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

    # ---- REWARD MODEL -----
    def train_reward_model(
        self,
        preferences,
        val_preferences,
        n_epochs,
        batch_size,
        eval_iters=10,
        eval_callback=None,
    ):
        """Train the reward model on preference data for complete trajectories"""
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
                    # Get rewards for complete trajectories
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

            if epoch % eval_iters == 0:
                with torch.no_grad():
                    if eval_callback is not None:
                        eval_callback(self, val_preferences)

    def evaluate_trajectory(self, trajectory):
        """Evaluate a complete trajectory using the reward model"""
        with torch.no_grad():
            states = torch.tensor(np.array(trajectory.states), dtype=torch.float32).to(
                self.device
            )
            actions = torch.tensor(np.array(trajectory.actions), dtype=torch.long).to(
                self.device
            )

            # Get trajectory-level reward
            # trajectory_reward = self.reward_net.reward_trajectory(
            #     (states, actions)
            # ).item()
            trajectory_reward = self.reward_net.forward(state=states, action=actions).sum()

            return trajectory_reward

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

    def update_policy(self, trajectories):
        """Update policy using PPO with trajectory data"""
        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_advantages = []
        all_returns = []

        # Flatten trajectories for batch processing
        for trajectory in trajectories:
            states = trajectory.states
            actions = trajectory.actions
            log_probs = trajectory.log_probs
            values = trajectory.values
            advantages = trajectory.advantages
            returns = trajectory.returns

            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_advantages.append(advantages)
            all_returns.append(returns)

        # Convert to tensors
        states = torch.tensor(np.concatenate(all_states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(np.concatenate(all_actions), dtype=torch.long).to(
            self.device
        )
        old_log_probs = torch.tensor(
            np.concatenate(all_log_probs), dtype=torch.float32
        ).to(self.device)
        old_values = torch.tensor(np.concatenate(all_values), dtype=torch.float32).to(
            self.device
        )
        advantages = torch.tensor(
            np.concatenate(all_advantages), dtype=torch.float32
        ).to(self.device)
        returns = torch.tensor(np.concatenate(all_returns), dtype=torch.float32).to(
            self.device
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Sample mini-batches
        batch_size = min(
            64, states.shape[0]
        )  # Smaller batch size for potentially smaller dataset
        indices = BatchSampler(
            SubsetRandomSampler(range(states.shape[0])), batch_size, drop_last=False
        )

        epoch_metrics = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "kl_div": 0,
            "gradient_norm": 0,
        }

        num_batches = 0

        # Update in mini-batches
        for batch_indices in indices:
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_old_values = old_values[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Compute actor and critic losses
            actor_total_loss, policy_loss, entropy, kl_div = self.actor_loss(
                batch_states, batch_actions, batch_old_log_probs, batch_advantages
            )
            value_loss = self.critic_loss(batch_states, batch_old_values, batch_returns)

            # Combined loss
            total_loss = actor_total_loss + self.value_coef * value_loss

            # Optimization step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Calculate gradient norm
            total_norm = 0.0
            for p in list(self.actor.parameters()) + list(self.critic.parameters()):
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5

            # Clip gradients to prevent large updates
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Accumulate metrics
            epoch_metrics["total_loss"] += total_loss.item()
            epoch_metrics["policy_loss"] += policy_loss.item()
            epoch_metrics["value_loss"] += value_loss.item()
            epoch_metrics["entropy"] += entropy.item()
            epoch_metrics["kl_div"] += kl_div.item()
            epoch_metrics["gradient_norm"] += total_norm

            num_batches += 1

        # Average metrics over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, num_batches)

        return epoch_metrics

    def train(self, num_episodes: int):
        """Main training loop with trajectory-based rewards"""
        self.reward_net.eval()  # Set reward model to evaluation mode
        total_episodes = 0
        total_epoch_metrics = []

        # Main training loop
        for iteration in range(num_episodes):
            print(f"Starting iteration {iteration + 1}/{num_episodes}")

            # Collect multiple complete trajectories for this iteration
            trajectories_this_iteration = []
            episodes_this_iteration = 0

            # Collect a certain number of episodes/trajectories for this iteration
            while episodes_this_iteration < 5:  # Collect 5 trajectories per iteration
                # New trajectory
                current_trajectory = Trajectory()

                # Reset environment
                state, _ = self.env.reset()
                episode_length = 0
                episode_reward = 0
                done = False

                # Collect a single trajectory
                while not done and episode_length < self.max_steps_per_episode:
                    # Get action from policy
                    action, log_prob, value = self.get_action(state)

                    # Take action in environment
                    next_state, env_reward, terminated, truncated, _ = self.env.step(
                        action
                    )
                    done = terminated or truncated

                    # Store step in current trajectory
                    # Just store env_reward temporarily, we'll update with the trajectory reward later
                    current_trajectory.add_step(
                        state, action, log_prob, value, env_reward, done
                    )

                    # Update for next step
                    state = next_state
                    episode_reward += env_reward
                    episode_length += 1

                # Get value of last state for bootstrapping if not done
                last_value = 0
                if not done:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(
                            self.device
                        )
                        last_value = self.critic(state_tensor).cpu().item()

                # Finalize trajectory with computed returns and advantages
                current_trajectory.finalize(self.gamma, last_value)

                # Evaluate entire trajectory with reward model
                if self.reward_net is not None:
                    trajectory_reward = self.evaluate_trajectory(current_trajectory)

                    # Scale the episode reward to distribute over steps
                    if episode_length > 0:
                        step_reward = trajectory_reward / episode_length

                        # Replace all step rewards with the trajectory-based reward
                        current_trajectory.rewards = (
                            torch.ones_like(torch.from_numpy(current_trajectory.rewards)) * step_reward
                        )

                        # Recalculate returns and advantages with new rewards
                        current_trajectory.finalize(self.gamma, last_value)

                # Store the complete trajectory in memory
                trajectories_this_iteration.append(current_trajectory)

                episodes_this_iteration += 1
                total_episodes += 1
                print(
                    f"Episode {total_episodes}, Length: {episode_length}, "
                    f"Env Reward: {episode_reward:.4f}, "
                    f"Trajectory Reward: {current_trajectory.trajectory_return:.4f}"
                )

            # Now store all trajectories in memory
            for trajectory in trajectories_this_iteration:
                self.memory.store_trajectory(trajectory)

            # Get all trajectories for training
            trajectories = self.memory.get_memory()

            # Perform multiple epochs of updates
            for epoch in range(self.n_epochs):
                # Update policy with all trajectories
                epoch_metrics = self.update_policy(trajectories)
                total_epoch_metrics.append(epoch_metrics)

                # Log metrics
                print(
                    f"Iteration {iteration + 1}, Epoch {epoch + 1}/{self.n_epochs}: "
                    f"Loss: {epoch_metrics['total_loss']:.4f}, "
                    f"Policy Loss: {epoch_metrics['policy_loss']:.4f}, "
                    f"Value Loss: {epoch_metrics['value_loss']:.4f}, "
                    f"Entropy: {epoch_metrics['entropy']:.4f}, "
                    f"KL Div: {epoch_metrics['kl_div']:.4f}, "
                    f"gradient_norm: {epoch_metrics['gradient_norm']:.4f}"
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
            axes[i].plot(
                values, marker="o", linestyle="-"
            )  # Plot with markers and lines
            axes[i].set_title(f"{metric} over Steps")
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(metric)

        plt.tight_layout()  # Adjust spacing
        plt.show()

        print(f"Training completed. Total episodes: {total_episodes}")
        return self.actor, self.critic, self.reward_net
