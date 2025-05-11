import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import copy
from torch.utils.data import BatchSampler, SubsetRandomSampler
from models.ppo_policy import ActorNetwork, CriticNetwork
from models.reward import RewardModel

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
        env,
        device,
        actor_model_path: str,
        lr=0.001,
        gamma=0.99,
        clip_epsilon=0.2,
        beta_critic=0.01,
        beta_kl=0.01,
        lam=0.95,
        seed=42,
    ):
        super().__init__()
        self.device = device
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = ActorNetwork(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        ).to(device)

        # Load pretrained actor from "half" model
        self.actor.load_state_dict(
            torch.load(Path(actor_model_path), map_location=device)
        )

        # Frozen reference actor
        # Clone a nn.Module

        self.actor_ref = copy.deepcopy(self.actor)
        # self.actor_ref = self.actor.clone()
        self.actor_ref.eval()

        # Critic network for value function
        self.critic = CriticNetwork(
            state_dim=self.state_dim,
        )
        # TODO: load pretrained critic from "half" model
        # self.critic.load_state_dict(
        #     torch.load(Path(critic_model_path), map_location=device)
        # )
        self.critic.to(device)

        # Reward model
        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            hidden_dim=256,
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
        self.beta_critic = beta_critic
        self.beta_kl = beta_kl
        self.lam = lam
        self.seed = seed

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
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    def update_policy(self, memory, n_epochs):
        # import pdb

        # pdb.set_trace()
        states, actions, old_probs, values, rewards, dones = memory.get_memory()

        advantage = np.zeros(len(rewards))
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - int(dones[t])) * gae
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1 - int(dones[t])) * gae
            advantage[t] = gae

        # Normalize advantage
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
        advantage = torch.tensor(advantage).to(self.device)

        # for _ in range(n_epochs):
        loss = {
            "actor_loss": [],
            "critic_loss": [],
            "kl_loss": [],
            "total_loss": 0,
        }
        for epoch in tqdm(range(n_epochs), desc="Training Policy"):
            batches = memory.generate_batches()
            values = torch.tensor(values).to(self.device)

            for batch in batches:
                batch_states = torch.tensor(states[batch]).to(self.device)
                batch_actions = torch.tensor(actions[batch]).to(self.device)
                batch_old_probs = torch.tensor(old_probs[batch]).to(self.device)

                dist = self.actor(batch_states)
                critic_values = self.critic(batch_states).squeeze()

                batch_new_probs = dist.log_prob(batch_actions)

                prob_ratio = torch.exp(batch_new_probs - batch_old_probs)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(
                        prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    )
                    * advantage[batch]
                )

                # PPO loss
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_values) ** 2
                critic_loss = critic_loss.mean()

                # RLHF loss (KL divergence between actor and reference actor)
                dist_ref = self.actor_ref(batch_states)
                batch_ref_probs = dist_ref.log_prob(batch_actions)
                approx_kl = torch.mean(batch_new_probs - batch_ref_probs)

                total_loss = (
                    actor_loss
                    + self.beta_critic * critic_loss
                    - self.beta_kl * approx_kl
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                loss["actor_loss"].append(actor_loss.item())
                loss["critic_loss"].append(critic_loss.item())
                loss["kl_loss"].append(approx_kl.item())
                loss["total_loss"] = total_loss.item()

            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Actor Loss: {np.mean(loss['actor_loss']):.4f}, "
                f"Critic Loss: {np.mean(loss['critic_loss']):.4f}, "
                f"KL Loss: {np.mean(loss['kl_loss']):.4f}, "
                f"Total Loss: {loss['total_loss']:.4f}"
            )
            print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Actor Loss: {np.mean(loss['actor_loss']):.4f}, "
                f"Critic Loss: {np.mean(loss['critic_loss']):.4f}, "
                f"KL Loss: {np.mean(loss['kl_loss']):.4f}, "
                f"Total Loss: {loss['total_loss']:.4f}"
            )

    def train(
        self, preferences, reward_epochs, policy_epochs, batch_size, max_steps=100
    ):
        # Train the reward model
        self.train_reward_model(
            preferences=preferences,
            n_epochs=reward_epochs,
            batch_size=batch_size,
        )

        # Freeze the reward model
        self.reward_net.eval()

        # Train the policy (actor and critic)
        # for episode in range(policy_epochs):
        for episode in tqdm(range(policy_epochs), desc="Episodes for Policy Training"):
            state, _ = self.env.reset(seed=self.seed)
            done = False

            memory = PPOMemory(batch_size)
            for step in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                dist = self.actor(state_tensor)

                action = torch.squeeze(dist.sample())
                prob = torch.squeeze(dist.log_prob(action)).item()
                action = action.item()
                value = self.critic(state_tensor).item()

                next_state, _, done, _, _ = self.env.step(action)
                reward = self.reward_net(state_tensor).item()

                memory.store_memory(state, action, prob, value, reward, done)

                if done:
                    break

            self.update_policy(memory, policy_epochs)
            memory.clear_memory()
