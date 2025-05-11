import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from models.ppo_policy import ActorNetwork
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
        beta=0.01,
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
        )

        # Load pretrained actor from "half" model
        self.actor.load_state_dict(
            torch.load(Path(actor_model_path), map_location=device)
        )

        # Frozen reference actor
        self.actor_ref = self.actor.clone()
        self.actor_ref.eval()

        self.reward_net = RewardModel(
            state_dim=self.state_dim,
            hidden_dim=256,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr},
                {"params": self.reward_net.parameters(), "lr": lr},
            ]
        )
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
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
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
