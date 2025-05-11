import gymnasium as gym
import torch
import argparse
import time
import numpy as np

from src.models.PPO_mountaincarContinuous import ActorNetwork
from src.models.helpers import TanhNormal

# ---- Config ----
ENV_ID = "MountainCarContinuous-v0"
NUM_EPISODES = 1
SLEEP_DURATION = 0.01
MODEL_PATH = "dpo_actor_seed0_K1000.pth"

device = torch.device("cpu")  # safer for MPS devices

def visualize_model(env: gym.Env, actor: ActorNetwork):
    for episode in range(NUM_EPISODES):
        print(f"\n🎬 Starting Episode {episode + 1}/{NUM_EPISODES}")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not terminated and not truncated:
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, sigma = actor(s)
                dist = TanhNormal(mu, sigma)
                action = dist.mean.cpu().numpy()[0]  # deterministic for viz

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            time.sleep(SLEEP_DURATION)

        print(f"🏁 Episode finished after {steps} steps. Total reward: {total_reward:.2f}")
        if terminated: print("Reason: Agent reached a terminal state.")
        if truncated:  print("Reason: Episode truncated (e.g., time limit).")

if __name__ == "__main__":
    print(f"📦 Loading model from {MODEL_PATH}")
    
    env = gym.make(ENV_ID, render_mode="human")
    n_actions = env.action_space.shape
    input_dims = env.observation_space.shape
    action_bound = float(env.action_space.high[0])

    actor = ActorNetwork(n_actions, input_dims, action_bound).to(device)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    actor.eval()

    visualize_model(env, actor)
    env.close()
    print("✅ Visualization finished.")
