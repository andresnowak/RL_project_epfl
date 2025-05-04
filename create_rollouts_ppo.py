import gymnasium as gym
from stable_baselines3 import PPO
import torch
import argparse

from src.generate_demonstrations import collect_paired_demonstrations
from src.create_env import create_env_continuous

print("--- Running Example ---")

# Define parameters based on the user request
MODEL_PATH = "./ppo_mountain_car_continuous/20250504_165641/checkpoints/checkpoint_200000" # Use Continuous version
PARTIAL_MODEL_PATH = "./ppo_mountain_car_continuous/20250504_165641/checkpoints/checkpoint_60000"  # Use Continuous version
ENV_ID = "MountainCarContinuous-v0"
CSV_FILE = "ppo_mountain_car_continuous_rollouts.csv"
DIR_NAME = "ppo_mountain_car_continuous_rollouts"
NUM_EPISODES = 5
DETERMINISTIC_ROLLOUT = False # Use stochastic actions for variety

device = "cpu" 
model = PPO.load(MODEL_PATH, device=device)
partial_model = PPO.load(PARTIAL_MODEL_PATH, device=device)

print(f"Model loaded from {MODEL_PATH}")
print(f"Partial model loaded form {PARTIAL_MODEL_PATH}")

# Create environment
env = create_env_continuous(ENV_ID)

# NOTE: we use the original reward

print("\nCalling generate_and_save_rollouts...")

collect_paired_demonstrations(
    partial_model=partial_model,
    full_model=model,
    env=env,
    output_dir=DIR_NAME,
    num_episodes=5,
)

env.close()