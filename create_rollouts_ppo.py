import gymnasium as gym
from stable_baselines3 import PPO
import torch
import argparse

from src.generate_demonstrations import collect_paired_demonstrations
from src.create_env import create_env_continuous

print("--- Running Example ---")

# Define parameters based on the user request
MODEL_PATH = "./ppo_mountain_car_continuous/20250504_155941/checkpoints/checkpoint_180000" # Use Continuous version
MODEL_PATH_2 = "./ppo_mountain_car_continuous/20250504_155941/checkpoints/checkpoint_160000"  # Use Continuous version
ENV_ID = "MountainCarContinuous-v0"
CSV_FILE = "ppo_mountain_car_continuous_rollouts.csv"
DIR_NAME = "ppo_mountain_car_continuous_rollouts"
NUM_EPISODES = 5
DETERMINISTIC_ROLLOUT = False # Use stochastic actions for variety

device = "cpu" 
model = PPO.load(MODEL_PATH, device=device)
model_2 = PPO.load(MODEL_PATH, device=device)

print(f"Model loaded from {MODEL_PATH}")

# Create environment
env = create_env_continuous(ENV_ID)

# NOTE: we use the original reward

print("\nCalling generate_and_save_rollouts...")

collect_paired_demonstrations(
    model_2,
    model,
    env,
    DIR_NAME,
    5,
)

env.close()