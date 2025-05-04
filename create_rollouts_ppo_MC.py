import gymnasium as gym
from RL_PPO.PPO_mountaincar import *
import torch
import argparse

from src.generate_demonstrations_MC import collect_paired_demonstrations

# from src.create_env import create_env_continuous

print("--- Running Example ---")

# Define parameters based on the user request
MODEL_PATH = "RL_PPO/model/best_actor_model_MC"  # Use Continuous version
PARTIAL_MODEL_PATH = "RL_PPO/model/half_actor_model_MC"  # Use Continuous version
CSV_FILE = "ppo_mountain_car_rollouts.csv"
DIR_NAME = "ppo_mountain_car_rollouts"
NUM_EPISODES = 100
DETERMINISTIC_ROLLOUT = False  # Use stochastic actions for variety


env = gym.make("MountainCar-v0")
model = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
model.load_models(MODEL_PATH)
partial_model = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
partial_model.load_models(PARTIAL_MODEL_PATH)


print(f"Model loaded from {MODEL_PATH}")
print(f"Partial model loaded form {PARTIAL_MODEL_PATH}")

# Create environment
env = gym.make("MountainCar-v0")

# NOTE: we use the original reward

print("\nCalling generate_and_save_rollouts...")

collect_paired_demonstrations(
    partial_model=partial_model,
    full_model=model,
    env=env,
    output_dir=DIR_NAME,
    num_episodes=NUM_EPISODES,
)

env.close()
