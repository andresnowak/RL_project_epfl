import gymnasium as gym
from src.models.PPO_discrete import *
import torch
import argparse

from src.generate_demonstrations_cartpole import collect_paired_demonstrations

# from src.create_env import create_env_continuous

print("--- Running Example ---")

# Define parameters based on the user request
MODEL_PATH = "checkpoints_cartpole/best_actor_model_CP"  # Use Continuous version
PARTIAL_MODEL_PATH = "checkpoints_cartpole/half_actor_model_CP"  # Use Continuous version
CSV_FILE = "ppo_cartpole_rollouts.csv"
DIR_NAME = "ppo_cartpole_rollouts"
NUM_EPISODES = 100
DETERMINISTIC_ROLLOUT = False  # Use stochastic actions for variety


env = gym.make("CartPole-v1")
model = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
model.load_models(MODEL_PATH)
partial_model = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
partial_model.load_models(PARTIAL_MODEL_PATH)


print(f"Model loaded from {MODEL_PATH}")
print(f"Partial model loaded form {PARTIAL_MODEL_PATH}")


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
