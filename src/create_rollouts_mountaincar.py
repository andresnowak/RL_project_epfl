import gymnasium as gym
from agents.ppo_discrete import *
import torch
import argparse

from generate_demonstrations import collect_paired_demonstrations_cartpole, collect_paired_demonstrations_mountaincar

# from src.create_env import create_env_continuous

print("--- Running Example ---")

# Define parameters based on the user request
ENV_NAME = "MountainCar-v0"
MODEL_PATH = "checkpoints/best_actor_model_" + ENV_NAME  # Use Continuous version
PARTIAL_MODEL_PATH = "checkpoints/half_actor_model_" + ENV_NAME  # Use Continuous version
DIR_NAME = "rollouts"
NUM_EPISODES = 1000
DETERMINISTIC_ROLLOUT = False  # Use stochastic actions for variety


env = gym.make(ENV_NAME)
model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
model.load_state_dict(torch.load(MODEL_PATH))
partial_model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
partial_model.load_state_dict(torch.load(PARTIAL_MODEL_PATH))


print(f"Model loaded from {MODEL_PATH}")
print(f"Partial model loaded form {PARTIAL_MODEL_PATH}")


# NOTE: we use the original reward

print("\nCalling generate_and_save_rollouts...")

collect_paired_demonstrations_mountaincar(
    partial_model=partial_model,
    full_model=model,
    env=env,
    output_dir=DIR_NAME,
    num_episodes=NUM_EPISODES,
)

env.close()
