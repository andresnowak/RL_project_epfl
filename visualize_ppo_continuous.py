import gymnasium as gym
from stable_baselines3 import PPO
import torch
import argparse
import time

from src.generate_demonstrations import collect_paired_demonstrations
from src.create_env import create_env_continuous


ENV_ID = "MountainCarContinuous-v0"
num_episodes = 1
sleep_duration = 0.01

device = "cpu"


def visualize_model(env: gym.Env, model):
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not terminated and not truncated:
            action, _states = model.predict(obs)

            # Take action in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Render is often handled automatically by env.step() when render_mode="human"
            # but you can uncomment the line below if needed for specific envs/wrappers
            # env.render()

            total_reward += reward
            steps += 1

            # Add a small delay to make visualization easier to follow
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            if terminated or truncated:
                print(f"Episode finished after {steps} steps.")
                print(f"Total Reward: {total_reward:.2f}")
                if terminated:
                    print("Reason: Agent reached a terminal state.")
                if truncated:
                    print("Reason: Episode truncated (e.g., time limit reached).")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=ENV_ID,
        help="Gym environment ID (e.g., MountainCarContinuous-v0)",
    )

    args = parser.parse_args()

    model_path = args.model
    env_id = args.env_id

    print(f"Model loaded from {model_path}")

    # Create environment
    env = gym.make(env_id, render_mode="human")

    model = PPO.load(model_path)

    visualize_model(env, model)

    print("\nClosing environment.")
    env.close()
    print("Visualization finished.")
