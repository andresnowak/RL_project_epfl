# from utils import visualize_policy
import gymnasium as gym
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

from data.trajectory import Trajectory
from agents.rlhf_agent import PPORLHFAgent

from utils import load_trajectories, evaluate_policy, visualize_policy, save_agent_metrics

ENV_NAME = "Acrobot-v1"
MODEL_SAVE_DIR = f"../RLHF_models/{ENV_NAME}"

if __name__ == "__main__":
    for size in [500, 1000, 2000, 3000]:
        print(f"Size of dataset: {size}")
        for seed in [0, 1, 2]:
            # Load data and initialize
            preferences = load_trajectories(
                f"../rollouts/{ENV_NAME}", size, seed, ENV_NAME
            )

            train_prefs, val_prefs = train_test_split(
                preferences, 
                test_size=0.2,
                random_state=42
            )

            env = gym.make(ENV_NAME)
            agent = PPORLHFAgent(env, "cpu", f"../checkpoints/half_actor_model_{ENV_NAME}", n_epochs=10, entropy_coef=0.001, lr_reward=1e-3)

            # ========== BEFORE TRAINING ==========
            print("\n=== Pre-Training Evaluation ===")
            
            # Evaluate policy
            pre_train_reward = evaluate_policy(agent, env, 50)
            
            # Visualize
            record_env = gym.make(ENV_NAME, render_mode="rgb_array")
            record_env = gym.wrappers.RecordVideo(record_env, f"videos_before_training_{ENV_NAME}/")
            visualize_policy(agent.actor, record_env, 3)

            # ========== TRAINING ==========
            print("\n=== Training ===")
            reward_metrics = agent.train_reward_model(train_prefs, val_prefs, n_epochs=100, batch_size=128)
            save_agent_metrics(
                reward_metrics, size, seed, MODEL_SAVE_DIR, "reward", ENV_NAME
            )

            actor, critic, reward_net, metrics = agent.train(100)
            save_agent_metrics(
                metrics, size, seed, MODEL_SAVE_DIR, "agent", ENV_NAME
            )

            # ========== AFTER TRAINING ==========
            print("\n=== Post-Training Evaluation ===")

            print("\nReward Model (After Training):")
            
            # Evaluate policy
            post_train_reward = evaluate_policy(agent, env, 50)
            
            # Visualize
            record_env = gym.make(ENV_NAME, render_mode="rgb_array")
            record_env = gym.wrappers.RecordVideo(record_env, f"videos_after_training_{ENV_NAME}/")
            visualize_policy(agent.actor, record_env, 3)

            print(f"\nImprovement: {post_train_reward - pre_train_reward:.2f}")

            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

            torch.save(
                {
                    os.path.join(MODEL_SAVE_DIR, "policy"): actor.state_dict(),
                    os.path.join(
                        MODEL_SAVE_DIR, "reward_model"
                    ): reward_net.state_dict(),
                },
                os.path.join(MODEL_SAVE_DIR, f"s{seed}_K{size}_{ENV_NAME}.pth"),
            )