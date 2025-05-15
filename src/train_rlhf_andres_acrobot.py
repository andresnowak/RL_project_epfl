from agents.rlhf_agent import PPORLHFAgent
# from utils import visualize_policy
import gymnasium as gym
import pandas as pd
from data.trajectory import Trajectory
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os

ENV_NAME = "Acrobot-v1"


def load_trajectories(data_dir):
    trajectories_1_df = pd.read_csv(
        os.path.join(data_dir, f"full_model_trajectories_{ENV_NAME}.csv")
    )
    trajectories_2_df = pd.read_csv(
        os.path.join(data_dir, f"partial_model_trajectories_{ENV_NAME}.csv")
    )
    preferences_df = pd.read_csv(
        os.path.join(data_dir, f"preference_pairs_{ENV_NAME}.csv")
    )

    obs_columns = [col for col in trajectories_1_df.columns if col.startswith("obs_")]

    preferences = []
    for i in range(len(preferences_df)):
        traj_id1 = preferences_df.iloc[i]["traj1_id"]
        traj_id2 = preferences_df.iloc[i]["traj2_id"]
    
        states_1 = trajectories_1_df[trajectories_1_df.episode_id == traj_id1][
            obs_columns
        ].values
        states_2 = trajectories_2_df[trajectories_2_df.episode_id == traj_id2][
            obs_columns
        ].values
        actions_1 = trajectories_1_df[trajectories_1_df.episode_id == traj_id1][
            ["action"]
        ].values
        actions_2 = trajectories_2_df[trajectories_2_df.episode_id == traj_id2][
            ["action"]
        ].values
        t1 = Trajectory(states=states_1, actions=actions_1)
        t2 = Trajectory(states=states_2, actions=actions_2)

        if preferences_df.iloc[i]["chosen_one"] == 1:
            t1, t2 = t2, t1
        preferences.append((t1, t2))

    return preferences


def visualize_policy(policy_model, env, num_episodes=5, seed=42):
    """
    Visualizes the policy by running a few episodes in the environment.
    """
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            dist = policy_model(state_tensor)
            action = torch.squeeze(dist.sample()).item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            env.render()
    env.close()

def evaluate_policy(agent, env, num_episodes=10):
    """Evaluate policy by running episodes and returning mean reward"""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)
                action = agent.actor(state_tensor).sample().cpu().numpy()

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    mean_reward = sum(total_rewards) / num_episodes
    print(f"Mean Episode Reward: {mean_reward:.2f}")
    return mean_reward


def evaluate_reward_model(agent: PPORLHFAgent, preferences):
    """Evaluate reward model accuracy on preference pairs with variable-length trajectories"""
    correct = 0
    total = 0

    for traj1, traj2 in preferences:
        # Convert to tensors and move to device
        states1 = traj1.states.to(agent.device)
        actions1 = traj1.actions.to(agent.device)
        states2 = traj2.states.to(agent.device)
        actions2 = traj2.actions.to(agent.device)

        # Compute rewards (handle variable lengths)
        with torch.no_grad():
            # Option 1: Sum of rewards (for variable lengths)
            # r1 = agent.reward_net(
            #     states1, actions1
            # ).sum()  # Sum rewards along trajectory
            # r2 = agent.reward_net(states2, actions2).sum()

            # Option 2: Average reward (normalizes for length)
            r1 = agent.reward_net(states1, actions1).mean()
            r2 = agent.reward_net(states2, actions2).mean()

        # Compare rewards (assuming traj1 is the preferred one in the pair)
        if (r1 > r2).item():
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Reward Model Accuracy: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    # Load data and initialize
    preferences = load_trajectories("../rollouts")

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
    agent.train_reward_model(train_prefs, val_prefs, n_epochs=100, batch_size=128, eval_callback=evaluate_reward_model)
    evaluate_reward_model(agent, val_prefs)  # Evaluate on held-out validation set
    # exit()

    agent.train(100)

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