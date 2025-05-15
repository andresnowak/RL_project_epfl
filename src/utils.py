import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from data.trajectory import Trajectory


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def select_models_by_fraction(
    score_history,
    actor_model_history,
    critic_model_history,
    performance_fraction=0.5,
    window_size=100,
):
    # Smooth the score history
    avg_scores = np.array(
        [
            np.mean(score_history[max(0, i - window_size) : i + 1])
            for i in range(len(score_history))
        ]
    )

    # Best and worst scores
    # we grab the last model that had the best score, just because it is more probable that model is better than the first best model
    max_value = np.max(avg_scores)
    # Reverse search for max value
    best_idx = len(avg_scores) - 1 - np.argmax(avg_scores[::-1])
    best_score = avg_scores[best_idx]
    worst_score = np.min(avg_scores)
    print(best_score)

    # Compute target as a fraction between best and worst (normalized scale)
    target_score = worst_score + performance_fraction * (best_score - worst_score)
    print(target_score)

    # Exclude best model from candidates
    candidate_indices = np.delete(np.arange(len(avg_scores)), best_idx)
    candidate_scores = np.delete(avg_scores, best_idx)

    # Find model whose score is closest to the target
    secondary_relative_idx = np.argmin(np.abs(candidate_scores - target_score))
    secondary_idx = candidate_indices[secondary_relative_idx]

    return (
        actor_model_history[best_idx],
        critic_model_history[best_idx],
        actor_model_history[secondary_idx],
        critic_model_history[secondary_idx],
        best_idx,
        secondary_idx,
    )


# ------------ RLHF utils -----------

def load_trajectories(data_dir, size: int, seed: int, env_name: str):
    trajectories_1_df = pd.read_csv(
        os.path.join(data_dir, f"full_model_trajectories_{env_name}.csv")
    )
    trajectories_2_df = pd.read_csv(
        os.path.join(data_dir, f"partial_model_trajectories_{env_name}.csv")
    )
    preferences_df = pd.read_csv(
        os.path.join(data_dir, f"preference_pairs_{env_name}.csv")
    )

    rng = np.random.default_rng(seed)
    total = len(preferences_df)
    if size > total:
        raise ValueError(
            f"Requested size {size} exceeds total preference pairs {total}"
        )
    indices = rng.choice(total, size=size, replace=False)

    preferences_df = preferences_df.iloc[indices].reset_index(drop=True)

    trajectories_1_df = trajectories_1_df[
        trajectories_1_df["episode_id"].isin(preferences_df["traj1_id"])
    ].reset_index(drop=True)
    trajectories_2_df = trajectories_2_df[
        trajectories_2_df["episode_id"].isin(preferences_df["traj2_id"])
    ].reset_index(drop=True)

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


def evaluate_reward_model(agent, preferences):
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
            r1 = agent.reward_net(
                states1, actions1
            ).sum()  # Sum rewards along trajectory
            r2 = agent.reward_net(states2, actions2).sum()

            # Option 2: Average reward (normalizes for length)
            # r1 = agent.reward_net(states1, actions1).mean()
            # r2 = agent.reward_net(states2, actions2).mean()

        # Compare rewards (assuming traj1 is the preferred one in the pair)
        if (r1 > r2).item():
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Reward Model Accuracy: {accuracy:.2%}")
    return accuracy


def save_agent_metrics(metrics, size, seed, save_dir, model_type: str, env_name: str):
    metrics_keys = list(metrics[0].keys())
    num_metrics = len(metrics_keys)

    # Create subplots
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    # Plot each metric
    for i, metric in enumerate(metrics_keys):
        values = [entry[metric] for entry in metrics]
        axes[i].plot(values, marker="o", linestyle="-")  # Plot with markers and lines
        axes[i].set_title(f"{metric} over Steps")
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel(metric)

    plt.tight_layout()  # Adjust spacing

    os.makedirs(os.path.join(save_dir, "metrics"), exist_ok=True)
    plot_path = os.path.join(
        save_dir, f"metrics/{model_type}_metrics_plot_s{seed}_K{size}_{env_name}.png"
    )
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {plot_path}")
    # plt.show()