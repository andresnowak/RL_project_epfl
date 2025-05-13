import gymnasium as gym
import numpy as np
import torch
import pandas as pd  # Import pandas
import os
from typing import Dict, List, Tuple, Any


def softmax(r1: np.float32, r2: np.float32) -> np.float32:
    """Bradley Terry model specifically for PPO"""
    # reward for best trajectory model, reward for the partial trajectory model
    # this returns the probability of the best model being chosen
    return np.exp(r1) / (np.exp(r1) + np.exp(r2))


def _flatten_data(data: Any, prefix: str) -> Dict[str, Any]:
    """Flattens numpy arrays or scalars into a dict for DataFrame rows."""
    if isinstance(data, np.ndarray):
        # Flatten and prefix numpy arrays
        return {f"{prefix}_{i}": val for i, val in enumerate(data.flatten())}
    elif isinstance(data, torch.Tensor):
        return {
            f"{prefix}_{i}": val.item() if val.numel() == 1 else val.cpu().numpy()
            for i, val in enumerate(data.flatten())
        }
    elif np.isscalar(data):
        # Keep scalars as is with a single key
        return {prefix: data}
    else:
        # Attempt to convert other types (like tensors detached from graph)
        try:
            if hasattr(data, "item"):  # Check if it's a tensor-like scalar
                return {prefix: data.item()}
            elif hasattr(data, "tolist"):  # Check if it's tensor-like array/list
                flattened = np.array(data).flatten()
                return {f"{prefix}_{i}": val for i, val in enumerate(flattened)}
            else:
                # Fallback for unexpected types
                return {prefix: data}
        except Exception:
            # Handle cases where conversion fails
            return {prefix: str(data)}  # Store as string if unsure


def collect_paired_demonstrations_cartpole(
    partial_model,
    full_model,
    env: gym.Env,
    output_dir: str,
    num_episodes: int = 50,
    deterministic: bool = True,
):
    """
    Collects paired demonstrations using Pandas and saves trajectories to CSV.

    Args:
        partial_model: The halfway trained model (50% performance)
        full_model: The fully trained model (100% performance)
        env: Gym environment instance
        output_dir: Directory to save the collected data
        num_episodes: Number of episodes to collect per model (number of trajectories)
        deterministic: Whether to use deterministic actions
    """
    os.makedirs(output_dir, exist_ok=True)
    model_names = ["partial", "full"]
    models = [partial_model, full_model]
    all_trajectory_summaries = {name: [] for name in model_names}

    # --- Step 1: Collect trajectories and save using Pandas ---

    trajectories_df = {}

    for model_name, model in zip(model_names, models):
        csv_path = os.path.join(output_dir, f"{model_name}_model_trajectories_CartPole-v1.csv")
        print(f"\nCollecting data from {model_name} model...")

        all_steps_data = []  # Accumulate all steps for this model
        global_episode_id_counter = 0

        for episode in range(num_episodes):
            current_episode_id = global_episode_id_counter
            obs, _ = env.reset()
            done = False
            episode_rewards = 0
            step_in_episode = 0
            episode_steps_data = []  # Store steps for *this* episode temporarily

            while not done:
                # Get action from model
                # with torch.no_grad():
                #     th_obs = torch.as_tensor(np.expand_dims(obs, axis=0)).to(
                #         model.device
                #     )
                obs = torch.tensor(obs, dtype=torch.float)
                dist = model(obs)
                action = dist.sample()
                action = torch.squeeze(action).item()
                # action, _ = model.predict(th_obs, deterministic=deterministic)

                # Step environment
                # SB3 predict usually returns numpy action.
                if isinstance(action, np.ndarray) and action.shape == (1,):  # scalar action in array
                    action_to_step = action.item()
                elif isinstance(action, np.ndarray):  # vector action
                    action_to_step = action.flatten()
                else:  # scalar action
                    action_to_step = action

                try:
                    next_obs, reward, terminated, truncated, info = env.step(action_to_step)
                    # Handle potential proxy objects if using wrappers like RecordVideo
                    if isinstance(reward, np.ndarray) and reward.size == 1:
                        reward = reward.item()
                    if isinstance(terminated, np.ndarray) and terminated.size == 1:
                        terminated = terminated.item()
                    if isinstance(truncated, np.ndarray) and truncated.size == 1:
                        truncated = truncated.item()

                except Exception as e:
                    print(f"Error during env.step: {e}")
                    print(f"Action provided: {action_to_step} (type: {type(action_to_step)})")
                    # Decide how to handle the error, e.g., skip episode or raise
                    # For now, let's break the inner loop and maybe skip this episode
                    done = True  # Mark as done to exit loop
                    reward = 0  # Avoid issues with undefined reward
                    next_obs = obs  # Use current obs as next_obs to avoid error
                    terminated = True
                    truncated = False  # Or True based on logic
                    print("Skipping rest of episode due to step error.")
                    continue  # Skip adding this problematic step

                done = terminated or truncated
                # done = terminated
                episode_rewards += reward
                step_in_episode += 1

                # Prepare step data dictionary
                step_data = {
                    "episode_id": current_episode_id,
                    "step": step_in_episode,
                    "reward": reward,
                    "done": int(done),
                }
                # Flatten observations and actions
                step_data.update(_flatten_data(obs, "obs"))
                step_data.update(_flatten_data(action, "action"))  # Use original action from predict
                step_data.update(_flatten_data(next_obs, "next_obs"))

                episode_steps_data.append(step_data)
                obs = next_obs

            # --- After episode finishes ---
            print(f"Model '{model_name}' - Episode {episode + 1}/{num_episodes}: Return={episode_rewards:.2f}, Length={step_in_episode}")

            # Add episode return to each step in the episode's data
            for step_data in episode_steps_data:
                step_data["return"] = episode_rewards

            # Add this episode's steps to the main list for this model
            all_steps_data.extend(episode_steps_data)

            # Store summary for preference pairing
            all_trajectory_summaries[model_name].append(
                {
                    "episode_id": current_episode_id,
                    "return": episode_rewards,
                    "length": step_in_episode,
                }
            )
            global_episode_id_counter += 1  # Increment global counter

        # --- After all episodes for a model ---
        if not all_steps_data:
            print(f"Warning: No data collected for model {model_name}. Skipping CSV save.")
            continue

        # Convert collected steps to DataFrame
        trajectory_df = pd.DataFrame(all_steps_data)

        # Reorder columns for better readability (optional but recommended)
        cols = ["episode_id", "step"]
        obs_cols = sorted([col for col in trajectory_df.columns if col.startswith("obs_")])
        action_cols = sorted([col for col in trajectory_df.columns if col.startswith("action")])  # handle action scalar/vector
        next_obs_cols = sorted([col for col in trajectory_df.columns if col.startswith("next_obs_")])
        other_cols = ["reward", "done", "return"]
        trajectory_df = trajectory_df[cols + obs_cols + action_cols + other_cols[:2] + next_obs_cols + other_cols[2:]]

        trajectories_df[model_name] = trajectory_df

        # Save to CSV
        trajectory_df.to_csv(csv_path, index=False)
        print(f"Saved trajectories for {model_name} model to {csv_path}")

    # --- Step 2: Create preference pairs ---
    # Pass the collected summaries to the pairing function
    pairs_df = create_paired_demonstrations(trajectories_df)
    pairs_df.to_csv(os.path.join(output_dir, "preference_pairs_CartPole-v1.csv"))
    print("Saved preference pairs")


def collect_paired_demonstrations_mountaincar(
    partial_model,
    full_model,
    env: gym.Env,
    output_dir: str,
    num_episodes: int = 50,
    deterministic: bool = True,
):
    """
    Collects paired demonstrations using Pandas and saves trajectories to CSV.

    Args:
        partial_model: The halfway trained model (50% performance)
        full_model: The fully trained model (100% performance)
        env: Gym environment instance
        output_dir: Directory to save the collected data
        num_episodes: Number of episodes to collect per model (number of trajectories)
        deterministic: Whether to use deterministic actions
    """
    os.makedirs(output_dir, exist_ok=True)
    model_names = ["partial", "full"]
    models = [partial_model, full_model]
    all_trajectory_summaries = {name: [] for name in model_names}

    # --- Step 1: Collect trajectories and save using Pandas ---

    trajectories_df = {}

    for model_name, model in zip(model_names, models):
        csv_path = os.path.join(output_dir, f"{model_name}_model_trajectories_{env.spec.id}.csv")
        print(f"\nCollecting data from {model_name} model...")

        all_steps_data = []  # Accumulate all steps for this model
        global_episode_id_counter = 0

        for episode in range(num_episodes):
            current_episode_id = global_episode_id_counter
            obs, _ = env.reset()
            done = False
            episode_rewards = 0
            step_in_episode = 0
            episode_steps_data = []  # Store steps for *this* episode temporarily

            while not done:
                # Get action from model
                # with torch.no_grad():
                #     th_obs = torch.as_tensor(np.expand_dims(obs, axis=0)).to(
                #         model.device
                #     )
                obs = torch.tensor(obs, dtype=torch.float)
                dist = model(obs)
                action = dist.sample()
                action = torch.squeeze(action).item()

                # Step environment
                # SB3 predict usually returns numpy action.
                if isinstance(action, np.ndarray) and action.shape == (1,):  # scalar action in array
                    action_to_step = action.item()
                elif isinstance(action, np.ndarray):  # vector action
                    action_to_step = action.flatten()
                else:  # scalar action
                    action_to_step = action

                try:
                    next_obs, reward, terminated, truncated, info = env.step(action_to_step)
                    # Handle potential proxy objects if using wrappers like RecordVideo
                    if isinstance(reward, np.ndarray) and reward.size == 1:
                        reward = reward.item()
                    if isinstance(terminated, np.ndarray) and terminated.size == 1:
                        terminated = terminated.item()
                    if isinstance(truncated, np.ndarray) and truncated.size == 1:
                        truncated = truncated.item()

                except Exception as e:
                    print(f"Error during env.step: {e}")
                    print(f"Action provided: {action_to_step} (type: {type(action_to_step)})")
                    # Decide how to handle the error, e.g., skip episode or raise
                    # For now, let's break the inner loop and maybe skip this episode
                    done = True  # Mark as done to exit loop
                    reward = 0  # Avoid issues with undefined reward
                    next_obs = obs  # Use current obs as next_obs to avoid error
                    terminated = True
                    truncated = False  # Or True based on logic
                    print("Skipping rest of episode due to step error.")
                    continue  # Skip adding this problematic step

                done = terminated or truncated
                episode_rewards += reward
                step_in_episode += 1

                # Prepare step data dictionary
                step_data = {
                    "episode_id": current_episode_id,
                    "step": step_in_episode,
                    "reward": reward,
                    "done": int(done),
                }
                # Flatten observations and actions
                step_data.update(_flatten_data(obs, "obs"))
                step_data.update(_flatten_data(action, "action"))  # Use original action from predict
                step_data.update(_flatten_data(next_obs, "next_obs"))

                episode_steps_data.append(step_data)
                obs = next_obs

            # --- After episode finishes ---
            print(f"Model '{model_name}' - Episode {episode + 1}/{num_episodes}: Return={episode_rewards:.2f}, Length={step_in_episode}")

            # Add episode return to each step in the episode's data
            for step_data in episode_steps_data:
                step_data["return"] = episode_rewards

            # Add this episode's steps to the main list for this model
            all_steps_data.extend(episode_steps_data)

            # Store summary for preference pairing
            all_trajectory_summaries[model_name].append(
                {
                    "episode_id": current_episode_id,
                    "return": episode_rewards,
                    "length": step_in_episode,
                }
            )
            global_episode_id_counter += 1  # Increment global counter

        # --- After all episodes for a model ---
        if not all_steps_data:
            print(f"Warning: No data collected for model {model_name}. Skipping CSV save.")
            continue

        # Convert collected steps to DataFrame
        trajectory_df = pd.DataFrame(all_steps_data)

        # Reorder columns for better readability (optional but recommended)
        cols = ["episode_id", "step"]
        obs_cols = sorted([col for col in trajectory_df.columns if col.startswith("obs_")])
        action_cols = sorted([col for col in trajectory_df.columns if col.startswith("action")])  # handle action scalar/vector
        next_obs_cols = sorted([col for col in trajectory_df.columns if col.startswith("next_obs_")])
        other_cols = ["reward", "done", "return"]
        trajectory_df = trajectory_df[cols + obs_cols + action_cols + other_cols[:2] + next_obs_cols + other_cols[2:]]

        trajectories_df[model_name] = trajectory_df

        # Save to CSV
        trajectory_df.to_csv(csv_path, index=False)
        print(f"Saved trajectories for {model_name} model to {csv_path}")

    # --- Step 2: Create preference pairs ---
    # Pass the collected summaries to the pairing function
    pairs_df = create_paired_demonstrations(trajectories_df)
    pairs_df.to_csv(os.path.join(output_dir, f"preference_pairs_{env.spec.id}.csv"))
    print("Saved preference pairs")


def create_paired_demonstrations(trajectories_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names = list(trajectories_df.keys())

    unique_ids = trajectories_df[model_names[0]]["episode_id"].unique()

    # return is the same in each part, it is the total reward obtained in that trajectory
    chosen_models = []
    for id in unique_ids:
        first_model_df = trajectories_df[model_names[0]]
        second_model_df = trajectories_df[model_names[1]]

        first_last_reward = first_model_df[first_model_df["episode_id"] == id][["return"]].iloc[-1]
        second_last_reward = second_model_df[second_model_df["episode_id"] == id][["return"]].iloc[-1]

        chosen_model = np.random.binomial(1, softmax(first_last_reward, second_last_reward))[0]
        chosen_models.append(chosen_model)

    pairs_df = {"traj1_id": unique_ids, "traj2_id": unique_ids, "chosen_one": chosen_models}
    pairs_df = pd.DataFrame(pairs_df)

    return pairs_df
