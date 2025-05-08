import gymnasium as gym
import numpy as np
import torch
import pandas as pd  # Import pandas
import os
from typing import Dict, List, Tuple, Any
from stable_baselines3.common.base_class import BaseAlgorithm


def _flatten_data(data: Any, prefix: str) -> Dict[str, Any]:
    """Flattens numpy arrays or scalars into a dict for DataFrame rows."""
    if isinstance(data, np.ndarray):
        # Flatten and prefix numpy arrays
        return {f"{prefix}_{i}": val for i, val in enumerate(data.flatten())}
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


def collect_paired_demonstrations(
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
        num_episodes: Number of episodes to collect per model
        deterministic: Whether to use deterministic actions
    """
    os.makedirs(output_dir, exist_ok=True)
    model_names = ["partial", "full"]
    models = [partial_model, full_model]
    all_trajectory_summaries = {name: [] for name in model_names}

    # --- Step 1: Collect trajectories and save using Pandas ---
    global_episode_id_counter = (
        0  # Use a global counter for unique episode IDs across models
    )

    for model_name, model in zip(model_names, models):
        csv_path = os.path.join(output_dir, f"{model_name}_model_trajectories.csv")
        print(f"\nCollecting data from {model_name} model...")

        all_steps_data = []  # Accumulate all steps for this model

        for episode in range(num_episodes):
            current_episode_id = global_episode_id_counter
            obs, _ = env.reset()
            done = False
            episode_rewards = 0
            step_in_episode = 0
            episode_steps_data = []  # Store steps for *this* episode temporarily

            while not done:
                # Get action from model
                with torch.no_grad():
                    th_obs = torch.as_tensor(np.expand_dims(obs, axis=0)).to(
                        model.device
                    )

                    action, _ = model.predict(th_obs, deterministic=deterministic)

                    # Get log probability if available
                    log_prob = None
                    # Need to handle potential dict observations/actions for evaluate_actions
                    try:
                        if hasattr(model, "policy") and hasattr(
                            model.policy, "evaluate_actions"
                        ):
                            th_action = torch.as_tensor(action).to(model.device)

                            # Note: evaluate_actions might need specific formatting depending on the policy type
                            # This assumes a standard SB3 policy structure. Adjust if needed.
                            _, log_prob_tensor, _ = model.policy.evaluate_actions(
                                th_obs, th_action
                            )
                            log_prob = (
                                log_prob_tensor.cpu().numpy().item()
                            )  # Assuming single value log_prob
                    except Exception as e:
                        # print(f"Warning: Could not get log_prob. Error: {e}")
                        log_prob = None  # Assign None if log_prob calculation fails

                # Step environment
                # SB3 predict usually returns numpy action.
                if isinstance(action, np.ndarray) and action.shape == (
                    1,
                ):  # scalar action in array
                    action_to_step = action.item()
                elif isinstance(action, np.ndarray):  # vector action
                    action_to_step = action.flatten()
                else:  # scalar action
                    action_to_step = action

                try:
                    next_obs, reward, terminated, truncated, info = env.step(
                        action_to_step
                    )
                    # Handle potential proxy objects if using wrappers like RecordVideo
                    if isinstance(reward, np.ndarray) and reward.size == 1:
                        reward = reward.item()
                    if isinstance(terminated, np.ndarray) and terminated.size == 1:
                        terminated = terminated.item()
                    if isinstance(truncated, np.ndarray) and truncated.size == 1:
                        truncated = truncated.item()

                except Exception as e:
                    print(f"Error during env.step: {e}")
                    print(
                        f"Action provided: {action_to_step} (type: {type(action_to_step)})"
                    )
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
                    "log_prob": log_prob
                    if log_prob is not None
                    else np.nan,  # Use NaN for missing log_prob
                }
                # Flatten observations and actions
                step_data.update(_flatten_data(obs, "obs"))
                step_data.update(
                    _flatten_data(action, "action")
                )  # Use original action from predict
                step_data.update(_flatten_data(next_obs, "next_obs"))

                episode_steps_data.append(step_data)
                obs = next_obs

            # --- After episode finishes ---
            print(
                f"Model '{model_name}' - Episode {episode + 1}/{num_episodes}: Return={episode_rewards:.2f}, Length={step_in_episode}"
            )

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
            print(
                f"Warning: No data collected for model {model_name}. Skipping CSV save."
            )
            continue

        # Convert collected steps to DataFrame
        trajectory_df = pd.DataFrame(all_steps_data)

        # Reorder columns for better readability (optional but recommended)
        cols = ["episode_id", "step"]
        obs_cols = sorted(
            [col for col in trajectory_df.columns if col.startswith("obs_")]
        )
        action_cols = sorted(
            [col for col in trajectory_df.columns if col.startswith("action")]
        )  # handle action scalar/vector
        next_obs_cols = sorted(
            [col for col in trajectory_df.columns if col.startswith("next_obs_")]
        )
        other_cols = ["reward", "done", "log_prob", "return"]
        trajectory_df = trajectory_df[
            cols
            + obs_cols
            + action_cols
            + other_cols[:2]
            + next_obs_cols
            + other_cols[2:]
        ]

        # Save to CSV
        trajectory_df.to_csv(csv_path, index=False)
        print(f"Saved trajectories for {model_name} model to {csv_path}")

    # --- Step 2: Create preference pairs ---
    # Pass the collected summaries to the pairing function
    create_preference_pairs(all_trajectory_summaries, output_dir)


def create_preference_pairs(
    trajectory_summaries: Dict[str, List[Dict]],
    output_dir: str,
    max_pairs_per_traj: int = 5,  # Limit pairs per preferred trajectory
):
    """
    Creates preference pairs using Pandas based on trajectory summaries.

    Args:
        trajectory_summaries: Dict containing lists of episode summaries
                              (episode_id, return, length) for each model.
        output_dir: Directory to save the preference pairs CSV.
        max_pairs_per_traj: Maximum number of lower-return trajectories to pair
                            with each higher-return trajectory.
    """
    all_summaries_list = []
    for model_name, summaries in trajectory_summaries.items():
        for summary in summaries:
            all_summaries_list.append(
                {
                    "model": model_name,
                    "episode_id": summary["episode_id"],
                    "return": summary["return"],
                    "length": summary["length"],
                }
            )

    if not all_summaries_list:
        print("No trajectory summaries found. Cannot create preference pairs.")
        return

    # Create a DataFrame from the summaries
    summary_df = pd.DataFrame(all_summaries_list)

    # Sort by return (higher is better)
    summary_df = summary_df.sort_values(by="return", ascending=False).reset_index(
        drop=True
    )

    preference_pairs = []
    num_trajectories = len(summary_df)

    # Iterate through sorted trajectories to create pairs
    for i in range(num_trajectories):
        preferred_traj = summary_df.iloc[i]
        pairs_count_for_this_traj = 0
        # Compare with subsequent (lower return) trajectories
        for j in range(i + 1, num_trajectories):
            if pairs_count_for_this_traj >= max_pairs_per_traj:
                break  # Stop if we've made enough pairs for this preferred trajectory

            rejected_traj = summary_df.iloc[j]

            # Ensure there's a positive return difference
            return_diff = preferred_traj["return"] - rejected_traj["return"]
            if return_diff > 0:  # Strictly prefer higher return
                preference_pairs.append(
                    {
                        "preferred_episode_id": preferred_traj["episode_id"],
                        "preferred_model": preferred_traj["model"],
                        "preferred_return": preferred_traj["return"],
                        "rejected_episode_id": rejected_traj["episode_id"],
                        "rejected_model": rejected_traj["model"],
                        "rejected_return": rejected_traj["return"],
                        "return_diff": return_diff,
                    }
                )
                pairs_count_for_this_traj += 1
            # If return_diff is 0, we could potentially use length as a tie-breaker,
            # but for simplicity, we only pair based on strictly better returns here.

    if not preference_pairs:
        print(
            "Could not create any preference pairs (e.g., all returns were equal or only one trajectory)."
        )
        return

    # Create DataFrame for pairs and save to CSV
    pairs_df = pd.DataFrame(preference_pairs)
    pairs_path = os.path.join(output_dir, "preference_pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)

    print(
        f"\nCreated {len(preference_pairs)} preference pairs based on trajectory returns."
    )
    print(f"Preference pairs saved to {pairs_path}")
