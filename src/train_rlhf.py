from agents.rlhf_agent import PPORLHFAgent
from utils import visualize_policy
import gymnasium as gym
import pandas as pd
import torch
from data.trajectory import Trajectory


def load_trajectories(data_dir):
    trajectories_1_df = pd.read_csv(data_dir + "full_model_trajectories.csv")
    trajectories_2_df = pd.read_csv(data_dir + "partial_model_trajectories.csv")
    preferences_df = pd.read_csv(data_dir + "preference_pairs.csv")

    preferences = []
    for i in range(len(preferences_df)):
        traj_id1 = preferences_df.iloc[i]["traj1_id"]
        traj_id2 = preferences_df.iloc[i]["traj2_id"]
        states_1 = trajectories_1_df[trajectories_1_df.episode_id == traj_id1][
            ["obs_0", "obs_1"]
        ].values
        states_2 = trajectories_2_df[trajectories_2_df.episode_id == traj_id2][
            ["obs_0", "obs_1"]
        ].values
        t1 = Trajectory(states=states_1)
        t2 = Trajectory(states=states_2)
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


if __name__ == "__main__":
    preferences = load_trajectories(
        data_dir="./data/ppo_mountain_car_rollouts/",
    )

    env = gym.make("MountainCar-v0")
    agent = PPORLHFAgent(
        env,
        device="cpu",
        actor_model_path="/Users/iris/Documents/master@epfl/ma2/rl/rl_github/rl_2/checkpoints_mountain_car/half_actor_model_MC",
        seed=42,
    )

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos_before_training/",
        episode_trigger=lambda episode_id: True,  # record every episode
    )
    visualize_policy(
        policy_model=agent.actor,
        env=env,
        num_episodes=5,
    )

    agent.train(
        preferences=preferences,
        reward_epochs=10,
        policy_epochs=10,
        batch_size=2,
    )

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos_after_training/",
        episode_trigger=lambda episode_id: True,  # record every episode
    )
    visualize_policy(
        policy_model=agent.actor,
        env=env,
        num_episodes=5,
    )
