from agents.rlhf_agent_andres import PPORLHFAgent
# from utils import visualize_policy
import gymnasium as gym
import pandas as pd
from data.trajectory import Trajectory
import torch


def load_trajectories(data_dir):
    trajectories_1_df = pd.read_csv(
        data_dir
        + "/full_model_trajectories_Acrobot-v1.csv"
    )
    trajectories_2_df = pd.read_csv(
        data_dir + "/partial_model_trajectories_Acrobot-v1.csv"
    )
    preferences_df = pd.read_csv(
        data_dir + "/preference_pairs_Acrobot-v1.csv"
    )

    preferences = []
    for i in range(len(preferences_df)):
        traj_id1 = preferences_df.iloc[i]["traj1_id"]
        traj_id2 = preferences_df.iloc[i]["traj2_id"]
        states_1 = trajectories_1_df[trajectories_1_df.episode_id == traj_id1][
            ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "obs_5"]
        ].values
        states_2 = trajectories_2_df[trajectories_2_df.episode_id == traj_id2][
            ["obs_0", "obs_1", "obs_2", "obs_3", "obs_4", "obs_5"]
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
        data_dir="../data/ppo_acrobot_rollouts",
    )
    env_id = "Acrobot-v1"
    env = gym.make(env_id)

    agent = PPORLHFAgent(
        env,
        "cpu",
        "../checkpoints_2/half_actor_model_Acrobot-v1",
    )

    # t1 = Trajectory(
    #     states=[env.reset()[0], env.reset()[0]],  # Dummy trajectory for demonstration
    # )
    # t2 = Trajectory(
    #     states=[env.reset()[0]],  # Dummy trajectory for demonstration
    # )
    # prefereces = [(t1, t2)]

    # agent.evaluate(
    #     num_episodes=5,
    # )

    # visualize_policy(
    #     env,
    #     agent.policy_net,
    #     num_episodes=5,
    # )

    env = gym.make(env_id, render_mode="rgb_array")
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
        100,
        preferences=preferences,
    )


    env = gym.make(env_id, render_mode="rgb_array")
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
