from agents.rlhf_agent import PPORLHFAgent
from utils import visualize_policy
import gymnasium as gym
import pandas as pd
from data.trajectory import Trajectory


def load_trajectoris(data_dir):
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


if __name__ == "__main__":
    preferences = load_trajectoris(
        data_dir="./data/ppo_mountain_car_rollouts/",
    )
    env = gym.make("MountainCarContinuous-v0")

    agent = PPORLHFAgent(env, device="cpu", is_continuous_env=True, seed=42)

    # t1 = Trajectory(
    #     states=[env.reset()[0], env.reset()[0]],  # Dummy trajectory for demonstration
    # )
    # t2 = Trajectory(
    #     states=[env.reset()[0]],  # Dummy trajectory for demonstration
    # )
    # prefereces = [(t1, t2)]

    agent.train(
        preferences=preferences,
        epochs=10,
        batch_size=2,
    )

    agent.evaluate(
        num_episodes=5,
    )

    # visualize_policy(
    #     env,
    #     agent.policy_net,
    #     num_episodes=5,
    # )
