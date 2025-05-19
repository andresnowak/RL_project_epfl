### plot each model performance with box plot
import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.ppo_policy import *

# Env
# ENV_NAME = "CartPole-v1"
ENV_NAME = "Acrobot-v1"

COLOR_CODE = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

# PATH
DIR = "../../checkpoints/"
MODEL_BEST_PPO_PATH = DIR + "best_actor_model_" + ENV_NAME
MODEL_HALF_PPO_PATH = DIR + "half_actor_model_" + ENV_NAME
MODEL_DPO_PATH = DIR + "DPO/s2_K2000_" + ENV_NAME + ".pth"
MODEL_RLHF_PPO_PATH = DIR + "RLHF_PPO/s2_K500_" + ENV_NAME + ".pth"

if __name__ == "__main__":

    env = gym.make(ENV_NAME)

    # Best model from PPO
    model_best_PPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    model_best_PPO.load_state_dict(torch.load(MODEL_BEST_PPO_PATH))
    # Half model from PPO
    model_half_PPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    model_half_PPO.load_state_dict(torch.load(MODEL_HALF_PPO_PATH))
    # Best DPO
    model_DPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    model_DPO.load_state_dict(torch.load(MODEL_DPO_PATH, map_location=torch.device("cpu")))
    # Best RLHF-PPO
    model_RLHF_PPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    temp_model = torch.load(MODEL_RLHF_PPO_PATH)
    model_RLHF_PPO.load_state_dict(temp_model["../RLHF_models/" + ENV_NAME + "/policy"])

    model_list = [model_half_PPO, model_best_PPO, model_DPO, model_RLHF_PPO]
    model_name_list = ["Reference Policy", "Best Policy", "DPO", "RLHF_PPO"]

    # Parameters
    n_episode = 10

    # for plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})

    # Run n_episode episodes for each model
    for i, (model, model_name) in enumerate(zip(model_list, model_name_list)):
        print("Start: ", model_name)
        model_score_list = []
        for _ in range(n_episode):
            state, _ = env.reset()
            done = False
            episode_score = 0
            while not done:
                action = model.choose_action(state)
                state_, reward, terminated, truncated, _ = env.step(action)
                state = state_
                episode_score += reward
                done = terminated or truncated
            model_score_list.append(episode_score)

        # boxplot
        plt.boxplot(
            model_score_list,
            positions=[i + 1],
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_CODE[i], color=COLOR_CODE[4]),  # Box face and edge
            medianprops=dict(color=COLOR_CODE[4], linewidth=2),  # Median line
            whiskerprops=dict(color=COLOR_CODE[4], linewidth=1.5),  # Whiskers
            capprops=dict(color=COLOR_CODE[4], linewidth=1.5),  # Caps
            flierprops=dict(marker="d", color=COLOR_CODE[i], markeredgecolor=COLOR_CODE[i]),  # Outliers
        )
        print("Mean: ", np.array(model_score_list).mean())

# Customizing the plot
plt.title(f"The model performance over {n_episode} episodes, {ENV_NAME}")
plt.xlabel("Models")
plt.ylabel("Rewards")
plt.xticks([i + 1 for i in range(len(model_name_list))], model_name_list)


# save plots
plt.savefig("../../plots/boxplots_" + ENV_NAME + ".pdf")
