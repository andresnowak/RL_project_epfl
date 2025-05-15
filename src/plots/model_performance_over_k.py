import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.ppo_policy import *

# Env
ENV_NAME = "CartPole-v1"
ENV_NAME = "Acrobot-v1"
COLOR_CODE = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

# PATH
DIR = "../../checkpoints/"

model_name_list = ["DPO", "RLHF_PPO"]
ks = [500, 1000, 2000, 3000]


if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    # Parameters
    n_episode = 100
    n_seed = 3

    # for plot
    plt.figure(figsize=(8, 6))

    # Run n_episode episodes for each model
    # DPO
    model_name = model_name_list[0]
    print("Start: ", model_name)
    model_score_list = []
    for k in ks:
        k_score_buffer = []
        for s in range(n_seed):
            model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
            model.load_state_dict(torch.load(DIR + model_name + "/s" + str(s) + "_K" + str(k) + "_" + ENV_NAME + ".pth", map_location=torch.device("cpu")))
            episode_score_buffer = []
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
                episode_score_buffer.append(episode_score)
            k_score_buffer.append(np.mean(episode_score_buffer))
        model_score_list.append(np.mean(k_score_buffer))

    plt.plot(np.arange(len(ks)), model_score_list, marker="o", linestyle="-", color=COLOR_CODE[0], label=model_name)

    # RHLF_PPO
    model_name = model_name_list[1]
    print("Start: ", model_name)
    model_score_list = []
    for k in ks:
        k_score_buffer = []
        for s in range(n_seed):
            model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
            temp_model = torch.load(DIR + model_name + "/s" + str(s) + "_K" + str(k) + "_" + ENV_NAME + ".pth")
            model.load_state_dict(temp_model["../RLHF_models/" + ENV_NAME + "/policy"])
            episode_score_buffer = []
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
                episode_score_buffer.append(episode_score)
            k_score_buffer.append(np.mean(episode_score_buffer))
        model_score_list.append(np.mean(k_score_buffer))

    plt.plot(np.arange(len(ks)), model_score_list, marker="o", linestyle="-", color=COLOR_CODE[2], label=model_name)

plt.xticks(np.arange(len(ks)), ks)

if ENV_NAME == "CartPole-v1":
    plt.ylim(300, 500)
elif ENV_NAME == "Acrobot-v1":
    plt.ylim(-140, -90)
plt.legend()
plt.xlabel("k")
plt.ylabel("Rewards")
plt.title(f"model performance over k: {ENV_NAME}")
plt.savefig("../../plots/model_performance_over_k_" + ENV_NAME + ".pdf")
