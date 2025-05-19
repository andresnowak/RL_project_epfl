import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.ppo_policy import *

# Env
ENV_NAME = "CartPole-v1"
ENV_NAME = "Acrobot-v1"
ENV_NAME = "LunarLander-v3"
COLOR_CODE = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

# PATH
DIR = "../../checkpoints/"

model_name_list = ["DPO", "RLHF_PPO"]
# model_name_list = ["RLHF_PPO"]
ks = [20, 30, 50, 100, 500, 1000, 2000, 3000]


class ActorNetwork_Lunar(nn.Module):
    def __init__(self, action_dim, state_dim, fc_dims=[256]):
        super(ActorNetwork_Lunar, self).__init__()

        layers = []
        input_dim = state_dim

        for fc_dim in fc_dims:
            layers.append(nn.Linear(input_dim, fc_dim))
            layers.append(nn.ReLU())
            input_dim = fc_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))

        self.actor = nn.Sequential(*layers)

    def forward(self, state):
        dist = self.actor(state) + 1e-8  # Add a small value so the ref frozen model probabilities don't go to zero
        dist = Categorical(dist)

        return dist

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        dist = self.forward(state)
        action = dist.sample()
        action = torch.squeeze(action).item()
        return action


if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    # Parameters
    n_episode = 100
    n_seed = 3

    # for plot
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})

    # Run n_episode episodes for each model
    # DPO
    model_name = model_name_list[0]
    print("Start: ", model_name)
    model_score_list = []
    for k in ks:
        k_score_buffer = []
        for s in range(n_seed):
            if ENV_NAME == "LunarLander-v3":
                model = ActorNetwork_Lunar(env.action_space.n, env.observation_space.shape[0], [256, 256, 256])
                model.load_state_dict(torch.load(DIR + model_name + "/s" + str(s) + "_K" + str(k) + "_" + ENV_NAME + ".pth", map_location=torch.device("cpu")))
            else:
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
            temp_model = torch.load(DIR + model_name + "/s" + str(s) + "_K" + str(k) + "_" + ENV_NAME + ".pth")

            # load model
            if ENV_NAME == "CartPole-v1":
                model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
                model.load_state_dict(temp_model["../RLHF_models/CartPole-v1/policy"])
            elif ENV_NAME == "Acrobot-v1":
                model = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
                model.load_state_dict(temp_model["policy"])
            elif ENV_NAME == "LunarLander-v3":
                model = ActorNetwork_Lunar(env.action_space.n, env.observation_space.shape[0], [256, 256, 256])
                model.load_state_dict(temp_model["../RLHF_models/LunarLander-v3/policy"])

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
        plt.ylim(0, 500)
    elif ENV_NAME == "Acrobot-v1":
        plt.ylim(-500, 0)
    elif ENV_NAME == "LunarLander-v3":
        # plt.ylim(-500, 0)
        pass
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Rewards")
    plt.title(f"model performance over k: {ENV_NAME}")
    plt.savefig("../../plots/model_performance_over_k_" + ENV_NAME + ".pdf")
