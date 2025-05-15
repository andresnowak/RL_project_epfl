### plot each model performance with box plot
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.ppo_policy import *

COLOR_CODE = ["#FF0000", "#B51F1F", "#00A79F", "#007480", "#413D3A", "#CAC7C7"]

if __name__ == "__main__":
    # Env
    ENV_NAME = "CartPole-v1"
    # ENV_NAME = "Acrobot-v1"
    env = gym.make(ENV_NAME)

    # PATH
    DIR = "../../checkpoints/"
    MODEL_BEST_PPO_PATH = DIR + "best_actor_model_" + ENV_NAME
    MODEL_HALF_PPO_PATH = DIR + "half_actor_model_" + ENV_NAME
    # MODEL_DPO_PATH =
    # MODEL_RLHF_PPO_PATH =

    # Best model from PPO
    model_best_PPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    model_best_PPO.load_state_dict(torch.load(MODEL_BEST_PPO_PATH))
    # Half model from PPO
    model_half_PPO = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    model_half_PPO.load_state_dict(torch.load(MODEL_HALF_PPO_PATH))
    # # Best DPO
    # model_DPO =
    # # Best RLHF-PPO
    # model_RLHF_PPO =

    model_list = [model_half_PPO, model_best_PPO]
    model_name_list = ["half PPO", "best PPO"]

    # Parameters
    n_episode = 10

    # for plot
    plt.figure(figsize=(8, 6))

    # Run 10 episodes for each model
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
            flierprops=dict(marker="o", color=COLOR_CODE[1], markeredgecolor=COLOR_CODE[1]),  # Outliers
        )

# Customizing the plot
plt.title("Box Plot Example")
plt.xlabel("Models")
plt.ylabel("Rewards")
plt.xticks([i + 1 for i in range(len(model_name_list))], model_name_list)

# Displaying the plot
plt.show()
