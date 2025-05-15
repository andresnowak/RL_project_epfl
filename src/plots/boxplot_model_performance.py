### plot each model performance with box plot
import matplotlib.pyplot as plt
from models.ppo_policy import *

# Env
ENV_NAME = "CartPole-v1"
# ENV_NAME = "Acrobot-v1"

# PATH
DIR = "checkpoints/"
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

# Parameters
n_episode = 10

# Run 10 episodes for each model
for model in model_list:
    for _ in range(n_episode):
        state, _ = env.reset()
        done = False
        while not done:
            action = model.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
