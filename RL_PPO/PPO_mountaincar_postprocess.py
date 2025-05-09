# this is test environment for PPO_mountaincar-2.py
import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PPO_mountaincar import ActorNetwork, PPO

visualize = False
if visualize:
    render_mode = "human"
else:
    render_mode = None
env = gym.make("MountainCar-v0", render_mode=render_mode)
agent1 = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
agent1.load_models("RL_PPO/model/best_actor_model")
agent2 = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
agent2.load_models("RL_PPO/model/half_actor_model")

if visualize:
    n_try = 1
    for _ in range(n_try):
        state, _ = env.reset()
        for _ in range(500):

            action, probs, value = agent1.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            state = state_
            env.render()
            if done or truncated:
                break

    env.close()

# generate trajectory
K = 30
trajectory_buffer = []
trajectory = []  # [(t_1,t_2)_1,...,(t_1,t_2)_K]
reward_buffer = []  # [(R_1,R_2)_1,...,(R_1,R_2)_K]

for k in range(K):
    # agent1
    state, _ = env.reset()
    done = False
    cum_reward1 = 0
    cum_reward2 = 0
    while not done:
        action, probs, value = agent1.choose_action(state)
        state_, reward, terminated, truncated, _ = env.step(action)
        state = state_
        cum_reward1 += reward
        done = terminated

    # agent2
    state, _ = env.reset()
    done = False

    while not done:
        action, probs, value = agent2.choose_action(state)
        state_, reward, terminated, truncated, _ = env.step(action)
        state = state_
        cum_reward2 += reward
        done = terminated

    reward_buffer.append([cum_reward1, cum_reward2])
print(reward_buffer)
