# this is test environment for PPO_mountaincar-2.py
import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PPO_mountaincarContinuous import ActorNetwork, PPO

visualize = True
if visualize:
    render_mode = "human"
else:
    render_mode = None
env = gym.make("MountainCarContinuous-v0", render_mode="human")
agent1 = PPO(n_actions=env.action_space.shape, input_dims=env.observation_space.shape, action_bound=float(env.action_space.high[0]))
agent1.load_models("RL_PPO/model/best_actor_model_MCContinuous")
# agent2 = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
# agent2.load_models("RL_PPO/model/はlf_actor_model_MCContinuous")

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
