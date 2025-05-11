from copy import deepcopy
import gymnasium as gym
from itertools import count
from src.models.PPO_discrete import *

from src.models.utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # N = 30
    batch_size = 32
    n_epochs = 10
    alpha = 1e-3  # lr
    agent = PPO(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    n_episode = 300
    # min_rewards = -1000

    score_history = []
    model_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    seed = 1
    torch.manual_seed(seed)

    for i in range(n_episode):
        observation, _ = env.reset(seed=seed)
        done = False
        score = 0

        while 1:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, terminated)
            observation = observation_

            done = terminated or truncated

            if done:
                print(f"episode {i} done... Score: {score}")
                agent.learn()
                learn_iters += 1
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                model_history.append(deepcopy(agent.actor.state_dict()))
                break

        print("episode", i, "score %.1f" % score, "avg score %.1f" % avg_score, "time_steps", n_steps, "learning_steps", learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "checkpoints_cartpole/learning_curve.png")

    env.close()

    # Get prefered model and half quality one
    avg_score_history = np.zeros(len(score_history))
    for i in range(len(score_history)):
        avg_score_history[i] = np.mean(score_history[max(0, i - 100) : (i + 1)])
    avg_score_history = avg_score_history.tolist()
    best_idx = avg_score_history.index(max(avg_score_history))
    best_model = model_history[best_idx]


    coef = 1.5
    half_idx = min(range(len(avg_score_history)), key=lambda i: abs(avg_score_history[i] - avg_score_history[best_idx] / coef) + abs(score_history[i] - score_history[best_idx] / coef))
    half_model = model_history[half_idx]

    # save model
    print("save best model with score of", score_history[best_idx])
    torch.save(best_model, "checkpoints_cartpole/best_actor_model_CP")
    print("save half model with score of", score_history[half_idx])
    torch.save(half_model, "checkpoints_cartpole/half_actor_model_CP")

    # visualize the best model

    env = gym.make("CartPole-v1", render_mode="human")
    agent = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
    agent.load_models("checkpoints_cartpole/best_actor_model_CP")

    n_try = 1
    for _ in range(n_try):
        state, _ = env.reset()
        for _ in range(500):

            action, probs, value = agent.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            state = state_
            env.render()
            if done or truncated:
                break

    env.close()
