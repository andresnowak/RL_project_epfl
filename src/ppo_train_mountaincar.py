from copy import deepcopy
from itertools import count
from agents.ppo_discrete import *
import gymnasium as gym
from utils import plot_learning_curve

ENV_NAME = "MountainCar-v0"

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # N = 30
    batch_size = 32
    n_epochs = 10
    alpha = 1e-3  # lr
    agent = PPO_DISCRETE_AGENT(env=env, device=device, alpha=alpha, n_epochs=n_epochs, batch_size=batch_size)
    n_episode = 150

    score_history = []
    model_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_episode):
        observation, _ = env.reset()
        done = False
        score = 0

        while 1:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, terminated)
            observation = observation_

            done = terminated

            if done:
                print(f"episode {i} done... Score: {score}")
                agent.learn()
                learn_iters += 1
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                model_history.append(deepcopy(agent.ppo_policy.actor.state_dict()))
                break

        print("episode", i, "score %.1f" % score, "avg score %.1f" % avg_score, "time_steps", n_steps, "learning_steps", learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "checkpoints/learning_curve_" + ENV_NAME + ".png")

    env.close()

    # Get prefered model and half quality one
    avg_score_history = np.zeros(len(score_history))
    for i in range(len(score_history)):
        avg_score_history[i] = np.mean(score_history[max(0, i - 100) : (i + 1)])
    avg_score_history = avg_score_history.tolist()
    best_idx = avg_score_history.index(max(avg_score_history))
    best_model = model_history[best_idx]

    half_idx = min(range(len(avg_score_history)), key=lambda i: abs(avg_score_history[i] - avg_score_history[best_idx] * 3) + abs(score_history[i] - score_history[best_idx] * 3))
    half_model = model_history[half_idx]

    # save model
    print("save best model with score of", score_history[best_idx])
    torch.save(best_model, "checkpoints/best_actor_model_" + ENV_NAME)
    print("save half model with score of", score_history[half_idx])
    torch.save(half_model, "checkpoints/half_actor_model_" + ENV_NAME)
