from copy import deepcopy
from itertools import count
from agents.ppo_discrete import *
import gymnasium as gym
from utils import plot_learning_curve, select_models_by_fraction

ENV_NAME = "Acrobot-v1"

if __name__ == "__main__":
    # NOTE: Here we could just get arguments from cli for the env_name
    env = gym.make(ENV_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # N = 30
    batch_size = 32
    n_epochs = 10
    alpha = 1e-3  # lr
    agent = PPO_DISCRETE_AGENT(env=env, device=device, alpha=alpha, n_epochs=n_epochs, batch_size=batch_size)
    n_episode = 200

    score_history = []
    actor_model_history = []
    critic_model_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    eval_interval = 10
    eval_episodes = 10

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

            done = terminated or truncated

            if done:
                print(f"episode {i} done... Score: {score}")
                agent.learn()
                learn_iters += 1
                # score_history.append(score)
                # avg_score = np.mean(score_history[-100:])
                # actor_model_history.append(deepcopy(agent.ppo_policy.actor.state_dict()))
                # critic_model_history.append(deepcopy(agent.ppo_policy.critic.state_dict()))
                break

        print("episode", i, "score %.1f" % score, "time_steps", n_steps, "learning_steps", learn_iters)

        # Eval
        if i % eval_interval == 0 and i > 0:
            eval_scores = []
            for _ in range(eval_episodes):
                obs, _ = env.reset()
                done = False
                ep_score = 0
                while not done:
                    with torch.no_grad():
                        action, _, _ = agent.choose_action(
                            obs
                        )  # modify choose_action if needed
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_score += reward
                    done = terminated or truncated
                eval_scores.append(ep_score)

            mean_eval_score = np.mean(eval_scores)
            score_history.append(mean_eval_score)
            actor_model_history.append(deepcopy(agent.ppo_policy.actor.state_dict()))
            critic_model_history.append(deepcopy(agent.ppo_policy.critic.state_dict()))

            print(
                f"[Evaluation @ Episode {i}] Avg Score over {eval_episodes} episodes: {mean_eval_score:.2f}"
            )

    
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "checkpoints/learning_curve_" + ENV_NAME + ".png")

    env.close()


    (
        best_actor_model,
        best_critic_model,
        half_actor_model,
        half_critic_model,
        best_idx,
        half_idx,
    ) = select_models_by_fraction(
        score_history,
        actor_model_history,
        critic_model_history,
        performance_fraction=0.5,
        window_size=1,
    )

    # save actor model
    print("save best model with score of", score_history[best_idx])
    torch.save(best_actor_model, "checkpoints/best_actor_model_" + ENV_NAME)
    print("save half model with score of", score_history[half_idx])
    torch.save(half_actor_model, "checkpoints/half_actor_model_" + ENV_NAME)
    # save critic model
    torch.save(best_critic_model, "checkpoints/best_critic_model_" + ENV_NAME)
    torch.save(half_critic_model, "checkpoints/half_critic_model_" + ENV_NAME)
