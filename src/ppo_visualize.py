from agents.ppo_discrete import *
import gymnasium as gym

ENV_NAME = "CartPole-v1"
# ENV_NAME = "MountainCar-v0"
MODEL_PATH = "checkpoints/best_actor_model_" + ENV_NAME


if __name__ == "__main__":

    env = gym.make(ENV_NAME, render_mode="human")
    actor = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    actor.load_state_dict(torch.load(MODEL_PATH))

    n_try = 1
    for _ in range(n_try):
        state, _ = env.reset()
        for _ in range(500):
            state = torch.tensor(state, dtype=torch.float)
            dist = actor(state)
            action = dist.sample()
            action = torch.squeeze(action).item()
            state_, reward, done, truncated, _ = env.step(action)
            state = state_
            env.render()
            if done or truncated:
                break

    env.close()
