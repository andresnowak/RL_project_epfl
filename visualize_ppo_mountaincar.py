from src.models.PPO_discrete import *
import gymnasium as gym

MODEL_PATH = "checkpoints_mountain_car/best_actor_model_MC"
ENV_NAME = "MountainCar-v0"

if __name__ == "__main__":

    env = gym.make(ENV_NAME, render_mode="human")
    agent = PPO(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
    agent.load_models(MODEL_PATH)

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
