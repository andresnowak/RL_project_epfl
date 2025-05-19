from agents.ppo_discrete import *
import gymnasium as gym

ENV_NAME = "CartPole-v1"
# ENV_NAME = "MountainCar-v0"
MODEL_PATH = "checkpoints/half_actor_model_" + ENV_NAME

DIR = "checkpoints/"
MODEL_DPO_PATH = DIR + "DPO/s2_K3000_" + ENV_NAME + ".pth"
MODEL_RLHF_PPO_PATH = DIR + "RLHF_PPO/s2_K500_" + ENV_NAME + ".pth"


if __name__ == "__main__":

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: True, disable_logger=True)

    actor = ActorNetwork(env.action_space.n, env.observation_space.shape[0])
    # actor.load_state_dict(torch.load(MODEL_DPO_PATH, map_location=torch.device("cpu")))
    temp_model = torch.load(MODEL_RLHF_PPO_PATH)
    actor.load_state_dict(temp_model["../RLHF_models/" + ENV_NAME + "/policy"])

    n_try = 1
    for _ in range(n_try):
        state, _ = env.reset()
        for _ in range(500):
            action = actor.choose_action(state)
            state_, reward, done, truncated, _ = env.step(action)
            state = state_

            # Render to collect frames for the video
            env.render()

            if done or truncated:
                break

    env.close()
