import gymnasium as gym

from src.modified_rewards import MountainCarContinuousRewardWrapper

def create_env_continuous(env_id: str, seed: int = 0) -> gym.Env:
    env = gym.make(env_id)
    env = MountainCarContinuousRewardWrapper(env)

    return env