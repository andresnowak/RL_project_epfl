import gymnasium as gym
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from src.modified_rewards import MountainCarContinuousRewardWrapper
from src.checkpoint_creator import CheckpointCallback
from src.find_middle_and_best_model import find_model_by_performance


def run(
    env: gym.Env,
    eval_env: gym.Env,
    log_dir: str,
    checkpoint_dir: str,
    save_model_name: str,
):
    callback = CheckpointCallback(eval_env, checkpoint_dir)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)
    model.learn(total_timesteps=200_000, callback=callback)

    # Save final model
    final_path = os.path.join(log_dir, save_model_name)
    model.save(final_path)

    # Update log with final model
    mean_reward, mean_length, solve_rate = callback._evaluate_model(model)
    composite_score = mean_reward * 10 - mean_length

    callback.checkpoint_data.append(
        {
            "step": model.num_timesteps,
            "model_path": final_path,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "solve_rate": solve_rate,
            "is_best": composite_score >= callback.best_score,
        }
    )

    env.close()
    eval_env.close()
    return {
        "final_model": final_path,
        "best_model": callback.best_model_path,
        "best_reward": callback.best_score,
    }


def eval(eval_env: gym.Env, model_path):
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Model: {model_path}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    eval_env.close()


if __name__ == "__main__":
    ENV_ID = "MountainCarContinuous-v0"
    log_dir = "./ppo_mountain_car_continuous_tensorboard/"
    save_model_name = "ppo_mountain_car_continuous_final"
    # Setup checkpoint callback
    checkpoint_dir = "./checkpoints/ppo_mountain_car_continuous"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(ENV_ID)
    env = MountainCarContinuousRewardWrapper(env)
    env = Monitor(env, log_dir)

    # Create eval environment for checkpoints
    eval_env = gym.make(ENV_ID)
    eval_env = MountainCarContinuousRewardWrapper(eval_env)
    eval_env = Monitor(eval_env)

    # Train and save checkpoints
    run(env, eval_env, log_dir, checkpoint_dir, save_model_name)

    # Example: Load models at different performance levels
    eval_env = gym.make(ENV_ID)
    eval_env = MountainCarContinuousRewardWrapper(eval_env)
    eval_env = Monitor(eval_env)

    # Evaluate final model
    print("\nFinal model:")
    final_model_path = os.path.join(checkpoint_dir, "best_model")
    print(final_model_path)
    eval(eval_env, final_model_path)

    # Find and evaluate 50% performance model
    print("\n50% performance model:")
    results = find_model_by_performance(checkpoint_dir, 0.5)
    half_model_path = results["partial_model"]

    if half_model_path:
        eval(eval_env, half_model_path)
    else:
        print("Could not find 50% performance model in checkpoints")

    print(results)
