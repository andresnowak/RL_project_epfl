import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pandas as pd
import numpy as np


class CheckpointCallback:
    def __init__(self, eval_env: gym.Env, save_dir: str, checkpoint_freq: int =20000, n_eval_episodes: int=5):
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_data = []
        self.best_score = -float("inf")  # Composite score tracking
        self.best_model_path = None
        os.makedirs(save_dir, exist_ok=True)

    def _evaluate_model(self, model):
        """Returns (mean_reward, mean_length, n_solved)"""
        episode_rewards = []
        episode_lengths = []
        solved_count = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            total_reward = 0
            length = 0

            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                length += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            if not truncated: 
                solved_count += 1

        return (
            np.mean(episode_rewards),
            np.mean(episode_lengths),
            solved_count / self.n_eval_episodes,
        )

    def __call__(self, locals_, globals_):
        step = locals_["self"].num_timesteps
        if step % self.checkpoint_freq == 0:
            model = locals_["self"]

            # Evaluate with comprehensive metrics
            mean_reward, mean_length, solve_rate = self._evaluate_model(model)

            # Calculate composite score (prioritize higher rewards first, then shorter lengths)
            composite_score = (
                mean_reward * 10 - mean_length
            )  # Scale reward to dominate

            # Save checkpoint
            model_path = os.path.join(self.save_dir, f"checkpoint_{step}")
            model.save(model_path)

            # Track best model (using composite score)
            if composite_score > self.best_score:
                self.best_score = composite_score
                best_path = os.path.join(self.save_dir, "best_model")
                model.save(best_path)
                self.best_model_path = best_path

            # Record comprehensive metrics
            self.checkpoint_data.append(
                {
                    "step": step,
                    "model_path": model_path,
                    "mean_reward": mean_reward,
                    "mean_length": mean_length,
                    "solve_rate": solve_rate,
                    "is_best": (composite_score == self.best_score),
                }
            )

            # Save to CSV
            pd.DataFrame(self.checkpoint_data).to_csv(
                os.path.join(self.save_dir, "training_log.csv"), index=False
            )

            print(
                f"Step {step}: "
                f"Mean reward={mean_reward:.1f} | "
                f"Mean length={mean_length:.0f} | "
                f"Solved={solve_rate:.0%}"
                f"{' (BEST)' if composite_score == self.best_score else ''}"
            )

        return True


# class CheckpointCallback_2:
#     def __init__(self, eval_env, save_dir, checkpoint_freq=20000, n_eval_episodes=5):
#         self.eval_env = eval_env
#         self.save_dir = save_dir
#         self.checkpoint_freq = checkpoint_freq
#         self.n_eval_episodes = n_eval_episodes
#         self.checkpoint_data = []
#         self.best_reward: float = -float("inf")
#         self.best_model_path = None
#         os.makedirs(save_dir, exist_ok=True)

#     def __call__(self, locals_, globals_):
#         step = locals_["self"].num_timesteps
#         if step % self.checkpoint_freq == 0:
#             model = locals_["self"]

#             # Evaluate model
#             mean_reward, _ = evaluate_policy(
#                 model, self.eval_env, n_eval_episodes=self.n_eval_episodes
#             )

#             # Save checkpoint
#             model_path = os.path.join(self.save_dir, f"checkpoint_{step}")
#             model.save(model_path)

#             # Track best model
#             if mean_reward > self.best_reward:
#                 self.best_reward = mean_reward
#                 best_path = os.path.join(self.save_dir, "best_model")
#                 model.save(best_path)
#                 self.best_model_path = best_path

#             # Record checkpoint info
#             self.checkpoint_data.append(
#                 {
#                     "step": step,
#                     "model_path": model_path,
#                     "mean_reward": mean_reward,
#                     "is_best": (mean_reward == self.best_reward),
#                 }
#             )

#             # Save to CSV
#             pd.DataFrame(self.checkpoint_data).to_csv(
#                 os.path.join(self.save_dir, "checkpoint_log.csv"), index=False
#             )

#             print(
#                 f"Checkpoint {step} | Mean reward: {mean_reward:.1f} | Best: {self.best_reward:.1f}"
#             )

#         return True
