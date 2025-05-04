import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch # For setting seed
import os # For creating plot directory

# Import the PPO class from the other file
from src.models.PPO_mountaincarContinuous import PPO
from src.modified_rewards import MountainCarContinuousRewardWrapper

# --- Plotting Function ---
def plot_learning_curve_matplotlib(x, scores, figure_file):
    # Ensure directory exists
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-99):(i+1)]) # 100-episode average
    plt.figure(figsize=(10, 6))
    plt.plot(x, scores, alpha=0.3, label='Episode Score')
    plt.plot(x, running_avg, color='orange', label='Avg Score (100 episodes)')
    plt.title('PPO Training Progress on MountainCarContinuous-v0')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    # Add a horizontal line at the target score if applicable
    # plt.axhline(y=90, color='r', linestyle='--', label='Target Score (90)')
    plt.savefig(figure_file)
    plt.close()
    print(f"Learning curve saved to {figure_file}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    ENV_NAME = 'MountainCarContinuous-v0'
    N_EPISODES = 30         # Number of episodes to run
    SEED = 42                # Random seed
    PRINT_INTERVAL = 10      # How often to print episode stats
    CHECKPOINT_DIR = f"tmp/ppo_{ENV_NAME}_learn_in_agent/" # Directory for model saves and plots
    FIGURE_FILE = os.path.join(CHECKPOINT_DIR, f'ppo_{ENV_NAME}_rewards.png') # Plot file name

    # Hyperparameters for the PPO Agent (passed during initialization)
    ALPHA = 3e-4          # Learning rate
    GAMMA = 0.99          # Discount factor
    GAE_LAMBDA = 0.95     # GAE lambda parameter
    POLICY_CLIP = 0.2     # PPO clipping parameter
    BATCH_SIZE = 64       # Batch size for learning updates
    N_EPOCHS = 10         # Number of epochs per learning update
    FC_DIMS = 256         # Hidden layer dimensions

    # --- Environment Setup ---
    # Use try-except for environment creation
    try:
        env = gym.make(ENV_NAME)
    except Exception as e:
        print(f"Error creating environment {ENV_NAME}: {e}")
        exit()

    env = MountainCarContinuousRewardWrapper(env)

    # Set seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Environment seeding is handled inside agent.learn via env.reset(seed=...)

    # --- Agent Initialization ---
    # Create the directory for checkpoints before initializing agent
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    agent = PPO(
        env=env, # Pass env for spec reading during init
        gamma=GAMMA,
        alpha=ALPHA,
        gae_lambda=GAE_LAMBDA,
        policy_clip=POLICY_CLIP,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device="cpu" # Auto-detect device ('cpu', 'cuda', 'mps')
    )

    # --- Start Training ---
    # Call the agent's learn method, which contains the main loop
    print(f"Starting training by calling agent.learn()...")
    score_history, model_history = agent.learn(
        num_timesteps=N_EPISODES,

    )

    # --- Plotting Results ---
    if score_history:
        episodes = [i + 1 for i in range(len(score_history))]
        plot_learning_curve_matplotlib(episodes, score_history, FIGURE_FILE)
    else:
        print("No scores recorded or returned, cannot plot learning curve.")

    best_model = model_history[-1]

    torch.save(best_model, "RL_PPO/model/best_actor_model_MCContinuous")

    # --- Cleanup ---
    env.close()
    print("--- Script Finished ---")