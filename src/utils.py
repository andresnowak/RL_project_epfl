import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def select_models_by_fraction(
    score_history,
    actor_model_history,
    critic_model_history,
    performance_fraction=0.5,
    window_size=100,
):
    # Smooth the score history
    avg_scores = np.array(
        [
            np.mean(score_history[max(0, i - window_size) : i + 1])
            for i in range(len(score_history))
        ]
    )

    # Best and worst scores
    # we grab the last model that had the best score, just because it is more probable that model is better than the first best model
    max_value = np.max(avg_scores)
    # Reverse search for max value
    best_idx = len(avg_scores) - 1 - np.argmax(avg_scores[::-1])
    best_score = avg_scores[best_idx]
    worst_score = np.min(avg_scores)
    print(best_score)

    # Compute target as a fraction between best and worst (normalized scale)
    target_score = worst_score + performance_fraction * (best_score - worst_score)
    print(target_score)

    # Exclude best model from candidates
    candidate_indices = np.delete(np.arange(len(avg_scores)), best_idx)
    candidate_scores = np.delete(avg_scores, best_idx)

    # Find model whose score is closest to the target
    secondary_relative_idx = np.argmin(np.abs(candidate_scores - target_score))
    secondary_idx = candidate_indices[secondary_relative_idx]

    return (
        actor_model_history[best_idx],
        critic_model_history[best_idx],
        actor_model_history[secondary_idx],
        critic_model_history[secondary_idx],
        best_idx,
        secondary_idx,
    )