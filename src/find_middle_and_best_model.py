import pandas as pd
import os

def find_model_by_performance(checkpoint_dir, performance_percent=0.5):
    """
    Gets two key models:
    1. Best overall model (highest composite score)
    2. Model at target performance percentile
    """
    try:
        df = pd.read_csv(os.path.join(checkpoint_dir, "training_log.csv"))
        if df.empty:
            return None

        # Calculate composite scores for all checkpoints
        df["composite_score"] = df["mean_reward"] * 10 - df["mean_length"]

        # 1. Get best model
        best_idx = df["composite_score"].idxmax()
        best_model = df.loc[best_idx, "model_path"]

        # 2. Get model at performance percentile
        sorted_df = df.sort_values("composite_score")
        min_score = sorted_df["composite_score"].min()
        max_score = sorted_df["composite_score"].max()
        target_score = min_score + (max_score - min_score) * performance_percent

        partial_idx = (sorted_df["composite_score"] - target_score).abs().idxmin()
        partial_model = sorted_df.loc[partial_idx, "model_path"]

        return {
            "best_model": best_model,
            "partial_model": partial_model,
            "best_metrics": df.loc[best_idx].to_dict(),
            "partial_metrics": df.loc[partial_idx].to_dict(),
        }

    except FileNotFoundError:
        print(f"Checkpoint log not found in {checkpoint_dir}")
        return None