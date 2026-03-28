"""Plot state vs action comparison from a BORG dataset episode.

Usage::

    python scripts/plot_actions.py \
        --dataset-path /workspace/data/box_pickup_dataset \
        --episode-id 000000 \
        --output-path joint_action_comparison.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(args):
    parquet_path = f"{args.dataset_path}/data/chunk-000/episode_{args.episode_id}.parquet"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataset: {parquet_path}")
    print("Columns:", df.columns.tolist())

    state_array = np.stack(df["observation.state"].to_numpy())
    action_array = np.stack(df["action"].to_numpy())

    num_state = state_array.shape[1]
    num_action = action_array.shape[1]
    num_joints = max(num_state, num_action)

    state_df = pd.DataFrame(state_array, columns=[f"state.joint_{i + 1}" for i in range(num_state)])
    action_df = pd.DataFrame(
        action_array, columns=[f"action.joint_{i + 1}" for i in range(num_action)]
    )
    full_df = pd.concat([state_df, action_df], axis=1)

    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2 * nrows))
    axes = axes.flatten()

    for i in range(num_joints):
        if f"state.joint_{i + 1}" in full_df:
            axes[i].plot(full_df[f"state.joint_{i + 1}"], label="state", color="blue")
        if f"action.joint_{i + 1}" in full_df:
            axes[i].plot(
                full_df[f"action.joint_{i + 1}"], label="action", color="red", linestyle="--"
            )
        axes[i].set_title(f"Joint {i + 1}")
        axes[i].set_xlabel("Frame index")
        axes[i].set_ylabel("Position (radians)")
        axes[i].grid(True)
        axes[i].legend()
    for i in range(num_joints, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Saved joint comparison plot to: {args.output_path}")

    # MSE check
    shared_joints = min(num_state, num_action)
    print("\nMean Squared Error between state and action per joint:")
    for i in range(shared_joints):
        mse = np.mean((full_df[f"state.joint_{i + 1}"] - full_df[f"action.joint_{i + 1}"]) ** 2)
        print(f"  Joint {i + 1}: MSE = {mse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--episode-id", type=str, default="000000")
    parser.add_argument("--output-path", type=str, default="joint_action_comparison.png")
    args = parser.parse_args()
    main(args)
