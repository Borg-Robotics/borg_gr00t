"""Offline evaluation: load model + replay episode in a single process.

Loads the finetuned model directly (no server), replays a recorded episode,
compares predicted actions against ground truth, and saves a plot.

Usage::

    python scripts/evaluate_offline.py \
        --model-path /path/to/checkpoint \
        --dataset-path /path/to/data \
        --episode-id 000000
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tyro
from borg_gr00t.embodiment import resolve_embodiment_enum
from borg_gr00t.modality_config import (
    inject_modality_config_into_checkpoint,
    inject_statistics_into_checkpoint,
)
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.policy.gr00t_policy import Gr00tPolicy

ACTION_KEYS = [
    "l_arm_pivot_1_joint",
    "l_arm_pivot_2_joint",
    "l_arm_pivot_3_joint",
    "l_arm_pivot_4_joint",
    "l_arm_pivot_5_joint",
    "l_arm_pivot_6_joint",
    "l_gripper_position",
    "r_arm_pivot_1_joint",
    "r_arm_pivot_2_joint",
    "r_arm_pivot_3_joint",
    "r_arm_pivot_4_joint",
    "r_arm_pivot_5_joint",
    "r_arm_pivot_6_joint",
    "r_gripper_position",
]


@dataclass
class Args:
    model_path: str
    """Path to finetuned checkpoint directory."""

    dataset_path: str
    """Path to LeRobot-format dataset."""

    embodiment_tag: str = "borg"
    """Embodiment tag."""

    episode_id: int = 0
    """Episode index to replay."""

    max_steps: int = 0
    """Max steps to evaluate (0 = all)."""

    device: str = "cuda"
    """Device for inference."""

    output_dir: str = "."
    """Where to save the plot."""


def main(args: Args):
    tag_enum = resolve_embodiment_enum(args.embodiment_tag)

    # Patch checkpoint with BORG config and normalization statistics
    inject_modality_config_into_checkpoint(args.model_path)
    inject_statistics_into_checkpoint(args.model_path, args.dataset_path)

    # Load policy
    print(f"Loading model from {args.model_path} ...")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag_enum,
        device=args.device,
    )
    modality_config = policy.get_modality_config()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    episode_data = dataset[args.episode_id]
    num_steps = len(episode_data)
    if args.max_steps > 0:
        num_steps = min(num_steps, args.max_steps)
    print(f"Episode {args.episode_id}: {num_steps} steps")

    gt_actions = []
    pred_actions = []

    for step_idx in tqdm(range(num_steps), desc="Replaying"):
        step_data = extract_step_data(
            episode_data,
            step_index=step_idx,
            modality_configs=modality_config,
            embodiment_tag=tag_enum,
            allow_padding=True,
        )

        # Build observation dict (same format as notebook)
        observation = {
            "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
            "state": {k: step_data.states[k][None] for k in step_data.states},
            "action": {k: step_data.actions[k][None] for k in step_data.actions},
            "language": {
                modality_config["language"].modality_keys[0]: [[step_data.text]],
            },
        }

        predicted, _info = policy.get_action(observation)

        # Extract first timestep (index 0) for each action key
        gt_vec = []
        pred_vec = []
        for key in ACTION_KEYS:
            if key in step_data.actions:
                gt_vec.append(step_data.actions[key][0].flatten())
            if key in predicted:
                pred_vec.append(np.array(predicted[key]).flatten()[0:1])

        gt_actions.append(np.concatenate(gt_vec))
        pred_actions.append(np.concatenate(pred_vec))

    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)

    print(f"\nCollected {len(pred_actions)} predictions")
    print(f"Predicted shape: {pred_actions.shape}, GT shape: {gt_actions.shape}")

    # Per-joint MAE
    num_joints = gt_actions.shape[1]
    mae_per_joint = [
        mean_absolute_error(gt_actions[:, j], pred_actions[:, j]) for j in range(num_joints)
    ]

    print("\nMean Absolute Error per Joint:")
    for j, mae in enumerate(mae_per_joint):
        print(f"  {ACTION_KEYS[j]:30s}: {mae:.6f}")
    print(f"\n  Average MAE: {np.mean(mae_per_joint):.6f}")

    # Plot
    ncols = 2
    nrows = (num_joints + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2 * nrows))
    axes = axes.flatten()
    for j in range(num_joints):
        axes[j].plot(gt_actions[:, j], label="Ground Truth", color="blue")
        axes[j].plot(pred_actions[:, j], label="Predicted", color="red", linestyle="--")
        axes[j].set_title(ACTION_KEYS[j])
        axes[j].legend(fontsize=7)
        axes[j].grid(True)
    for j in range(num_joints, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_path = f"{args.output_dir}/replay_eval_ep{args.episode_id:06d}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
