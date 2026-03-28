"""BORG modality configuration for Isaac GR00T N1.6.

This module defines and registers the BORG robot's modality config with the
upstream GR00T config system. Import this module (or pass its path via
``--modality-config-path``) to make the BORG config available for finetuning.

Usage in finetuning::

    python -m gr00t.experiment.launch_finetune \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-path /path/to/borg_dataset \
        --embodiment-tag new_embodiment \
        --modality-config-path borg_gr00t/modality_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# fmt: off
BORG_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "l_arm_pivot_1_joint",
            "l_arm_pivot_2_joint",
            "l_arm_pivot_3_joint",
            "l_arm_pivot_4_joint",
            "l_arm_pivot_5_joint",
            "l_arm_pivot_6_joint",
            "l_gripper_position",
            "l_gripper_contact",
            "r_arm_pivot_1_joint",
            "r_arm_pivot_2_joint",
            "r_arm_pivot_3_joint",
            "r_arm_pivot_4_joint",
            "r_arm_pivot_5_joint",
            "r_arm_pivot_6_joint",
            "r_gripper_position",
            "r_gripper_contact",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
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
        ],
        action_configs=[
            # Left arm joints (6) — relative for N1.6
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # Left gripper — absolute (binary-like control)
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # Right arm joints (6) — relative for N1.6
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # Right gripper — absolute (binary-like control)
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}
# fmt: on

# Register on import so that finetuning picks it up automatically
register_modality_config(BORG_MODALITY_CONFIG, EmbodimentTag.NEW_EMBODIMENT)


def inject_modality_config_into_checkpoint(checkpoint_dir: str) -> None:
    """Patch processor_config.json in a checkpoint to include the BORG modality config.

    This is needed because finetuning saves processor configs from the base model,
    which doesn't include custom embodiment configs. Call this after copying
    processor files from the base model, or before loading a finetuned checkpoint.
    """
    import json
    from pathlib import Path

    from gr00t.data.utils import to_json_serializable

    config_path = Path(checkpoint_dir) / "processor_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    modality_configs = config.get("processor_kwargs", {}).get("modality_configs", {})
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    if tag not in modality_configs:
        modality_configs[tag] = to_json_serializable(BORG_MODALITY_CONFIG)
        config.setdefault("processor_kwargs", {})["modality_configs"] = modality_configs
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Injected BORG modality config into {config_path}")


def inject_statistics_into_checkpoint(checkpoint_dir: str, dataset_path: str) -> None:
    """Inject BORG normalization statistics into a checkpoint's statistics.json.

    The base model's statistics.json does not contain entries for the BORG
    embodiment (``new_embodiment``). This reads per-joint statistics from the
    dataset's ``meta/`` directory, structures them according to the BORG modality
    config, and merges them into the checkpoint so the model can normalise
    states and actions at inference time.
    """
    import json
    from pathlib import Path

    stats_path = Path(checkpoint_dir) / "statistics.json"
    tag = EmbodimentTag.NEW_EMBODIMENT.value

    # Load existing statistics (may come from the base model)
    if stats_path.exists():
        with open(stats_path) as f:
            statistics = json.load(f)
    else:
        statistics = {}

    if tag in statistics:
        return

    # Read raw dataset statistics
    meta_dir = Path(dataset_path) / "meta"
    with open(meta_dir / "stats.json") as f:
        raw_stats = json.load(f)
    with open(meta_dir / "modality.json") as f:
        modality_meta = json.load(f)

    # Structure statistics per joint group using the BORG modality config
    mapping = {"state": "observation.state", "action": "action"}
    dataset_statistics: dict = {}

    for modality, default_key in mapping.items():
        dataset_statistics[modality] = {}
        for joint_key in BORG_MODALITY_CONFIG[modality].modality_keys:
            meta = modality_meta[modality][joint_key]
            start_idx = meta["start"]
            end_idx = meta["end"]
            actual_key = meta.get("original_key", default_key)
            dataset_statistics[modality][joint_key] = {
                stat_type: raw_stats[actual_key][stat_type][start_idx:end_idx]
                for stat_type in raw_stats[actual_key]
            }

    # Add relative action statistics if available
    rel_stats_path = meta_dir / "relative_stats.json"
    if rel_stats_path.exists():
        with open(rel_stats_path) as f:
            dataset_statistics["relative_action"] = json.load(f)

    statistics[tag] = dataset_statistics
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"Injected BORG statistics into {stats_path}")
