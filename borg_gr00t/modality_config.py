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
