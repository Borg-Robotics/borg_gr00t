# Setup New Robot

This guide explains how to add support for a new robot using the external config pattern (no upstream fork required).

## 1. Create a modality config

Create a Python file that defines and registers your robot's modality config:

```python
# my_robot/modality_config.py
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

MY_ROBOT_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_head"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_1", "joint_2", ...],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joint_1", "joint_2", ...],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # ... one per action key
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(MY_ROBOT_CONFIG, EmbodimentTag.NEW_EMBODIMENT)
```

## 2. Prepare dataset

Your dataset should follow the LeRobot format with keys matching your modality config.

## 3. Finetune

```bash
python -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /path/to/dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path my_robot/modality_config.py \
    --output-dir /path/to/output
```

## 4. Inference

```bash
python -m gr00t.eval.run_gr00t_server \
    --model-path /path/to/output \
    --embodiment-tag new_embodiment
```

See `borg_gr00t/` for a complete working example.
