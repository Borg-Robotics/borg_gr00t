"""Action conversion utilities for BORG with GR00T N1.6.

N1.6 uses state-relative actions during training/inference. These helpers
convert between absolute joint angles (used by the robot) and relative deltas
(used by the model).
"""

import numpy as np


def absolute_to_relative(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert absolute joint targets to state-relative deltas.

    Used for dataset preparation (converting N1.5-style absolute actions to
    N1.6-style relative actions).

    Args:
        actions: Absolute joint targets, shape (T, D) or (D,).
        state: Current joint state, shape (D,).

    Returns:
        Relative action deltas, same shape as actions.
    """
    return actions - state


def relative_to_absolute(deltas: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert state-relative deltas to absolute joint targets.

    Used during inference to convert model output back to robot commands.

    Args:
        deltas: Relative action deltas from model, shape (T, D) or (D,).
        state: Current joint state, shape (D,).

    Returns:
        Absolute joint targets, same shape as deltas.
    """
    return deltas + state
