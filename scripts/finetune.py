"""BORG finetuning wrapper for GR00T N1.6.

Thin subprocess wrapper that calls the upstream finetuning entry point with
BORG-specific defaults injected. User-supplied arguments override defaults.

Usage::

    python scripts/finetune.py \
        --dataset-path /workspace/data/box_pickup_dataset \
        --max-steps 500 \
        --output-dir /tmp/borg-finetune
"""

from pathlib import Path
import subprocess
import sys

UPSTREAM_CANDIDATES = [
    Path("/opt/Isaac-GR00T"),  # Docker
    Path(__file__).resolve().parent.parent / "Isaac-GR00T",  # local dir
]


def find_upstream():
    for p in UPSTREAM_CANDIDATES:
        script = p / "gr00t" / "experiment" / "launch_finetune.py"
        if script.exists():
            return script
    raise FileNotFoundError(
        "Cannot find upstream launch_finetune.py. "
        "Expected Isaac-GR00T at: " + ", ".join(str(p) for p in UPSTREAM_CANDIDATES)
    )


def main():
    script = find_upstream()
    modality_cfg = Path(__file__).resolve().parent.parent / "borg_gr00t" / "modality_config.py"

    defaults = {
        "--embodiment-tag": "new_embodiment",
        "--modality-config-path": str(modality_cfg),
        "--base-model-path": "nvidia/GR00T-N1.6-3B",
    }

    user_args = sys.argv[1:]
    user_keys = {a.split("=")[0] for a in user_args if a.startswith("--")}

    inject = []
    for k, v in defaults.items():
        if k not in user_keys:
            inject.extend([k, v])

    cmd = [sys.executable, str(script)] + inject + user_args
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
