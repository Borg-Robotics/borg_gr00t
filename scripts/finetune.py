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
import shutil
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


PROCESSOR_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "statistics.json",
]


def _resolve_base_model(path_or_repo: str) -> Path:
    """Resolve a HuggingFace repo ID or local path to a directory."""
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, local_files_only=True))


def _copy_processor_files(base_model_path: str, output_dir: str):
    """Copy processor/tokenizer files from the base model into the checkpoint."""
    base = _resolve_base_model(base_model_path)
    out = Path(output_dir)
    if not out.is_dir():
        return
    copied = []
    for fname in PROCESSOR_FILES:
        src = base / fname
        dst = out / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
    if copied:
        print(f"Copied processor files to {out}: {', '.join(copied)}")


def _get_arg(args: list[str], key: str, default: str | None = None) -> str | None:
    """Extract the value for --key from an args list."""
    for i, a in enumerate(args):
        if a == key and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"{key}="):
            return a.split("=", 1)[1]
    return default


def main():
    script = find_upstream()
    modality_cfg = Path(__file__).resolve().parent.parent / "borg_gr00t" / "modality_config.py"

    defaults = {
        "--embodiment-tag": "NEW_EMBODIMENT",
        "--modality-config-path": str(modality_cfg),
        "--base-model-path": "nvidia/GR00T-N1.6-3B",
    }

    user_args = sys.argv[1:]
    user_keys = {a.split("=")[0] for a in user_args if a.startswith("--")}

    inject = []
    for k, v in defaults.items():
        if k not in user_keys:
            inject.extend([k, v])

    all_args = inject + user_args
    cmd = [sys.executable, str(script)] + all_args
    result = subprocess.run(cmd)

    if result.returncode == 0:
        base_model = _get_arg(all_args, "--base-model-path", defaults["--base-model-path"])
        output_dir = _get_arg(all_args, "--output-dir")
        if output_dir:
            _copy_processor_files(base_model, output_dir)
            from borg_gr00t.modality_config import inject_modality_config_into_checkpoint

            inject_modality_config_into_checkpoint(output_dir)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
