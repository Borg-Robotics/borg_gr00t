# CLAUDE.md

## Project Overview

BORG robot extensions for NVIDIA Isaac GR00T N1.6. Standalone package that installs gr00t as a dependency. Provides BORG-specific modality configs, inference scripts, and finetuning wrappers.

## Commands

```bash
# Format
make format              # ruff check --fix + ruff format

# Lint & test
make run-checks          # ruff check + ruff format --check + pytest
```

## Code Style

- **ruff**: line-length = 100, target py310, `__init__.py` ignores F401

## Key Entry Points

```bash
# Fine-tuning (subprocess wrapper around upstream launch_finetune.py; injects BORG defaults)
python scripts/finetune.py --dataset-path /path/to/data

# Inference (upstream PolicyServer)
python scripts/inference_service.py --server --model-path /path/to/checkpoint

# Inference (raw ZMQ for robot client)
python scripts/inference_server_zmq.py --server --model-path /path/to/checkpoint --port 5556

# Evaluation
python scripts/evaluate_replay.py --dataset-path /path/to/data --episode-id 000005

# Plotting
python scripts/plot_actions.py --dataset-path /path/to/data
```

## Architecture

- **`borg_gr00t/modality_config.py`** — Defines and registers BORG modality config with upstream `register_modality_config()`
- **`borg_gr00t/embodiment.py`** — Translates `"borg"` → `"new_embodiment"` for upstream compatibility
- **`borg_gr00t/actions.py`** — Relative ↔ absolute action conversion (N1.6 uses relative actions)
- **`borg_gr00t/configs/`** — JSON modality config for BORG W1

## N1.6 Key Differences from N1.5

- `Gr00tPolicy` constructor: `(embodiment_tag, model_path, *, device, strict)` — no more `modality_config`/`modality_transform`
- Entry points: `gr00t/eval/run_gr00t_server.py` (inference), `gr00t/experiment/launch_finetune.py` (training)
- Actions are state-relative, not absolute
- Custom modality configs registered via `register_modality_config()` in `gr00t/configs/data/embodiment_configs.py`
- `PolicyServer` replaces `RobotInferenceServer`

## Docker

```bash
docker compose -f docker/compose.yaml up --build
docker exec -it borg-gr00t bash
```
