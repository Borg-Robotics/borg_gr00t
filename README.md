# Borg GR00T

BORG robot extensions for [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T). This package provides BORG-specific modality configs, inference scripts, and finetuning wrappers without forking the upstream repo.

## Installation

### Prerequisites

- NVIDIA driver >= 550 (for CUDA 12.x support)
- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- System packages: `build-essential`, `cmake`, `ffmpeg`, `git-lfs`

### Local development

```bash
git clone --recursive git@github.com:Borg-Robotics/borg_gr00t.git

uv venv
source .venv/bin/activate
cd Isaac-GR00T
uv pip install setuptools wheel_stub
uv sync --no-build-isolation --no-install-project --active
uv pip install -e . --no-deps
cd ..

uv pip install -e ".[dev]"
```

> **flash-attn note:** The upstream `[tool.uv.sources]` points to a pre-built flash-attn wheel. If it fails on your GPU architecture, you can install from source with `pip install flash-attn --no-build-isolation` (requires CUDA toolkit and ~10 min compile time).

### Verify installation

```bash
python -c "from borg_gr00t.modality_config import BORG_MODALITY_CONFIG; print('OK')"
python -c "from borg_gr00t.embodiment import resolve_embodiment_tag; print(resolve_embodiment_tag('borg'))"
```

### Docker (recommended)

```bash
docker compose -f docker/compose.yaml up --build -d
docker exec -it borg-gr00t bash
```

## Quick Start

### Finetuning

```bash
python scripts/finetune.py \
    --dataset-path ./data \
    --max-steps 500 \
    --output-dir /tmp/borg-finetune
```

BORG defaults (`--embodiment-tag new_embodiment`, `--base-model-path nvidia/GR00T-N1.6-3B`, `--modality-config-path`) are injected automatically. Pass any upstream `launch_finetune.py` argument to override.

### Inference (using upstream PolicyServer)

```bash
# Server
python scripts/inference_service.py --server \
    --model-path /tmp/borg-finetune \
    --embodiment-tag borg

# Client test
python scripts/inference_service.py --client
```

### Inference (ZMQ server for robot)

```bash
# Server (accepts base64 video + flat joint states from robot)
python scripts/inference_server_zmq.py --server \
    --model-path /tmp/borg-finetune \
    --embodiment-tag borg \
    --port 5556

# Client test
python scripts/inference_server_zmq.py --client --port 5556
```

### Evaluate replay

```bash
python scripts/evaluate_replay.py \
    --dataset-path ./data \
    --episode-id 000005 \
    --port 5556
```

### Evaluate offline (no server)

```bash
python scripts/evaluate_offline.py \
    --model-path /tmp/borg-finetune \
    --dataset-path ./data \
    --episode-id 000000
```

## Architecture

```
borg_gr00t/
├── borg_gr00t/          # Core package
│   ├── modality_config.py  # BORG modality config (registers with gr00t)
│   ├── embodiment.py       # 'borg' → 'new_embodiment' tag resolution
│   ├── actions.py          # Relative ↔ absolute action conversion
│   └── configs/            # JSON modality configs
├── scripts/             # Runnable scripts
│   ├── finetune.py         # Finetuning wrapper (calls upstream launch_finetune.py)
│   ├── inference_service.py   # PolicyServer-based inference
│   ├── inference_server_zmq.py # Raw ZMQ server for robot
│   ├── evaluate_replay.py     # Replay evaluation (requires running server)
│   ├── evaluate_offline.py    # Offline evaluation (loads model directly)
│   └── plot_actions.py        # Action visualization
└── docker/              # Docker setup
```
