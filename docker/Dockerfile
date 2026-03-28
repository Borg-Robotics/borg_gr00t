FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt <<EOT
apt-get update
apt-get install -qqy --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libvulkan-dev \
    libxext6 \
    libxrender-dev \
    sudo \
    tensorrt \
    unzip
EOT

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/conda/

# Clone upstream gr00t and install via uv sync
COPY Isaac-GR00T /borg_gr00t/Isaac-GR00T
RUN --mount=type=cache,target=/root/.cache/uv \
    <<EOT
uv pip install --system setuptools wheel wheel-stub
cd /borg_gr00t/Isaac-GR00T
uv sync --no-build-isolation --no-install-project
uv pip install --system -e /borg_gr00t/Isaac-GR00T --no-deps
uv pip install --system flash-attn --no-build-isolation
EOT

# Install borg_gr00t deps (cached unless pyproject.toml changes)
COPY pyproject.toml /borg_gr00t/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /borg_gr00t/__init__.py && \
    uv pip install --system -e "/borg_gr00t[dev]" --no-build-isolation

# Copy source and re-register editable install (fast, no deps to fetch)
COPY . /borg_gr00t
RUN uv pip install --system -e /borg_gr00t --no-deps

ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /borg_gr00t
