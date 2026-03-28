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

# Clone Borg fork of gr00t and install via uv sync (reads [tool.uv.sources])
RUN git clone --depth 1 https://github.com/Borg-Robotics/Isaac-GR00T.git /opt/Isaac-GR00T
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system setuptools wheel wheel-stub && \
    cd /opt/Isaac-GR00T && \
    uv sync --extra base --no-build-isolation --no-install-project && \
    uv pip install --system -e /opt/Isaac-GR00T --no-deps

# Install borg_gr00t
COPY pyproject.toml /opt/borg_gr00t/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e /opt/borg_gr00t --no-deps

COPY . /opt/borg_gr00t

ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /workspace
