# Jetson benchmark image
# Usage:
#   docker build -f jetson/Dockerfile --build-arg BASE_IMAGE=<provided_image> -t track-bench:jetson .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace/tesi_project_amarino

# Minimal OS deps for running inference + OpenCV bindings (Jetson images usually already include many of these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY tesi_project_amarino/ ./

# Install Python deps (note: on Jetson, torch/torchvision are typically preinstalled in the base image)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt || true

# Default entrypoint is an interactive shell; you will run benchmark commands manually.
CMD ["bash"]
