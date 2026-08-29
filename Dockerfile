# ============================================================
# LWViTs-for-weedmapping Docker Image
# Versions frozen from working PyCharm environment
# Python 3.12 | PyTorch 2.8.0+cu126
# ============================================================

# Base: Official Python 3.12 (Debian Bookworm -- more reliable repos)
# We don't need an nvidia/cuda base because the PyTorch cu126 wheel
# already contains the CUDA runtime libraries inside it
FROM python:3.12-slim-bookworm

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (setuptools pinned -- v80+ removed pkg_resources which super-gradients needs)
RUN pip install --upgrade pip wheel && pip install "setuptools<80"

# Set working directory
WORKDIR /workspace

# === STEP 1: PyTorch first (largest download, rarely changes) ===
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# === STEP 2: All packages (pinned versions for Python 3.12) ===
# Install the requirements FIRST so we have modern versions,
# and THEN ezdl with --no-deps (its setup.py has old 2023 pins
# that don't build on Python 3.12 -- the dependencies are covered here)
COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

# === STEP 3: super-gradients with --no-deps (pins onnxruntime==1.13.1) ===
RUN pip install --no-cache-dir --no-deps super-gradients==3.5.0

# === STEP 4: ezdl from our own fork (--no-deps to avoid stale pins) ===
RUN pip install --no-cache-dir --no-deps \
    "ezdl @ git+https://github.com/panagiotis890/ezdl.git@weedsgalore-losses"

# Copy project files
COPY . .

# Create directories for dataset/outputs
RUN mkdir -p datasets/WeedMap outputs

# Default command: interactive bash
CMD ["bash"]
