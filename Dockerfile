# ============================================================
# LWViTs-for-weedmapping Docker Image
# Versions frozen from working PyCharm environment
# Python 3.12 | PyTorch 2.8.0+cu126
# ============================================================

# Base: Official Python 3.12 (Debian Bookworm — πιο αξιόπιστα repos)
# Δεν χρειαζόμαστε nvidia/cuda base γιατί το PyTorch cu126 wheel
# περιέχει ήδη τις CUDA runtime libraries μέσα του
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

# Upgrade pip (setuptools pinned — v80+ removed pkg_resources which super-gradients needs)
RUN pip install --upgrade pip wheel && pip install "setuptools<80"

# Set working directory
WORKDIR /workspace

# === ΒΗΜΑ 1: PyTorch πρώτα (μεγαλύτερο download, σπάνια αλλάζει) ===
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# === ΒΗΜΑ 2: Όλα τα packages (pinned versions για Python 3.12) ===
# Εγκαθιστούμε PΡΩΤΑ τα requirements ώστε να έχουμε modern εκδόσεις,
# και ΜΕΤΑ το ezdl με --no-deps (το setup.py του έχει παλιά 2023 pins
# που δεν χτίζουν σε Python 3.12 — τα dependencies καλύπτονται εδώ)
COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

# === ΒΗΜΑ 3: super-gradients με --no-deps (pins onnxruntime==1.13.1) ===
RUN pip install --no-cache-dir --no-deps super-gradients==3.5.0

# === ΒΗΜΑ 4: ezdl από δικό μας fork (--no-deps για αποφυγή stale pins) ===
RUN pip install --no-cache-dir --no-deps \
    "ezdl @ git+https://github.com/panagiotis890/ezdl.git@weedsgalore-losses"

# Copy project files
COPY . .

# Create directories για dataset/outputs
RUN mkdir -p datasets/WeedMap outputs

# Default command: interactive bash
CMD ["bash"]
