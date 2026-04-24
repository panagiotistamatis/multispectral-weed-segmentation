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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# === ΒΗΜΑ 1: PyTorch πρώτα (μεγαλύτερο download, σπάνια αλλάζει) ===
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# === ΒΗΜΑ 2: ezdl από GitHub (specific commit — ίδιο με PyCharm env) ===
RUN pip install --no-cache-dir \
    "ezdl @ git+https://github.com/pasqualedem/ezdl.git@ffdd832349ca507600cc0bb5cece9d0bc1c83191"

# === ΒΗΜΑ 3: Όλα τα υπόλοιπα packages (pinned versions) ===
COPY requirements-frozen.txt .
RUN pip install --no-cache-dir -r requirements-frozen.txt

# Copy project files
COPY . .

# Create directories για dataset/outputs
RUN mkdir -p datasets/WeedMap outputs

# Default command: interactive bash
CMD ["bash"]
