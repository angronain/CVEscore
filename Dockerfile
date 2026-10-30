FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CUDA 12.8 wheels: the first PyTorch builds with Blackwell (sm_120, RTX 50-series) kernels.
# The wheel bundles the CUDA runtime, so the image only needs the host NVIDIA driver.
RUN pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

# Runtime-only env, kept below the pip installs so changing it doesn't invalidate those layers.
# USER/LOGNAME: compose runs as your UID, which has no /etc/passwd entry in the image.
# Without them, getpass.getuser() (called by torch on import) crashes with "uid not found".
ENV USER=cvescore \
    LOGNAME=cvescore \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

# Writable cache for any container user (compose runs as your UID so outputs aren't root-owned)
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

COPY src/ src/
COPY scripts/ scripts/
COPY app.py .
