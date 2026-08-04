# ==============================================================================
# Production Dockerfile for FutureGenImage AI Service
# Supports NVIDIA GPU passthrough, FastAPI serving, and cached Hugging Face weights
# ==============================================================================

# Base image with Python 3.11 and CUDA 12.1 support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    PATH="/venv/bin:$PATH"

# Install system dependencies & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3.11 -m venv /venv

# Upgrade pip & build tools inside virtualenv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file first to maximize Docker layer caching
COPY requirements.txt .

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code into container
COPY Api /app/Api
COPY Inference /app/Inference
COPY Train /app/Train
COPY Dataset /app/Dataset

# Create cache and output directories with correct permissions
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/Train/output/cool_posters_lora

# Create non-root system user for production security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /venv

# Switch to non-root user
USER appuser

# Expose FastAPI application port
EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/ || exit 1

# Production command to start FastAPI app with Uvicorn
CMD ["python3", "-m", "uvicorn", "Api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
