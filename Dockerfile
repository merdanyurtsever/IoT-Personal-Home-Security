# IoT Personal Home Security - Development & Deployment Container
# Uses Python 3.11 for maximum ML package compatibility
# Works on both x86_64 (dev) and ARM64 (Raspberry Pi 4)

FROM python:3.11-slim-bookworm

LABEL maintainer="IoT Home Security Project"
LABEL description="Face detection/recognition for embedded home security"
LABEL python.version="3.11"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Video/camera support
    libv4l-0 \
    v4l-utils \
    # Build tools (for some pip packages)
    build-essential \
    cmake \
    # Useful utilities
    wget \
    curl \
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements-docker.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data/models/face_detection \
             /app/data/models/face_recognition \
             /app/data/raw/faces/watch_list \
             /app/logs \
    && chown -R appuser:appuser /app

# Download models during build (optional, can also be mounted)
# RUN python -c "from src.face import OpenCVDNNDetector; OpenCVDNNDetector()"

# Switch to non-root user
USER appuser

# Expose port for API (if used)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; import numpy; print('OK')" || exit 1

# Default command - can be overridden
CMD ["python", "-m", "src.cli", "test"]
