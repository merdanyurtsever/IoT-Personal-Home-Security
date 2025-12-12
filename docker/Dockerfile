# IoT Home Security - Container Image
# =====================================
# Single Dockerfile for all platforms (x86_64, ARM64/Raspberry Pi)
# 
# Usage:
#   docker build -t iot-security .                    # Build image
#   docker run -p 8000:8000 iot-security              # Run API server
#   docker run iot-security python -m src.face test  # Test face module
#
# For Raspberry Pi, build on the Pi itself or use buildx:
#   docker buildx build --platform linux/arm64 -t iot-security .

FROM python:3.11-slim-bookworm

LABEL maintainer="IoT Home Security Project"
LABEL description="Face detection & recognition for home security"

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (minimal set that works on all platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    # Camera support
    libv4l-0 \
    v4l-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Copy requirements and install (layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories
RUN mkdir -p data/models data/raw/faces/watch_list logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; print('OK')" || exit 1

# Default command: start API server
CMD ["python", "-m", "src.cli", "start"]
