# IoT Home Security - Makefile
# ==============================
# Simple commands for development and running
#
# Usage:
#   make start     - Start API server
#   make detect    - Run face detection
#   make test      - Run tests
#   make help      - Show all commands

.PHONY: help start api detect camera test clean docker-build docker-run

# Default target
help:
	@echo "IoT Home Security - Commands"
	@echo "============================"
	@echo ""
	@echo "Running:"
	@echo "  make start        Start API server"
	@echo "  make detect       Test face detection"
	@echo "  make camera       Live camera detection"
	@echo "  make test         Run tests"
	@echo ""
	@echo "Face Module (standalone):"
	@echo "  make face-detect  Face detection (standalone module)"
	@echo "  make face-api     Face API server only"
	@echo "  make face-test    Test face module"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build container"
	@echo "  make docker-run   Run in container"
	@echo "  make docker-shell Shell in container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Clean generated files"
	@echo "  make install      Install dependencies"

# ============================================================
# RUNNING (Local)
# ============================================================

start:
	python -m src.cli start

api:
	python -m src.cli start

detect:
	python -m src.cli detect

camera:
	python -m src.cli detect --camera

test:
	python -m pytest tests/ -v

# ============================================================
# FACE MODULE (Standalone)
# ============================================================

face-detect:
	python -m src.face detect

face-camera:
	python -m src.face detect --camera

face-api:
	python -m src.face api

face-test:
	python -m src.face test

# ============================================================
# DOCKER
# ============================================================

docker-build:
	docker build -f docker/Dockerfile -t iot-home-security .

docker-run:
	docker run -p 8000:8000 -v ./data:/app/data iot-home-security

docker-shell:
	docker run -it --rm iot-home-security bash

docker-camera:
	docker run -it --rm --device /dev/video0 \
		-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix \
		iot-home-security python -m src.face detect --camera

# ============================================================
# DEVELOPMENT
# ============================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8

lint:
	black src/ tests/
	flake8 src/ tests/

# ============================================================
# CLEANUP
# ============================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true
	rm -rf *.egg-info/ build/ dist/ 2>/dev/null || true
	@echo "Cleaned!"
