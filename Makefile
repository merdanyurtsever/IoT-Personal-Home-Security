# IoT Home Security - Makefile
# ============================
#
# Usage:
#   make install     # Install dependencies
#   make start       # Start the system
#   make api         # Start API server
#   make test        # Run tests
#   make demo        # Run demo
#   make help        # Show help
#

.PHONY: help install install-dev install-pi start stop restart api test demo clean train

# Default target
help:
	@echo "IoT Home Security - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install dependencies (auto-detect)"
	@echo "  make install-dev    Install for development"
	@echo "  make install-pi     Install for Raspberry Pi"
	@echo "  make install-arm64  Install for ARM64 VM"
	@echo ""
	@echo "Running:"
	@echo "  make start          Start the security system"
	@echo "  make start-debug    Start with debug output"
	@echo "  make stop           Stop the system"
	@echo "  make restart        Restart the system"
	@echo "  make api            Start API server only"
	@echo ""
	@echo "Training:"
	@echo "  make train          Show training options"
	@echo "  make train-face     Train face recognition"
	@echo "  make train-sound    Train sound classification"
	@echo "  make train-all      Train all models"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make demo           Run interactive demo"
	@echo "  make detect         Test face detection"
	@echo "  make recognize      Test face recognition"
	@echo "  make classify       Test sound classification"
	@echo "  make camera         Test camera"
	@echo ""
	@echo "Maintenance:"
	@echo "  make logs           View logs"
	@echo "  make status         Check service status"
	@echo "  make clean          Clean up generated files"
	@echo ""

# Installation targets
install:
	@./run.sh install

install-dev:
	@./run.sh install --dev

install-pi:
	@./run.sh install --pi

install-arm64:
	@./run.sh install --arm64

# Run targets
start:
	@./run.sh start

start-debug:
	@./run.sh start --debug

stop:
	@./run.sh stop

restart:
	@./run.sh restart

api:
	@./run.sh api

# Test targets
test:
	@./run.sh test

test-cov:
	@./run.sh test --coverage

demo:
	@./run.sh demo

detect:
	@./run.sh detect

recognize:
	@./run.sh recognize

classify:
	@./run.sh classify

camera:
	@./run.sh camera

# Training targets
train:
	@./run.sh train

train-face:
	@./run.sh train --face

train-sound:
	@./run.sh train --sound

train-all:
	@./run.sh train --all

# Utility targets
logs:
	@./run.sh logs

status:
	@./run.sh status

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true
	@echo "Done!"

# Quick commands for common operations
run: start
dev: install-dev
pi: install-pi

# Docker targets
.PHONY: docker-build docker-run docker-dev docker-stop docker-clean docker-shell

docker-build:
	@echo "Building Docker image (Python 3.11 + ML packages)..."
	docker build -t iot-home-security:latest .

docker-build-pi:
	@echo "Building Docker image for Raspberry Pi..."
	docker build -f Dockerfile.pi -t iot-home-security:pi .

docker-run:
	@echo "Running security system in Docker..."
	docker-compose up security-app

docker-dev:
	@echo "Starting development container with camera access..."
	xhost +local:docker 2>/dev/null || true
	docker-compose run --rm dev

docker-camera:
	@echo "Running camera comparison in Docker..."
	xhost +local:docker 2>/dev/null || true
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-v /dev/video0:/dev/video0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		--device /dev/video0 \
		iot-home-security:latest \
		python scripts/camera_model_comparison.py

docker-api:
	@echo "Starting API server in Docker..."
	docker-compose up api

docker-stop:
	docker-compose down

docker-shell:
	docker run -it --rm iot-home-security:latest /bin/bash

docker-clean:
	docker-compose down --rmi local --volumes --remove-orphans
	docker image prune -f
