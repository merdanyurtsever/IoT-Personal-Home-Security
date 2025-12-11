# IoT Home Security - Makefile (Container-First)
# ================================================
#
# All commands run inside Docker containers by default.
# Use `make <cmd>-local` targets to run with local venv.
#
# Usage:
#   make build       # Build Docker image
#   make start       # Start system in container
#   make api         # Start API in container
#   make test        # Run tests in container
#   make shell       # Debug shell in container
#   make help        # Show help
#

.PHONY: help build start stop restart status api test demo shell logs clean
.PHONY: detect recognize classify camera train
.PHONY: start-local api-local test-local install install-dev install-pi

# Default target
help:
	@echo "IoT Home Security - Container-First Commands"
	@echo "============================================="
	@echo ""
	@echo "Docker Commands (default):"
	@echo "  make build          Build Docker image"
	@echo "  make start          Start system in container"
	@echo "  make stop           Stop containers"
	@echo "  make restart        Restart containers"
	@echo "  make status         Check container status"
	@echo "  make api            Start API server in container"
	@echo "  make shell          Open shell in container"
	@echo "  make logs           View container logs"
	@echo ""
	@echo "Testing (in container):"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make demo           Run interactive demo"
	@echo "  make detect         Test face detection"
	@echo "  make detect-camera  Test with camera"
	@echo "  make recognize      Test face recognition"
	@echo "  make classify       Test sound classification"
	@echo "  make camera         Test camera interface"
	@echo ""
	@echo "Training (in container):"
	@echo "  make train-face     Train face recognition"
	@echo "  make train-sound    Train sound classification"
	@echo ""
	@echo "Local Mode (uses venv instead of Docker):"
	@echo "  make start-local    Start with local venv"
	@echo "  make api-local      API with local venv"
	@echo "  make test-local     Tests with local venv"
	@echo "  make install        Install to local venv"
	@echo "  make install-dev    Install dev dependencies"
	@echo "  make install-pi     Install for Raspberry Pi"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean up generated files"
	@echo "  make docker-clean   Remove containers/images"
	@echo ""

# ============================================================
# DOCKER COMMANDS (Primary - run everything in containers)
# ============================================================

build:
	@./run.sh build

start:
	@./run.sh start

stop:
	@./run.sh stop

restart:
	@./run.sh restart

status:
	@./run.sh status

api:
	@./run.sh api

shell:
	@./run.sh shell

logs:
	@./run.sh logs --follow

# Testing in container
test:
	@./run.sh test

test-cov:
	@./run.sh test --coverage

demo:
	@./run.sh demo

detect:
	@./run.sh detect

detect-camera:
	@./run.sh detect --camera

recognize:
	@./run.sh recognize

classify:
	@./run.sh classify

camera:
	@./run.sh camera

# Training in container
train-face:
	@./run.sh train --face

train-sound:
	@./run.sh train --sound

train-all:
	@./run.sh train --all

# ============================================================
# LOCAL MODE (uses venv, for when Docker isn't available)
# ============================================================

start-local:
	@./run.sh start --local

api-local:
	@./run.sh api --local

test-local:
	@./run.sh test --local

demo-local:
	@./run.sh demo --local

# Installation (for local mode)
install:
	@./run.sh install

install-dev:
	@./run.sh install --dev

install-pi:
	@./run.sh install --pi

# ============================================================
# MAINTENANCE
# ============================================================

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true
	@rm -rf .venv/ venv/ 2>/dev/null || true
	@echo "Done!"

docker-clean:
	@echo "Cleaning Docker resources..."
	docker compose down --rmi local --volumes --remove-orphans 2>/dev/null || true
	docker image prune -f
	@echo "Done!"

# Quick aliases
run: start
dev: start
pi: install-pi
