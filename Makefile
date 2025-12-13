# IoT Home Security - Makefile
# ==============================
# Simple commands for development and running
#
# Usage:
#   make start     - Start API server
#   make detect    - Run face detection
#   make test      - Run tests
#   make help      - Show all commands

.PHONY: help start api detect camera test clean setup

# Default target
help:
	@echo "IoT Home Security - Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        Setup Python environment"
	@echo "  make install      Install dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make start        Start API server"
	@echo "  make detect       Test face detection"
	@echo "  make camera       Live camera detection"
	@echo "  make test         Run tests"
	@echo ""
	@echo "Face Module (standalone):"
	@echo "  make face-detect  Face detection"
	@echo "  make face-api     Face API server"
	@echo "  make face-test    Test face module"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Clean generated files"
	@echo "  make lint         Format and lint code"

# ============================================================
# SETUP
# ============================================================

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete. Run: source .venv/bin/activate"

install:
	pip install -r requirements.txt

# ============================================================
# RUNNING
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
# DEVELOPMENT
# ============================================================

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
