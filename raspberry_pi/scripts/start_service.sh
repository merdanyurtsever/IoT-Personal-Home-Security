#!/bin/bash
# =====================================================
# IoT Home Security - Startup Script
# =====================================================

set -e

PROJECT_DIR="/home/pi/security"
VENV_PATH="${PROJECT_DIR}/venv"
MAIN_SCRIPT="${PROJECT_DIR}/raspberry_pi/main.py"
CONFIG_PATH="${PROJECT_DIR}/raspberry_pi/config/config.yaml"
LOG_DIR="${PROJECT_DIR}/logs"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Activate virtual environment
source ${VENV_PATH}/bin/activate

# Export environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Run the main application
echo "Starting IoT Home Security System..."
exec python3 ${MAIN_SCRIPT} --config ${CONFIG_PATH} 2>&1 | tee -a ${LOG_DIR}/security.log
