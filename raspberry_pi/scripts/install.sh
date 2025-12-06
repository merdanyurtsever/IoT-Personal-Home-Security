#!/bin/bash
# =====================================================
# IoT Home Security - Raspberry Pi Installation Script
# =====================================================

set -e

echo "=========================================="
echo "IoT Home Security - Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This script is designed for Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    portaudio19-dev \
    libffi-dev \
    libssl-dev \
    git

# Install Pi Camera dependencies
echo -e "${GREEN}Installing Pi Camera dependencies...${NC}"
sudo apt-get install -y \
    python3-picamera2 \
    python3-libcamera

# Create project directory
PROJECT_DIR="/home/pi/security"
echo -e "${GREEN}Creating project directory at ${PROJECT_DIR}...${NC}"
mkdir -p ${PROJECT_DIR}
mkdir -p ${PROJECT_DIR}/models
mkdir -p ${PROJECT_DIR}/faces/known
mkdir -p ${PROJECT_DIR}/logs

# Create Python virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
cd ${PROJECT_DIR}
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install \
    numpy \
    opencv-python-headless \
    tflite-runtime \
    gpiozero \
    RPi.GPIO \
    pyyaml \
    paho-mqtt \
    librosa \
    soundfile \
    pyaudio

# Install picamera2 in venv
pip install picamera2

# Copy configuration
echo -e "${GREEN}Copying configuration files...${NC}"
# (This would copy from the repository)

# Setup systemd service
echo -e "${GREEN}Setting up systemd service...${NC}"
sudo cp /path/to/security-system.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable security-system

# Enable camera and GPIO
echo -e "${GREEN}Enabling camera and GPIO...${NC}"
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0

echo ""
echo -e "${GREEN}=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place trained models in ${PROJECT_DIR}/models/"
echo "2. Add known face images to ${PROJECT_DIR}/faces/known/"
echo "3. Edit configuration: ${PROJECT_DIR}/config/config.yaml"
echo "4. Start the service: sudo systemctl start security-system"
echo "5. View logs: journalctl -u security-system -f"
echo ""
echo -e "${YELLOW}Note: You may need to reboot for camera changes to take effect${NC}"
