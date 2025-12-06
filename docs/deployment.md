# Deployment Guide

This guide covers deploying the IoT Home Security system to a Raspberry Pi.

## Prerequisites

### Hardware
- Raspberry Pi 4 (2GB+ RAM) or Pi 5
- MicroSD card (32GB+)
- Pi Camera Module
- USB Microphone
- PIR Motion Sensor
- Buzzer and LEDs (optional)

### Software
- Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- Trained models (.tflite files)

## Step 1: Prepare Raspberry Pi OS

### Flash OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Select "Raspberry Pi OS (64-bit)"
3. Configure WiFi and SSH in settings
4. Flash to SD card

### Initial Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable required interfaces
sudo raspi-config
# → Interface Options → Camera → Enable
# → Interface Options → I2C → Enable
# Reboot when prompted
```

## Step 2: Transfer Project Files

### Option A: Git Clone

```bash
cd /home/pi
git clone https://github.com/yourusername/IoT-Personal-Home-Security.git security
cd security
```

### Option B: SCP from Development Machine

```bash
# From development machine
rsync -avz --exclude '.git' --exclude 'venv' --exclude '__pycache__' \
    --exclude 'ESC-50-master/audio' --exclude 'data/raw' \
    . pi@raspberrypi:/home/pi/security/
```

## Step 3: Run Installation Script

```bash
cd /home/pi/security
chmod +x raspberry_pi/scripts/install.sh
./raspberry_pi/scripts/install.sh
```

This script will:
- Install system dependencies
- Create Python virtual environment
- Install Python packages
- Setup systemd service

## Step 4: Deploy Trained Models

Transfer your trained models to the Pi:

```bash
# From development machine
scp data/models/face_detection/*.tflite pi@raspberrypi:/home/pi/security/models/
scp data/models/face_recognition/*.tflite pi@raspberrypi:/home/pi/security/models/
scp data/models/sound_classification/*.tflite pi@raspberrypi:/home/pi/security/models/
```

## Step 5: Add Known Faces

Transfer known face images:

```bash
# From development machine
scp -r data/raw/faces/known/* pi@raspberrypi:/home/pi/security/faces/known/
```

## Step 6: Configure the System

Edit the configuration file:

```bash
nano /home/pi/security/raspberry_pi/config/config.yaml
```

Key settings to adjust:

```yaml
# Camera settings
camera:
  resolution: [640, 480]  # Lower for better performance
  fps: 15

# Model paths
models:
  face_detection: "/home/pi/security/models/face_detection.tflite"
  face_recognition: "/home/pi/security/models/face_recognition.tflite"
  sound_classification: "/home/pi/security/models/audio_classifier.tflite"

# GPIO pins (adjust if needed)
gpio:
  motion_sensor: 17
  buzzer: 18
  led_alert: 25
```

## Step 7: Test the System

### Test Components Individually

```bash
cd /home/pi/security
source venv/bin/activate

# Test camera
python -c "from src.iot_home_security.sensors import CameraInterface; c = CameraInterface(use_picamera=True); c.start(); print('Camera OK'); c.stop()"

# Test microphone
python -c "from src.iot_home_security.sensors import MicrophoneInterface; m = MicrophoneInterface(); m.start(); print('Mic OK'); m.stop()"

# Test motion sensor
python -c "from src.iot_home_security.sensors import MotionSensor; s = MotionSensor(pin=17); print('Motion:', s.read())"
```

### Run Full System Test

```bash
python raspberry_pi/main.py --config raspberry_pi/config/config.yaml --debug
```

## Step 8: Start as Service

```bash
# Enable and start service
sudo systemctl enable security-system
sudo systemctl start security-system

# Check status
sudo systemctl status security-system

# View logs
journalctl -u security-system -f
```

## Step 9: Configure Auto-Start

The systemd service is configured to:
- Start automatically on boot
- Restart on failure
- Limit resource usage

Verify auto-start:
```bash
sudo reboot
# After reboot
sudo systemctl status security-system
```

## Updating the System

### Update Code

```bash
cd /home/pi/security
git pull origin main
sudo systemctl restart security-system
```

### Update Models

```bash
# Stop service
sudo systemctl stop security-system

# Replace models
cp /path/to/new/models/*.tflite /home/pi/security/models/

# Start service
sudo systemctl start security-system
```

## Monitoring

### Resource Usage

```bash
# Real-time monitoring
htop

# Memory usage
free -h

# Disk usage
df -h

# Temperature
vcgencmd measure_temp
```

### Application Logs

```bash
# Live logs
journalctl -u security-system -f

# Last 100 entries
journalctl -u security-system -n 100

# Errors only
journalctl -u security-system -p err
```

## Performance Tuning

### Reduce CPU Usage

```yaml
# In config.yaml
performance:
  frame_skip: 3  # Process every 3rd frame
  audio_buffer_seconds: 2.0  # Longer buffer, less frequent processing
```

### Reduce Memory Usage

```yaml
camera:
  resolution: [320, 240]  # Lower resolution
```

### Optimize for Coral TPU

If using Coral USB Accelerator:

```bash
# Install Edge TPU runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
journalctl -u security-system -n 50

# Test manually
cd /home/pi/security
source venv/bin/activate
python raspberry_pi/main.py --debug
```

### Camera Not Working

```bash
# Check camera connection
libcamera-hello

# Check for conflicts
sudo lsof /dev/video0
```

### High CPU Usage

```bash
# Check which process
top

# Increase frame skip or reduce resolution
# Edit raspberry_pi/config/config.yaml
```

### Memory Issues

```bash
# Check memory
free -h

# Increase swap (if needed)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```
