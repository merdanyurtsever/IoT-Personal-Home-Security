# Raspberry Pi Deployment

This directory contains all files needed to deploy the IoT Home Security system on a Raspberry Pi.

## Directory Structure

```
raspberry_pi/
├── main.py                     # Main entry point
├── config/
│   └── config.yaml             # RPi-specific configuration
├── scripts/
│   ├── install.sh              # Installation script
│   └── start_service.sh        # Service startup script
└── systemd/
    └── security-system.service # Systemd service file
```

## Quick Start

### 1. Transfer Files to Raspberry Pi

```bash
# From development machine
rsync -avz --exclude '.git' --exclude 'venv' --exclude '__pycache__' \
    . pi@raspberrypi:/home/pi/security/
```

### 2. Run Installation Script

```bash
# On Raspberry Pi
cd /home/pi/security
chmod +x raspberry_pi/scripts/install.sh
./raspberry_pi/scripts/install.sh
```

### 3. Configure the System

Edit the configuration file:
```bash
nano /home/pi/security/raspberry_pi/config/config.yaml
```

### 4. Add Models and Known Faces

```bash
# Copy trained models
cp models/*.tflite /home/pi/security/models/

# Add known face images
cp -r known_faces/* /home/pi/security/faces/known/
```

### 5. Start the Service

```bash
# Manual start
./raspberry_pi/scripts/start_service.sh

# Or using systemd
sudo systemctl start security-system
sudo systemctl status security-system
```

## Hardware Setup

### GPIO Pin Configuration

| Component | BCM Pin | Physical Pin |
|-----------|---------|--------------|
| PIR Sensor | GPIO 17 | Pin 11 |
| Buzzer | GPIO 18 | Pin 12 |
| Alert LED | GPIO 25 | Pin 22 |
| Status LED | GPIO 24 | Pin 18 |

### Wiring Diagram

```
Raspberry Pi GPIO
    ┌─────────────────────────────────────┐
    │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   │
    │  1 2 3 4 5 6 7 8 9 ...              │
    │                                      │
    │  Pin 11 (GPIO17) ──► PIR Sensor     │
    │  Pin 12 (GPIO18) ──► Buzzer         │
    │  Pin 22 (GPIO25) ──► Alert LED      │
    │  Pin 18 (GPIO24) ──► Status LED     │
    │                                      │
    │  Pin 2  (5V)     ──► PIR VCC        │
    │  Pin 6  (GND)    ──► Common Ground  │
    └─────────────────────────────────────┘
```

## Monitoring

### View Logs

```bash
# Real-time logs
journalctl -u security-system -f

# Last 100 lines
journalctl -u security-system -n 100

# Application logs
tail -f /home/pi/security/logs/security.log
```

### Check Status

```bash
# Service status
sudo systemctl status security-system

# Resource usage
htop
```

## Troubleshooting

### Camera Not Working

```bash
# Test camera
libcamera-hello

# Check if camera is enabled
sudo raspi-config nonint get_camera
```

### Audio Issues

```bash
# List audio devices
arecord -l

# Test microphone
arecord -d 5 test.wav && aplay test.wav
```

### GPIO Permission Issues

```bash
# Add user to gpio group
sudo usermod -a -G gpio pi
```
