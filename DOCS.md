# Documentation

Technical reference for the IoT Home Security system. Read README.md first for quick start.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Hardware Setup](#hardware-setup)
3. [Model Training](#model-training)
4. [Deployment](#deployment)
5. [Troubleshooting](#troubleshooting)

---

## Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                   YOUR DEVELOPMENT PC                        │
│                 (Windows/Linux x86_64)                       │
│                                                              │
│   Train models → Optimize → Export .tflite files            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   RASPBERRY PI 4/5                           │
│                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │   Camera    │   │ Microphone  │   │ PIR Sensor  │       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│          │                 │                 │               │
│          ▼                 ▼                 ▼               │
│   ┌─────────────────────────────────────────────────┐       │
│   │            PROCESSING (TFLite models)            │       │
│   │                                                  │       │
│   │  Face Detection → Face Recognition → Known?     │       │
│   │  Audio → Sound Classification → Threat?         │       │
│   └────────────────────────┬────────────────────────┘       │
│                            │                                 │
│                            ▼                                 │
│   ┌─────────────────────────────────────────────────┐       │
│   │                   ALERTS                         │       │
│   │   Buzzer | LED | Push Notification | SMS | MQTT │       │
│   └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

```
src/                      # Flat, simple structure
├── face.py               # FaceDetector, FaceRecognizer, DetectedFace, etc.
├── face_service.py       # FaceDatabase, FaceRecognizerService, processing
├── audio.py              # SoundClassifier, AudioPreprocessor, FeatureExtractor
├── sensors.py            # CameraInterface, MicrophoneInterface, MotionSensor
├── alerts.py             # LocalAlarm, NotificationManager, various notifiers
├── api.py                # FastAPI routes + Pydantic schemas
└── cli.py                # Command-line interface
```

### Key Classes

| Class | File | What It Does |
|-------|------|--------------|
| `FaceDetector` | `face.py` | Finds faces in images |
| `FaceRecognizer` | `face.py` | Matches faces to known people |
| `FaceSecurityPipeline` | `face.py` | Motion → detect → recognize flow |
| `FaceRecognizerService` | `face_service.py` | Manages face database & enrollment |
| `SoundClassifier` | `audio.py` | Classifies audio clips |
| `CameraInterface` | `sensors.py` | Captures video frames |
| `MotionSensor` | `sensors.py` | Detects motion via PIR |
| `LocalAlarm` | `alerts.py` | Controls buzzer/LED |
| `NotificationManager` | `alerts.py` | Sends push/SMS/MQTT alerts |

---

## Hardware Setup

### Parts List

| Component | Model | Purpose | Required? |
|-----------|-------|---------|-----------|
| Raspberry Pi | Pi 4 (4GB+) | Main computer | Yes |
| MicroSD Card | 32GB+ | Storage | Yes |
| Pi Camera | V2 or V3 | Face detection | Yes |
| USB Microphone | Any | Sound classification | Yes |
| PIR Sensor | HC-SR501 | Motion detection | Recommended |
| Buzzer | Active 5V | Local alarm | Optional |
| LED (Red) | 5mm | Alert indicator | Optional |
| LED (Green) | 5mm | Status indicator | Optional |

### Wiring

```
Raspberry Pi GPIO Pins
═══════════════════════════════════════════════════════

    PIR Sensor (HC-SR501)
    ├── VCC ────► Pin 2 (5V)
    ├── OUT ────► Pin 11 (GPIO 17)
    └── GND ────► Pin 6 (GND)

    Buzzer
    ├── + ──────► Pin 12 (GPIO 18) via 100Ω resistor
    └── - ──────► Pin 6 (GND)

    Alert LED (Red)
    ├── Anode ──► Pin 22 (GPIO 25) via 220Ω resistor
    └── Cathode ► Pin 25 (GND)

    Status LED (Green)
    ├── Anode ──► Pin 18 (GPIO 24) via 220Ω resistor
    └── Cathode ► Pin 25 (GND)

    Camera
    └── Ribbon cable into CSI port (blue side faces USB ports)

    Microphone
    └── USB port
```

### GPIO Summary

| Component | GPIO (BCM) | Physical Pin |
|-----------|------------|--------------|
| PIR Sensor | 17 | 11 |
| Buzzer | 18 | 12 |
| Alert LED | 25 | 22 |
| Status LED | 24 | 18 |

### Enable Camera

```bash
sudo raspi-config
# Interface Options → Camera → Enable
# Reboot
```

### Test Hardware

```bash
# Camera
libcamera-hello

# Microphone
arecord -l                    # List devices
arecord -d 5 test.wav         # Record 5 sec
aplay test.wav                # Play back
```

---

## Model Training

### Overview

Train on your PC (with GPU if possible), then export lightweight `.tflite` models for the Pi.

### Face Recognition

1. **Add face images:**
   ```
   data/raw/faces/watch_list/
   ├── alice/
   │   ├── img1.jpg
   │   ├── img2.jpg (5-10 images per person)
   │   └── ...
   ├── bob/
   │   └── ...
   ```

2. **Requirements per person:**
   - 5-10 clear photos
   - Various lighting
   - Front-facing preferred
   - At least 100x100 pixels for the face

3. **Train:**
   ```bash
   jupyter notebook notebooks/02_face_recognition_training.ipynb
   ```

### Sound Classification

Uses the ESC-50 dataset (included). Security-relevant sounds:

| Sound | Priority |
|-------|----------|
| glass_breaking | High |
| door_wood_knock | High |
| siren | High |
| gunshot | High |
| dog barking | Medium |
| crying_baby | Medium |

**Train:**
```bash
jupyter notebook notebooks/03_sound_classification_training.ipynb
```

### Export to TFLite

After training, convert to TFLite for the Pi:

```python
import tensorflow as tf

# Convert with optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Run optimization notebook:**
```bash
jupyter notebook notebooks/04_model_optimization.ipynb
```

### Expected Performance

| Model | Accuracy | Speed on Pi 4 |
|-------|----------|---------------|
| Face Detection | ~85% | ~30ms |
| Face Recognition | ~92% | ~80ms |
| Sound Classification | ~75% | ~150ms |

---

## Deployment

### Development Setup (Your PC)

```bash
# Clone
git clone https://github.com/yourusername/IoT-Personal-Home-Security.git
cd IoT-Personal-Home-Security

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install
pip install -r requirements.txt  # or requirements-arm64.txt for ARM
pip install -e .

# Test
./run.sh demo
```

### ARM64 VM Testing

```bash
# Use ARM64-specific requirements (no TensorFlow, uses TFLite runtime)
pip install -r requirements-arm64.txt
./run.sh demo
```

### Raspberry Pi Deployment

**Option 1: One-liner**
```bash
./scripts/install.sh --pi
```

**Option 2: Manual**
```bash
# On Pi
cd /home/pi
git clone <repo> security
cd security
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-arm64.txt
pip install -e .
```

### Transfer Models

From your dev machine:
```bash
scp data/models/*/*.tflite pi@raspberrypi:~/security/data/models/
scp -r data/raw/faces/watch_list/* pi@raspberrypi:~/security/data/raw/faces/watch_list/
```

### Run as Service

```bash
# Start manually
./run.sh start

# Or as systemd service
sudo systemctl enable security-system
sudo systemctl start security-system
sudo systemctl status security-system
```

### Configuration

Edit `config/config.yaml`:

```yaml
face_detection:
  enabled: true
  confidence_threshold: 0.8

sound_classification:
  enabled: true
  confidence_threshold: 0.7
  target_classes:
    - glass_breaking
    - door_wood_knock
    - siren

camera:
  resolution: [640, 480]
  fps: 15

alerts:
  local_alarm:
    enabled: true
    gpio_pin: 18
```

---

## Troubleshooting

### Camera Issues

```bash
# Test camera
libcamera-hello

# Check if detected
vcgencmd get_camera

# Check for processes using it
sudo lsof /dev/video0
```

### Audio Issues

```bash
# List audio devices
arecord -l

# Install pulseaudio if needed
sudo apt install pulseaudio
pactl list sources
```

### GPIO Permission Issues

```bash
sudo usermod -a -G gpio $USER
# Log out and back in
```

### Service Won't Start

```bash
# Check logs
journalctl -u security-system -n 50

# Test manually
cd ~/security
source .venv/bin/activate
python raspberry_pi/main.py --debug
```

### High CPU Usage

Edit `config/config.yaml`:
```yaml
performance:
  frame_skip: 3          # Process every 3rd frame
camera:
  resolution: [320, 240] # Lower resolution
```

### Out of Memory

```bash
# Check memory
free -h

# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Temperature Check

```bash
vcgencmd measure_temp
```

---

## Quick Reference

### Common Commands

```bash
./run.sh install        # Install dependencies
./run.sh start          # Start system
./run.sh stop           # Stop system
./run.sh status         # Check status
./run.sh demo           # Test all components
./run.sh test           # Run tests
./run.sh logs           # View logs
./run.sh api            # Start REST API only
```

### Systemd Commands

```bash
sudo systemctl start security-system
sudo systemctl stop security-system
sudo systemctl restart security-system
sudo systemctl status security-system
journalctl -u security-system -f       # Live logs
```

---

*For quick start, see [README.md](README.md)*
