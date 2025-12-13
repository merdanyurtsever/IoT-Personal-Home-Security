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
src/                          # Main source code
├── __init__.py               # Package exports
├── constants.py              # Centralized config & constants
├── alerts.py                 # Notification & alarm management
├── api.py                    # FastAPI REST endpoints
├── cli.py                    # Command-line interface
├── face/                     # Face detection & recognition (standalone module)
│   ├── __init__.py           # Module exports
│   ├── __main__.py           # Entry point (python -m src.face)
│   ├── cli.py                # Standalone face CLI
│   ├── api.py                # Standalone face API
│   ├── face_detector.py      # Multi-backend face detection
│   ├── face_recognizer.py    # Face embedding & matching
│   ├── pipeline.py           # Detection → Recognition flow
│   ├── utils.py              # Image preprocessing utilities
│   ├── detection/            # Detection backends
│   │   ├── base.py           # Abstract detector interface
│   │   ├── opencv_dnn.py     # OpenCV DNN (SSD) detector
│   │   ├── haar.py           # Haar cascade detector
│   │   ├── mediapipe.py      # MediaPipe face detector
│   │   └── dlib.py           # dlib HOG/CNN detector
│   └── recognition/          # Recognition backends
│       ├── database.py       # Face database management
│       ├── embeddings/       # Embedding extractors
│       └── types.py          # Data types
├── audio/                    # Sound classification
│   ├── classifier.py         # Multi-format model loader
│   ├── features.py           # MFCC/Mel spectrogram extraction
│   └── preprocessing.py      # Audio loading & normalization
└── sensors/                  # Hardware interfaces
    └── camera/               # Camera capture
```

### Key Classes

| Class | File | What It Does |
|-------|------|--------------|
| `FaceDetector` | `face/face_detector.py` | Multi-backend face detection |
| `FaceRecognizer` | `face/face_recognizer.py` | Embedding extraction & matching |
| `OpenCVDNNDetector` | `face/detection/opencv_dnn.py` | SSD-based face detection |
| `SoundClassifier` | `audio/classifier.py` | Classifies audio (TFLite, ONNX, sklearn) |
| `FeatureExtractor` | `audio/features.py` | MFCC & Mel spectrogram extraction |
| `AudioPreprocessor` | `audio/preprocessing.py` | Audio loading & normalization |
| `Config` | `constants.py` | Centralized configuration loader |
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
pip install -r requirements.txt

# Test
./run.sh demo
```

### ARM64 / Raspberry Pi Deployment

```bash
# Install dependencies (same requirements file works on all platforms)
pip install -r requirements.txt
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

## API Documentation

### Overview

The REST API runs on port 8000 by default and provides endpoints for face management, detection, and system status.

### Starting the API

```bash
./run.sh api              # Start API server
./run.sh api --reload     # Start with auto-reload (development)
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/faces` | List registered faces |
| POST | `/api/v1/faces/register` | Register a new face |
| DELETE | `/api/v1/faces/{name}` | Remove a registered face |
| POST | `/api/v1/detect` | Detect faces in uploaded image |
| POST | `/api/v1/recognize` | Recognize faces in uploaded image |
| GET | `/api/v1/status` | System status |

### Request Examples

**Register a Face:**
```bash
curl -X POST http://localhost:8000/api/v1/faces/register \
  -F "name=john_doe" \
  -F "image=@/path/to/face.jpg"
```

**Detect Faces:**
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "image=@/path/to/image.jpg"
```

**Response Format:**
```json
{
  "success": true,
  "faces": [
    {
      "bbox": [100, 50, 200, 200],
      "confidence": 0.95,
      "identity": "john_doe"
    }
  ]
}
```

### Authentication

Authentication is disabled by default. To enable:

```yaml
# config/config.yaml
api:
  auth:
    enabled: true
    method: "api_key"  # Options: none, api_key
```

---

## Security Considerations

### Data Privacy

- **Local Processing**: All face detection and recognition runs locally on your device
- **No Cloud Required**: The system works entirely offline after initial setup
- **Embedding Storage**: Face embeddings (not images) are stored in the local database
- **GDPR Compliance**: For EU deployments, ensure consent for biometric data collection

### Network Security

- **Local Network**: By default, the API binds to `0.0.0.0` - restrict to localhost in production
- **CORS**: Configure `cors_origins` in config.yaml to limit allowed origins
- **HTTPS**: Use a reverse proxy (nginx) with TLS for production deployments

```yaml
# config/config.yaml - Production settings
api:
  host: "127.0.0.1"  # Bind to localhost only
  cors_origins: ["https://yourdomain.com"]
```

### Credential Management

- **Firebase Config**: Store `config/firebase_config.json` securely (not in git)
- **MQTT Credentials**: Set in config.yaml, consider environment variables for production
- **API Keys**: Generate strong random keys if using API authentication

### File Permissions

```bash
# Restrict config file access
chmod 600 config/config.yaml
chmod 600 config/firebase_config.json

# Restrict face database
chmod 700 data/raw/faces/watch_list
```

---

## Logging & Monitoring

### Log Files

| File | Contents |
|------|----------|
| `logs/security.log` | Main application log |
| `logs/api.log` | API request/response logs |
| `logs/alerts.log` | Alert history |

### Log Configuration

Edit `config/logging.yaml` to customize log levels:

```yaml
version: 1
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/security.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  src:
    level: INFO
  src.face:
    level: DEBUG  # More verbose for debugging
```

### Log Levels

| Level | When to Use |
|-------|-------------|
| DEBUG | Detailed debugging (high volume) |
| INFO | Normal operation events |
| WARNING | Unexpected but handled situations |
| ERROR | Errors that need attention |
| CRITICAL | System failures |

### Viewing Logs

```bash
./run.sh logs              # View recent logs
./run.sh logs --follow     # Stream logs in real-time
journalctl -u security-system -f  # Systemd service logs
```

### Metrics (Future)

Prometheus metrics endpoint planned for future release:
- Face detection counts and latency
- Recognition match rates
- Alert frequencies
- System resource usage

---

## Development Guide

### Code Style

- **Python Version**: 3.9+
- **Formatter**: Black with default settings
- **Linter**: Ruff (replaces flake8/isort)
- **Type Hints**: Required for all public functions

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Running Tests

```bash
./run.sh test              # Run all tests
./run.sh test --coverage   # With coverage report
pytest tests/ -v           # Verbose output
pytest tests/test_face_detection.py -k "test_opencv"  # Specific test
```

### Project Structure

```
├── src/                   # Main source code
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks for training
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── data/                  # Models and data
│   ├── models/           # Pre-trained models
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data
└── logs/                  # Application logs
```

### Adding a New Detection Backend

1. Create `src/face/detection/mybackend.py`
2. Inherit from `BaseFaceDetector`
3. Implement `detect(image) -> List[DetectedFace]`
4. Register in `src/face/detection/__init__.py`

```python
from .base import BaseFaceDetector
from .types import DetectedFace

class MyBackendDetector(BaseFaceDetector):
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        # Your implementation
        pass
```

### Configuration Constants

Processing constants are centralized in `src/constants.py` and loaded from `config/config.yaml`. To add new constants:

1. Add defaults to `config/config.yaml` under appropriate section
2. Add to the corresponding dataclass in `src/constants.py`
3. Use via `get_*_config()` functions

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `./run.sh test`
5. Format code: `black src/ tests/`
6. Submit a pull request

---

*For quick start, see [README.md](README.md)*
