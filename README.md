# IoT Personal Home Security System

A **threat detection** security system for Raspberry Pi using **face recognition** and **sound classification**.

> **Key Concept**: This system detects people you want to be **alerted about** — either by their photo (watch list) or by physical features (threat profile). It's designed to protect you by identifying predetermined threats, not to distinguish "known vs unknown" people.

**Target:** Raspberry Pi 4 (4GB+ RAM)  
**Dev:** Windows/Linux x86 PCs, ARM64 VMs for testing

---

## What It Does

- **Watch List Detection** → Identifies specific people you registered (by photo)
- **Threat Profile Detection** → Detects people matching attributes (glasses, beard, tattoos, etc.)
- **Sound Classification** → Detects glass breaking, door knocks, alarms
- **Motion Trigger** → PIR sensor wakes up camera on motion
- **Alerts** → Buzzer, LED, push notifications, SMS, MQTT

---

## Face Detection & Recognition Backends

### Face Detection (default: `opencv_dnn`)

| Backend | Description | Dependencies |
|---------|-------------|--------------|
| `opencv_dnn` | **Default** - SSD ResNet, works everywhere | OpenCV only |
| `haar_cascade` | Classic OpenCV Haar Cascades | OpenCV only |
| `mediapipe` | Google MediaPipe, fast on ARM64 | `pip install mediapipe` |

### Face Recognition/Embedding (default: `opencv_dnn`)

| Backend | Description | Dependencies |
|---------|-------------|--------------|
| `opencv_dnn` | **Default** - OpenFace 128D embeddings | OpenCV only |
| `mobilenetv2` | Keras MobileNetV2 512D embeddings | TensorFlow |
| `tflite` | TFLite model for edge deployment | tflite-runtime |
| `dlib` | dlib/face_recognition 128D | dlib (requires compilation) |

```python
from src.face import FaceDetector, FaceRecognizer

# Defaults work on x86, ARM64, and Raspberry Pi
detector = FaceDetector()  # Uses opencv_dnn
recognizer = FaceRecognizer()  # Uses opencv_dnn (OpenFace)

# Or explicitly specify backend
detector = FaceDetector(backend="mediapipe")
recognizer = FaceRecognizer(model="mobilenetv2")
```

---

## Detection Modes

| Mode | Description |
|------|-------------|
| `WATCH_LIST` | Match faces against photos you registered |
| `THREAT_PROFILE` | Match faces by attributes (glasses, beard, etc.) |
| `EMBEDDING_FIRST` | Try photo match first, fallback to attributes |
| `ATTRIBUTE_FIRST` | Try attributes first, fallback to photo match |
| `BOTH` | Run both methods, combine results |

### Example: Attribute-Based Detection

```python
from src.face import (
    FaceSecurityPipeline, DetectionMode, FaceDetector,
    AttributeFilter, FaceAttribute, HaarAttributeDetector
)

# Detect people with specific features (no photos needed)
pipeline = FaceSecurityPipeline(
    detector=FaceDetector(),
    mode=DetectionMode.ATTRIBUTE_ONLY,
    attribute_detector=HaarAttributeDetector()
)

# Alert when detecting: person with glasses + beard + tattoo
pipeline.add_attribute_filter(
    "threat_profile_1",
    AttributeFilter()
        .require(FaceAttribute.GLASSES)
        .require(FaceAttribute.BEARD)
        .require(FaceAttribute.TATTOO)
)
```

---

## Quick Start

### 1. Setup (Pick One)

```bash
# Development PC (Windows/Linux)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
pip install -e .

# ARM64 VM (for testing)
pip install -r requirements-arm64.txt
pip install -e .

# Raspberry Pi
./scripts/install.sh --pi
```

### 2. Run

```bash
./run.sh demo        # Test all components
./run.sh start       # Start security system
./run.sh api         # Start REST API only
```

---

## Project Structure

```
IoT-Personal-Home-Security/
│
├── main.py                    # Main entry point
├── src/                       # All source code (flat)
│   ├── face.py                # Face detection, recognition & attributes
│   ├── face_service.py        # Watch list database & processing
│   ├── audio.py               # Sound classification
│   ├── sensors.py             # Camera, microphone, PIR
│   ├── alerts.py              # Notifications & alarms
│   ├── api.py                 # REST API
│   └── cli.py                 # Command-line interface
│
├── config/                    # Configuration files
├── data/
│   ├── raw/faces/watch_list/  # Put photos of people to watch here
│   └── models/                # Trained .tflite models
│
├── notebooks/                 # Jupyter notebooks for training
├── tests/                     # Unit tests
└── ESC-50-master/             # Sound classification dataset
```

---

## How It Works

```
Motion Detected → Camera Activates → Face Detected?
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
               On Watch List?                    Matches Threat Profile?
                   OR                                  (attributes)
                        │                                   │
                        ▼                                   ▼
                   ALERT! 🚨                           ALERT! 🚨
```

---

## Commands

| Command | What It Does |
|---------|--------------|
| `./run.sh install` | Install dependencies |
| `./run.sh start` | Start security system |
| `./run.sh stop` | Stop security system |
| `./run.sh demo` | Test all components |
| `./run.sh test` | Run unit tests |
| `./run.sh api` | Start Face Management API |
| `./run.sh logs` | View logs |
| `./run.sh status` | Check if running |

---

## Add People to Watch List

### Method 1: By Photo (Watch List)

1. Create a folder with the person's name:
   ```
   data/raw/faces/watch_list/
   └── john_doe/
       ├── photo1.jpg
       ├── photo2.jpg
       └── ... (5-10 photos)
   ```

2. Train the model:
   ```bash
   jupyter notebook notebooks/02_face_recognition_training.ipynb
   ```

3. Export to TFLite:
   ```bash
   jupyter notebook notebooks/04_model_optimization.ipynb
   ```

### Method 2: By Attributes (Threat Profile)

No photos needed! Define physical features to detect:

```python
# In your code or config
pipeline.add_attribute_filter(
    "suspicious_profile",
    AttributeFilter()
        .require(FaceAttribute.SUNGLASSES)
        .require(FaceAttribute.HAT)
        .require(FaceAttribute.MASK)
)
```

**Available Attributes:**
- Eyewear: `GLASSES`, `SUNGLASSES`
- Facial Hair: `BEARD`, `MUSTACHE`
- Hair: `BALD`, `BLONDE_HAIR`, `BROWN_HAIR`, `BLACK_HAIR`, `RED_HAIR`, `GRAY_HAIR`
- Accessories: `HAT`, `MASK`, `TATTOO`
- Age: `YOUNG`, `MIDDLE_AGED`, `SENIOR`
- Gender: `MALE`, `FEMALE`

---

## Configuration

Edit `config/config.yaml`:

```yaml
face_detection:
  enabled: true
  confidence_threshold: 0.8

sound_classification:
  enabled: true
  target_classes:
    - glass_breaking
    - door_wood_knock
    - siren

alerts:
  local_alarm:
    enabled: true
    gpio_pin: 18
```

---

## Hardware (Raspberry Pi)

### Required
- Raspberry Pi 4 (4GB+)
- Pi Camera Module
- USB Microphone
- 32GB+ MicroSD

### Optional
- PIR Motion Sensor (GPIO 17)
- Buzzer (GPIO 18)
- LEDs (GPIO 24, 25)

See [DOCS.md](DOCS.md) for wiring diagrams.

---

## Deploy to Raspberry Pi

```bash
# On Pi
git clone <this-repo> ~/security
cd ~/security
./scripts/install.sh --pi

# Start
./run.sh start

# Or run as service (auto-starts on boot)
sudo systemctl enable security-system
sudo systemctl start security-system
```

---

## Testing

```bash
./run.sh test              # All tests
./run.sh test --coverage   # With coverage report
pytest tests/              # Direct pytest
```

---

## Full Documentation

See **[DOCS.md](DOCS.md)** for:
- Architecture details
- Hardware wiring diagrams
- Model training guide
- Deployment instructions
- Troubleshooting

---

## License

MIT License - see [LICENSE](LICENSE)
