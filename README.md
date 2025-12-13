# IoT Personal Home Security System

A **threat detection** security system for Raspberry Pi using **face recognition** and **sound classification**.

> **Key Concept**: Detects people you want to be **alerted about** — by photo (watch list) or physical features (threat profile).

---

## Quick Start

```bash
# Setup environment
./run.sh setup

# Start the API server
./run.sh start

# Test face detection
./run.sh detect

# Live camera detection
./run.sh detect --camera

# Run tests
./run.sh test
```

**API:** http://localhost:8000  
**Docs:** http://localhost:8000/docs

---

## Distrobox (Optional)

For isolated development or deployment:

```bash
# Create container
distrobox assemble create --file distrobox.ini

# Enter container
distrobox enter iot-security

# Inside container, run as usual
./run.sh setup
./run.sh detect --camera
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Watch List** | Alert when specific people (by photo) are detected |
| **Threat Profile** | Alert by physical attributes (glasses, beard, etc.) |
| **Sound Detection** | Glass breaking, door knocks, alarms |
| **Motion Trigger** | PIR sensor activates camera |

---

## Face Backends

### Detectors

| Backend | Description |
|---------|-------------|
| `opencv_dnn` | SSD ResNet (default) |
| `haar_cascade` | Fast, lightweight |
| `mediapipe` | Good for ARM64 |
| `dlib` | Accurate + landmarks |

### Recognizers

| Backend | Embedding |
|---------|-----------|
| `opencv_dnn` | 128D (default) |
| `dlib` | 128D |
| `mobilenetv2` | 512D |
| `tflite` | 512D |

```python
from src import FaceDetector, FaceRecognizer

detector = FaceDetector(backend="opencv_dnn")
recognizer = FaceRecognizer(embedding_backend="opencv_dnn")
```

---

## CLI Commands

```bash
./run.sh start              # Start API server
./run.sh detect             # Test face detection
./run.sh detect --camera    # Live camera
./run.sh detect -i img.jpg  # Single image
./run.sh test               # Run tests
./run.sh face detect        # Use face module directly
./run.sh face api           # Face module API
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/faces` | GET | List faces |
| `/api/v1/faces` | POST | Upload face |
| `/api/v1/faces/{id}` | DELETE | Remove face |
| `/api/v1/recognize` | POST | Recognize face |

---

## Project Structure

```
├── src/
│   ├── face/                # 📦 INDEPENDENT Face Module (can be shared)
│   │   ├── __init__.py      # Public API
│   │   ├── __main__.py      # Run as: python -m src.face
│   │   ├── cli.py           # Standalone CLI
│   │   ├── api.py           # Standalone REST API
│   │   ├── detection/       # Face detectors
│   │   ├── recognition/     # Face recognizers & embeddings
│   │   ├── pipeline.py      # FaceSecurityPipeline
│   │   └── utils.py         # Image utilities
│   ├── audio/               # Sound classification
│   ├── sensors/             # Hardware interfaces
│   ├── api.py               # Main system API
│   └── cli.py               # Main system CLI
├── config/config.yaml       # Configuration
├── data/
│   ├── raw/faces/watch_list/  # Watch list photos
│   └── models/              # ML models
├── distrobox.ini            # Distrobox container config
├── requirements.txt         # Python dependencies
└── run.sh                   # Run script
```

### Face Module (Independent)

The `src/face/` module can be used independently:

```bash
# Run standalone
python -m src.face detect --image photo.jpg
python -m src.face detect --camera
python -m src.face api --port 8000
python -m src.face test

# Or use in code
from src.face import FaceDetector, FaceRecognizer

detector = FaceDetector()
recognizer = FaceRecognizer()
```

---

## Configuration

`config/config.yaml`:

```yaml
face_detection:
  backend: opencv_dnn  # haar_cascade, mediapipe, dlib

face_recognition:
  embedding_backend: opencv_dnn  # dlib, tflite, mobilenetv2
  similarity_threshold: 0.6

api:
  port: 8000
```

---

## Add Watch List

**File system:**
```
data/raw/faces/watch_list/
└── person_name/
    ├── photo1.jpg
    └── photo2.jpg
```

**API:**
```bash
curl -X POST http://localhost:8000/faces \
  -F "name=John" -F "image=@photo.jpg"
```

---

## Raspberry Pi

```bash
./run.sh setup
./run.sh start
```

| Hardware | GPIO |
|----------|------|
| PIR Sensor | 17 |
| Buzzer | 18 |
| LED | 24 |

---

## License

MIT - see [LICENSE](LICENSE)
