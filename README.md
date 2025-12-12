# IoT Personal Home Security System

A **threat detection** security system for Raspberry Pi using **face recognition** and **sound classification**.

> **Key Concept**: Detects people you want to be **alerted about** — by photo (watch list) or physical features (threat profile).

---

## Quick Start

### Docker (Recommended)

```bash
# Build Docker image
./run.sh build

# Start the system
./run.sh start

# Start API server only
./run.sh api

# Run tests
./run.sh test

# Face detection test
./run.sh detect

# Live camera detection
./run.sh detect --camera

# Interactive shell for debugging
./run.sh shell

# View logs
./run.sh logs --follow
```

### Using Make

```bash
make build     # Build Docker image
make start     # Start system
make api       # Start API
make test      # Run tests
make shell     # Debug shell
make help      # Show all commands
```

### Local Mode (without Docker)

```bash
# Use --local flag to run with Python venv instead of Docker
./run.sh start --local
./run.sh api --local
./run.sh test --local
```

**API:** http://localhost:8000  
**Docs:** http://localhost:8000/docs

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
# Inside container (docker compose run --rm app <command>)
start              # Start API server
detect             # Test face detection
detect --camera    # Live camera
detect -i img.jpg  # Single image
test               # Test components
test --all         # Include hardware
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
│   ├── __init__.py          # Re-exports (backwards compatible)
│   ├── visual/              # Face detection & recognition
│   │   ├── detection/       # Face detectors (opencv_dnn, haar, mediapipe, dlib)
│   │   ├── recognition/     # Face recognizers & embeddings
│   │   ├── pipeline.py      # FaceSecurityPipeline
│   │   └── utils.py         # Image utilities
│   ├── audio/               # Sound classification
│   │   ├── classifier.py    # SoundClassifier
│   │   ├── features.py      # Audio feature extraction
│   │   └── preprocessing.py # Audio preprocessing
│   ├── sensors/             # Hardware interfaces
│   │   └── camera/          # Camera capture (OpenCV, PiCamera)
│   ├── alerts.py            # Notifications
│   ├── api.py               # REST API
│   └── cli.py               # CLI
├── config/config.yaml       # Configuration
├── data/
│   ├── raw/faces/watch_list/  # Watch list photos
│   └── models/              # ML models
├── scripts/
│   └── camera_demo.py       # Live camera demo
├── docker-compose.yml
├── Dockerfile
└── run.sh                   # Universal run script
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

## Camera Access (Docker)

For live camera detection in Docker, the container runs with elevated privileges to access `/dev/video0`:

```bash
# This is handled automatically by run.sh
./run.sh detect --camera
```

**Note:** On systems with SELinux (Fedora/RHEL), camera access requires `--privileged --security-opt label=disable --user root`.

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
docker compose build
docker compose up
```

| Hardware | GPIO |
|----------|------|
| PIR Sensor | 17 |
| Buzzer | 18 |
| LED | 24 |

---

## License

MIT - see [LICENSE](LICENSE)
