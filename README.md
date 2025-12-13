# IoT Personal Home Security System

A **threat detection** security system for Raspberry Pi using **ArcFace face recognition** and **sound classification**.

> **Key Concept**: Detects people you want to be **alerted about** — by photo (watch list).

---

## Quick Start

```bash
# Setup environment
./run.sh setup

# Run face recognition viewfinder
python -m src.face

# With custom watch list
python -m src.face --watchlist /path/to/faces --threshold 0.35

# Run tests
./run.sh test
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Watch List** | Alert when specific people (by photo) are detected |
| **ArcFace Recognition** | State-of-the-art 512D embeddings (InsightFace buffalo_l) |
| **Brightness Enhancement** | CLAHE preprocessing for dim lighting |
| **Sound Detection** | Glass breaking, door knocks, alarms |

---

## Face Recognition

Uses **ArcFace** via InsightFace's `buffalo_l` model (ResNet-50, 512D embeddings).

```python
from src.face import ArcFaceRecognizer, FaceDatabase

# Initialize
recognizer = ArcFaceRecognizer(model_name="buffalo_l")
database = FaceDatabase()

# Add to watch list
import cv2
image = cv2.imread("suspect.jpg")
faces = recognizer.detect(image)
if faces:
    embedding = recognizer.extract_embedding(image, faces[0])
    database.add("suspect_name", embedding)

# Process live frame
matches = []
for face in recognizer.detect(frame):
    emb = recognizer.extract_embedding(frame, face)
    results = database.search(emb, threshold=0.35)
    matches.extend(results)
```

---

## Viewfinder Controls

| Key | Action |
|-----|--------|
| `B` | Toggle brightness enhancement |
| `R` | Re-enroll faces from watch list |
| `+`/`-` | Adjust recognition threshold |
| `Q`/`ESC` | Quit |

---

## Project Structure

```
├── src/
│   ├── face/                # 📦 Face Recognition Module
│   │   ├── viewfinder.py    # ArcFace recognizer + live viewfinder
│   │   ├── __init__.py      # Public API
│   │   ├── __main__.py      # Run as: python -m src.face
│   │   ├── requirements.txt # Module dependencies
│   │   └── README.md        # Module documentation
│   ├── audio/               # Sound classification
│   ├── sensors/             # Hardware interfaces
│   ├── api.py               # Main system API
│   └── cli.py               # Main system CLI
├── config/config.yaml       # Configuration
├── data/
│   ├── raw/faces/watch_list/  # Watch list photos
│   └── models/              # ML models
├── requirements.txt         # Python dependencies
└── run.sh                   # Run script
```

---

## Add Watch List

Place photos in `data/raw/faces/watch_list/`:

```
data/raw/faces/watch_list/
├── suspect1.jpg
├── suspect2.jpg
└── person_name.png
```

Filenames become the identity labels (without extension).

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
