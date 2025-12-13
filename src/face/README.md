# Face Recognition Module

A self-contained face recognition module using **ArcFace** (InsightFace buffalo_l model).

## Features

- **Real-time face detection** using RetinaFace
- **512D face embeddings** using ArcFace (ResNet-50 backbone)
- **Brightness enhancement** (CLAHE) for dim environments
- **Watch list matching** for threat detection
- **Simple API** for integration

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Note:** On some systems, `insightface` may require a C++ compiler. If installation fails:
```bash
# Set compiler (if needed)
export CXX=/path/to/g++
pip install insightface
```

### Run Viewfinder

```bash
# From project root
python -m src.face.viewfinder

# With custom watch list
python -m src.face.viewfinder --watch-list /path/to/faces

# Adjust threshold (lower = stricter matching)
python -m src.face.viewfinder --threshold 0.4
```

### Viewfinder Controls

| Key | Action |
|-----|--------|
| B | Toggle brightness enhancement |
| R | Re-enroll faces from watch list |
| +/- | Adjust recognition threshold |
| Q/ESC | Quit |

## API Usage

```python
from src.face.viewfinder import ArcFaceRecognizer, FaceDatabase, enhance_brightness

# Initialize recognizer
recognizer = ArcFaceRecognizer(model_name="buffalo_l")

# Load and process an image
import cv2
image = cv2.imread("photo.jpg")
image = enhance_brightness(image)  # Optional: enhance dim images

# Detect faces
faces = recognizer.detect_faces(image)
for (x, y, w, h) in faces:
    face_crop = image[y:y+h, x:x+w]
    
    # Extract embedding
    embedding = recognizer.extract_embedding(face_crop)
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")  # (512,)

# Build a face database
db = FaceDatabase()
db.add("person1", embedding1)
db.add("person2", embedding2)

# Match a query face
match_name, score = db.find_match(query_embedding, threshold=0.35)
if match_name:
    print(f"Match: {match_name} ({score:.0%})")
```

## Model Options

| Model | Accuracy | Speed | Size |
|-------|----------|-------|------|
| buffalo_l | Highest | Slower | ~280MB |
| buffalo_s | Good | Fast | ~120MB |
| buffalo_sc | Basic | Fastest | ~15MB |

Default is `buffalo_l` for best accuracy.

## Watch List

Place face images in a directory (default: `data/raw/faces/watch_list/`).

Supported formats: `.jpg`, `.jpeg`, `.png`

The system will:
1. Detect faces in each image
2. Extract embeddings
3. Match live camera faces against enrolled faces

## Threshold Guidelines

| Threshold | Description |
|-----------|-------------|
| 0.25-0.30 | Very strict (high security) |
| 0.35 | **Default** - balanced |
| 0.40-0.50 | Lenient (more matches) |

## Files

```
src/face/
├── __init__.py         # Module exports
├── viewfinder.py       # Main viewfinder application
├── requirements.txt    # Dependencies
└── README.md           # This file
```
