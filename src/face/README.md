# Face Recognition Module (ArcFace)

Self-contained face recognition using InsightFace's ArcFace model (512D embeddings).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a watch list folder with face images
mkdir watch_list
cp /path/to/known/faces/*.jpg watch_list/

# Run
python -m src.face
```

## Directory Structure

The module auto-detects watch list from these locations (in order):

| Path | Description |
|------|-------------|
| `watch_list/` | Current directory |
| `faces/` | Alternative name |
| `data/watch_list/` | Data subdirectory |
| `data/raw/faces/watch_list/` | Full project path |

Or specify with `-w /path/to/faces`.

Other directories:

| Path | Purpose |
|------|---------|
| `captures/` | Saved frames (auto-created when pressing 'S') |
| `~/.insightface/models/` | Model cache (auto-managed) |

## Usage

### Command Line

```bash
python -m src.face                      # Auto-detect watch list
python -m src.face -w ./faces           # Custom watch list folder
python -m src.face -t 0.4               # Custom threshold
python -m src.face -c 1                 # Use camera 1
python -m src.face -s ./saved           # Custom save directory
```

### As Library

```python
from src.face import ArcFaceRecognizer, FaceDatabase, load_watch_list
from pathlib import Path
import cv2

# Initialize
recognizer = ArcFaceRecognizer()

# Load watch list from directory
database = load_watch_list(recognizer, Path("watch_list"))

# Or add faces manually
image = cv2.imread("person.jpg")
embedding = recognizer.extract_embedding(image)
if embedding is not None:
    database.add("person_name", embedding)

# Match against database
match_name, score = database.find_match(query_embedding, threshold=0.35)
if match_name:
    print(f"Matched: {match_name} ({score:.2%})")
```

## Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reload watch list |
| `s` | Save current frame |
| `b` | Toggle brightness enhancement |
| `SPACE` | Pause/Resume |
| `+` / `-` | Adjust threshold |

## Watch List Format

Place face images in the watch list directory:
- Filename becomes the identity (e.g., `john_doe.jpg` → "john_doe")
- Supports: `.jpg`, `.jpeg`, `.png`
- One clear face per image recommended

## Requirements

```
numpy
opencv-python
insightface
onnxruntime
```
