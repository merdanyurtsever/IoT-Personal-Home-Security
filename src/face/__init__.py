"""Face Detection & Recognition Module.

A self-contained face recognition module using ArcFace (InsightFace buffalo_l).

Quick Start:
    # Run viewfinder
    python -m src.face
    
    # As library
    from src.face import ArcFaceRecognizer, FaceDatabase
    
    recognizer = ArcFaceRecognizer()
    faces = recognizer.detect(image)
    embedding = recognizer.extract_embedding(face_crop)

For full documentation, see README.md in this directory.
"""

from .viewfinder import (
    ArcFaceRecognizer,
    FaceDatabase,
    enhance_brightness,
    expand_bbox,
    cosine_similarity,
    load_watch_list,
    run_viewfinder,
)

__version__ = "3.0.0"

__all__ = [
    "ArcFaceRecognizer",
    "FaceDatabase",
    "enhance_brightness",
    "expand_bbox",
    "cosine_similarity",
    "load_watch_list",
    "run_viewfinder",
]

