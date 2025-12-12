"""Face Detection & Recognition Module.

An independent, self-contained module for face detection and recognition.
Can be used standalone or integrated into larger systems.

Usage:
    # As standalone CLI
    python -m src.face detect --image test.jpg
    python -m src.face detect --camera
    python -m src.face api --port 8000

    # As library
    from src.face import FaceDetector, FaceRecognizer
    
    detector = FaceDetector()
    faces = detector.detect(image)
    
    recognizer = FaceRecognizer()
    recognizer.register_face("Alice", image)
    results = recognizer.recognize_faces(image)

Structure:
    src/face/
    ├── __init__.py      # This file - public API
    ├── __main__.py      # Entry point for `python -m src.face`
    ├── cli.py           # Standalone CLI commands
    ├── api.py           # REST API for mobile apps
    ├── detection/       # Face detection backends
    ├── recognition/     # Face recognition & embeddings
    ├── pipeline.py      # End-to-end processing pipeline
    └── utils.py         # Image processing utilities
"""

from .utils import crop_face, align_face, preprocess_face, compute_face_quality

from .detection import (
    DetectedFace,
    BaseFaceDetector,
    HaarCascadeDetector,
    MediaPipeDetector,
    DlibFaceDetector,
    OpenCVDNNDetector,
    FaceDetector,
    DETECTION_BACKENDS,
)

from .recognition import (
    FaceCategory,
    RecognitionResult,
    FaceRecognizer,
    FaceDatabase,
    # Embeddings
    BaseEmbeddingBackend,
    DlibEmbeddingBackend,
    TFLiteEmbeddingBackend,
    MobileNetV2EmbeddingBackend,
    OpenCVDNNEmbeddingBackend,
    EMBEDDING_BACKENDS,
    # Attributes
    FaceAttribute,
    AttributeResult,
    AttributeProfile,
    BaseAttributeDetector,
    HaarAttributeDetector,
    AttributeFilter,
    AttributeFilterChain,
)

from .pipeline import FaceEvent, DetectionMode, FaceSecurityPipeline

# Version of this module
__version__ = "2.0.0"

__all__ = [
    # Utils
    "crop_face", "align_face", "preprocess_face", "compute_face_quality",
    # Detection
    "DetectedFace", "BaseFaceDetector", "HaarCascadeDetector", "MediaPipeDetector",
    "DlibFaceDetector", "OpenCVDNNDetector", "FaceDetector", "DETECTION_BACKENDS",
    # Recognition
    "FaceCategory", "RecognitionResult", "FaceRecognizer", "FaceDatabase",
    # Embeddings
    "BaseEmbeddingBackend", "DlibEmbeddingBackend", "TFLiteEmbeddingBackend",
    "MobileNetV2EmbeddingBackend", "OpenCVDNNEmbeddingBackend", "EMBEDDING_BACKENDS",
    # Attributes
    "FaceAttribute", "AttributeResult", "AttributeProfile",
    "BaseAttributeDetector", "HaarAttributeDetector",
    "AttributeFilter", "AttributeFilterChain",
    # Pipeline
    "FaceEvent", "DetectionMode", "FaceSecurityPipeline",
]
