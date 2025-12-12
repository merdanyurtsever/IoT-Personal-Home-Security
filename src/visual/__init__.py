"""Visual processing package - face detection, recognition, and pipeline.

This package consolidates all visual/face-related code:
- detection/: Face detection backends (Haar, MediaPipe, Dlib, OpenCV DNN)
- recognition/: Face recognition and embeddings
- utils.py: Image processing utilities
- pipeline.py: End-to-end face processing pipeline
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
