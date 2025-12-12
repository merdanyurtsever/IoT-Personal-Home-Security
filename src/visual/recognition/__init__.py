"""Face recognition module.

Contains:
- FaceRecognizer: Main recognizer class
- FaceDatabase: Persistent face storage
- Embedding backends (dlib, tflite, mobilenetv2, opencv_dnn)
- Attribute detection (glasses, beard, etc.)
"""

from .types import FaceCategory, RecognitionResult
from .recognizer import FaceRecognizer
from .database import FaceDatabase
from .attributes import (
    FaceAttribute,
    AttributeResult,
    AttributeProfile,
    BaseAttributeDetector,
    HaarAttributeDetector,
    AttributeFilter,
    AttributeFilterChain,
)
from .embeddings import (
    BaseEmbeddingBackend,
    DlibEmbeddingBackend,
    TFLiteEmbeddingBackend,
    MobileNetV2EmbeddingBackend,
    OpenCVDNNEmbeddingBackend,
    EMBEDDING_BACKENDS,
)

__all__ = [
    # Types
    "FaceCategory", "RecognitionResult",
    # Recognizer
    "FaceRecognizer",
    # Database
    "FaceDatabase",
    # Attributes
    "FaceAttribute", "AttributeResult", "AttributeProfile",
    "BaseAttributeDetector", "HaarAttributeDetector",
    "AttributeFilter", "AttributeFilterChain",
    # Embeddings
    "BaseEmbeddingBackend", "DlibEmbeddingBackend", "TFLiteEmbeddingBackend",
    "MobileNetV2EmbeddingBackend", "OpenCVDNNEmbeddingBackend", "EMBEDDING_BACKENDS",
]
