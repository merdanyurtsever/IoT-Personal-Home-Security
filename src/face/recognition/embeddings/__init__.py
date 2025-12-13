"""Face embedding backends.

Embedding backends extract numerical representations (embeddings) from face images
for comparison and recognition.
"""

from .base import BaseEmbeddingBackend
from .dlib import DlibEmbeddingBackend
from .tflite import TFLiteEmbeddingBackend
from .mobilenet import MobileNetV2EmbeddingBackend
from .opencv_dnn import OpenCVDNNEmbeddingBackend
from .insightface_backend import ArcFaceEmbeddingBackend, MobileFaceNetEmbeddingBackend

EMBEDDING_BACKENDS = {
    "dlib": DlibEmbeddingBackend,
    "tflite": TFLiteEmbeddingBackend,
    "mobilenetv2": MobileNetV2EmbeddingBackend,
    "opencv_dnn": OpenCVDNNEmbeddingBackend,
    "arcface": ArcFaceEmbeddingBackend,
    "mobilefacenet": MobileFaceNetEmbeddingBackend,
}

__all__ = [
    "BaseEmbeddingBackend",
    "DlibEmbeddingBackend",
    "TFLiteEmbeddingBackend",
    "MobileNetV2EmbeddingBackend",
    "OpenCVDNNEmbeddingBackend",
    "ArcFaceEmbeddingBackend",
    "MobileFaceNetEmbeddingBackend",
    "EMBEDDING_BACKENDS",
]
