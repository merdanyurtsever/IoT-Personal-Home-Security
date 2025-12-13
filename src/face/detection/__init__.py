"""Face detection backends.

Available backends:
- haar_cascade: Fast, lightweight OpenCV Haar Cascades
- mediapipe: Google MediaPipe, good for ARM64
- dlib: Accurate HOG/CNN detector with landmarks
- opencv_dnn: SSD ResNet, good balance of speed/accuracy (default)
- retinaface: State-of-the-art accuracy via InsightFace
- insightface: Generic InsightFace detector
"""

from .types import DetectedFace
from .base import BaseFaceDetector
from .haar import HaarCascadeDetector
from .mediapipe import MediaPipeDetector
from .dlib import DlibFaceDetector
from .opencv_dnn import OpenCVDNNDetector
from .insightface import RetinaFaceDetector, InsightFaceDetector
from .detector import FaceDetector

DETECTION_BACKENDS = {
    "haar_cascade": HaarCascadeDetector,
    "mediapipe": MediaPipeDetector,
    "dlib": DlibFaceDetector,
    "opencv_dnn": OpenCVDNNDetector,
    "retinaface": RetinaFaceDetector,
    "insightface": InsightFaceDetector,
}

__all__ = [
    "DetectedFace",
    "BaseFaceDetector",
    "HaarCascadeDetector",
    "MediaPipeDetector",
    "DlibFaceDetector",
    "OpenCVDNNDetector",
    "RetinaFaceDetector",
    "InsightFaceDetector",
    "FaceDetector",
    "DETECTION_BACKENDS",
]
