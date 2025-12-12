"""Face detection backends.

Available backends:
- haar_cascade: Fast, lightweight OpenCV Haar Cascades
- mediapipe: Google MediaPipe, good for ARM64
- dlib: Accurate HOG/CNN detector with landmarks
- opencv_dnn: SSD ResNet, good balance of speed/accuracy (default)
"""

from .types import DetectedFace
from .base import BaseFaceDetector
from .haar import HaarCascadeDetector
from .mediapipe import MediaPipeDetector
from .dlib import DlibFaceDetector
from .opencv_dnn import OpenCVDNNDetector
from .detector import FaceDetector

DETECTION_BACKENDS = {
    "haar_cascade": HaarCascadeDetector,
    "mediapipe": MediaPipeDetector,
    "dlib": DlibFaceDetector,
    "opencv_dnn": OpenCVDNNDetector,
}

__all__ = [
    "DetectedFace",
    "BaseFaceDetector",
    "HaarCascadeDetector",
    "MediaPipeDetector",
    "DlibFaceDetector",
    "OpenCVDNNDetector",
    "FaceDetector",
    "DETECTION_BACKENDS",
]
