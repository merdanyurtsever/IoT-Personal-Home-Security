"""Unified face detector with configurable backend."""

from typing import List, Tuple

import cv2
import numpy as np

from .types import DetectedFace
from .haar import HaarCascadeDetector
from .mediapipe import MediaPipeDetector
from .dlib import DlibFaceDetector
from .opencv_dnn import OpenCVDNNDetector


class FaceDetector:
    """Main face detector class with configurable backend.
    
    Default backend is opencv_dnn which works on all platforms (x86, ARM64, Pi).
    """
    
    BACKENDS = {
        "haar_cascade": HaarCascadeDetector,
        "mediapipe": MediaPipeDetector,
        "opencv_dnn": OpenCVDNNDetector,
        "dlib": DlibFaceDetector,
        "dlib_cnn": lambda **kwargs: DlibFaceDetector(use_cnn=True, **kwargs),
    }
    
    def __init__(self, backend: str = "opencv_dnn", **kwargs):
        """Initialize face detector with specified backend.
        
        Args:
            backend: Detection backend to use (default: opencv_dnn)
            **kwargs: Additional arguments for the detector
        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Available: {list(self.BACKENDS.keys())}"
            )
        
        self.backend_name = backend
        self.detector = self.BACKENDS[backend](**kwargs)
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image."""
        return self.detector.detect(image)
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        output = image.copy()
        
        for face in faces:
            cv2.rectangle(
                output,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                thickness,
            )
            
            if show_confidence:
                label = f"{face.confidence:.0%}"
                cv2.putText(
                    output, label,
                    (face.x, face.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            
            if show_landmarks and face.landmarks:
                for name, (lx, ly) in face.landmarks.items():
                    cv2.circle(output, (lx, ly), 2, (0, 0, 255), -1)
        
        return output
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """Return list of available detection backends."""
        return list(cls.BACKENDS.keys())
