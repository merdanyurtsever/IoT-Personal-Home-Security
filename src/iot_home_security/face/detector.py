"""Face detection module.

This module provides face detection functionality using various backends
including Haar Cascades, MTCNN, RetinaFace, and MediaPipe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box and confidence."""
    
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    landmarks: Optional[dict] = None
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        pass


class HaarCascadeDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascades."""
    
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        """Initialize Haar Cascade detector.
        
        Args:
            scale_factor: Scale factor for multi-scale detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size to detect
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load pre-trained cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Haar Cascade.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        
        # Convert to DetectedFace objects
        detected = []
        for (x, y, w, h) in faces:
            detected.append(DetectedFace(x=x, y=y, width=w, height=h))
        
        return detected


class FaceDetector:
    """Main face detector class with configurable backend."""
    
    BACKENDS = {
        "haar_cascade": HaarCascadeDetector,
        # TODO: Add more backends
        # "mtcnn": MTCNNDetector,
        # "retinaface": RetinaFaceDetector,
        # "mediapipe": MediaPipeDetector,
    }
    
    def __init__(self, backend: str = "haar_cascade", **kwargs):
        """Initialize face detector with specified backend.
        
        Args:
            backend: Detection backend to use
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
        """Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        return self.detector.detect(image)
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes on image.
        
        Args:
            image: BGR image as numpy array
            faces: List of detected faces
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        
        for face in faces:
            cv2.rectangle(
                output,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                thickness,
            )
        
        return output
