"""Haar Cascade face detector."""

from typing import List, Tuple

import cv2
import numpy as np

from .base import BaseFaceDetector
from .types import DetectedFace


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
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path}")
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Haar Cascade."""
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
