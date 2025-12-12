"""Base face detector interface."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .types import DetectedFace


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
