"""RetinaFace detector using InsightFace.

RetinaFace is a state-of-the-art face detector that provides:
- High accuracy face detection
- 5-point facial landmarks
- Face quality scores
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseFaceDetector
from .types import DetectedFace

logger = logging.getLogger(__name__)


class RetinaFaceDetector(BaseFaceDetector):
    """Face detector using RetinaFace via InsightFace.
    
    RetinaFace provides state-of-the-art detection accuracy with
    facial landmarks for alignment.
    
    Pros: Excellent accuracy, provides landmarks, handles occlusion well
    Cons: Slower than simpler detectors, requires insightface library
    Best for: High-accuracy applications where speed is less critical
    """
    
    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        model_name: str = "buffalo_l",
    ):
        """Initialize RetinaFace detector.
        
        Args:
            det_size: Detection input size (width, height)
            det_thresh: Detection confidence threshold
            model_name: InsightFace model pack:
                - buffalo_l: Large model, best accuracy
                - buffalo_sc: Smaller, faster
        """
        self._app = None
        self._initialized = False
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._model_name = model_name
    
    def _initialize(self) -> bool:
        """Lazy initialization of InsightFace model."""
        if self._initialized:
            return self._app is not None
        
        self._initialized = True
        
        try:
            from insightface.app import FaceAnalysis
            
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=['CPUExecutionProvider']
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size)
            
            # Set detection threshold
            if hasattr(self._app, 'det_model') and self._app.det_model:
                self._app.det_model.det_thresh = self._det_thresh
            
            logger.info(f"Initialized RetinaFace detector with {self._model_name}")
            return True
            
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RetinaFace: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using RetinaFace.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects with landmarks
        """
        if not self._initialize():
            return []
        
        try:
            faces = self._app.get(image)
            
            detected = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Ensure valid bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                
                # Extract landmarks if available (5-point)
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                # Get confidence score
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                
                detected.append(DetectedFace(
                    x=x1,
                    y=y1,
                    width=width,
                    height=height,
                    confidence=confidence,
                    landmarks=landmarks,
                ))
            
            return detected
            
        except Exception as e:
            logger.debug(f"RetinaFace detection failed: {e}")
            return []


class InsightFaceDetector(BaseFaceDetector):
    """Generic InsightFace detector wrapper.
    
    Provides a unified interface for different InsightFace model packs.
    Can be configured for speed vs accuracy tradeoffs.
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_sc",
        det_size: Tuple[int, int] = (320, 320),
        det_thresh: float = 0.5,
    ):
        """Initialize InsightFace detector.
        
        Args:
            model_name: Model pack name
                - buffalo_s: Smallest, fastest
                - buffalo_sc: Small, good balance
                - buffalo_l: Large, best accuracy
            det_size: Detection input size
            det_thresh: Detection threshold
        """
        self._app = None
        self._initialized = False
        self._model_name = model_name
        self._det_size = det_size
        self._det_thresh = det_thresh
    
    def _initialize(self) -> bool:
        """Lazy initialization."""
        if self._initialized:
            return self._app is not None
        
        self._initialized = True
        
        try:
            from insightface.app import FaceAnalysis
            
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=['CPUExecutionProvider']
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size)
            
            logger.info(f"Initialized InsightFace detector: {self._model_name}")
            return True
            
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using InsightFace.
        
        Args:
            image: BGR image
            
        Returns:
            List of DetectedFace objects
        """
        if not self._initialize():
            return []
        
        try:
            faces = self._app.get(image)
            
            detected = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                
                detected.append(DetectedFace(
                    x=x1,
                    y=y1,
                    width=width,
                    height=height,
                    confidence=confidence,
                    landmarks=landmarks,
                ))
            
            return detected
            
        except Exception as e:
            logger.debug(f"InsightFace detection failed: {e}")
            return []
