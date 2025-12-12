"""Dlib face embedding backend."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DlibEmbeddingBackend:
    """Face embedding using dlib's face recognition model (128D)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize dlib embedding backend.
        
        Args:
            model_path: Path to dlib_face_recognition_resnet_model_v1.dat
                       If None, will try default locations
        """
        self._face_rec = None
        self._shape_predictor = None
        self._model_path = model_path
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "dlib"
    
    @property
    def embedding_dim(self) -> int:
        return 128
    
    @property
    def embedding_size(self) -> int:
        """Alias for embedding_dim for backward compatibility."""
        return self.embedding_dim
    
    def _initialize(self) -> bool:
        """Lazy initialization of dlib models."""
        if self._initialized:
            return self._face_rec is not None
        
        self._initialized = True
        
        try:
            import dlib
        except ImportError:
            logger.warning("dlib not installed. Install with: pip install dlib")
            return False
        
        model_locations = [
            self._model_path,
            "dlib_face_recognition_resnet_model_v1.dat",
            Path.home() / ".dlib" / "dlib_face_recognition_resnet_model_v1.dat",
            "/usr/share/dlib/dlib_face_recognition_resnet_model_v1.dat",
        ]
        
        model_file = None
        for loc in model_locations:
            if loc and Path(str(loc)).exists():
                model_file = str(loc)
                break
        
        if not model_file:
            logger.warning(
                "dlib face recognition model not found. "
                "Download from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            )
            return False
        
        predictor_locations = [
            "shape_predictor_68_face_landmarks.dat",
            "shape_predictor_5_face_landmarks.dat",
            Path.home() / ".dlib" / "shape_predictor_68_face_landmarks.dat",
            Path.home() / ".dlib" / "shape_predictor_5_face_landmarks.dat",
        ]
        
        predictor_file = None
        for loc in predictor_locations:
            if Path(str(loc)).exists():
                predictor_file = str(loc)
                break
        
        try:
            self._face_rec = dlib.face_recognition_model_v1(model_file)
            if predictor_file:
                self._shape_predictor = dlib.shape_predictor(predictor_file)
            logger.info(f"Loaded dlib face recognition model from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dlib model: {e}")
            return False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 128D embedding using dlib.
        
        Args:
            face_image: BGR face image (already cropped)
            
        Returns:
            128D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            import dlib
            import cv2
            
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            h, w = rgb.shape[:2]
            full_rect = dlib.rectangle(0, 0, w, h)
            
            if self._shape_predictor is not None:
                shape = self._shape_predictor(rgb, full_rect)
                embedding = self._face_rec.compute_face_descriptor(rgb, shape)
            else:
                embedding = self._face_rec.compute_face_descriptor(rgb)
            
            return np.array(embedding)
            
        except Exception as e:
            logger.error(f"dlib embedding extraction failed: {e}")
            return None
    
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare embeddings using cosine similarity."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between embeddings."""
        return float(np.linalg.norm(embedding1 - embedding2))
