"""OpenCV DNN face embedding backend using OpenFace."""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenCVDNNEmbeddingBackend:
    """Face embedding using OpenFace model via OpenCV DNN (128D)."""
    
    MODEL_URL = (
        "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
    )
    MODEL_FILENAME = "openface_nn4.small2.v1.t7"
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize OpenCV DNN embedding backend.
        
        Args:
            model_path: Path to OpenFace model file (.t7)
                       If None, will try default locations or download
        """
        self._net = None
        self._model_path = model_path
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "opencv_dnn"
    
    @property
    def embedding_dim(self) -> int:
        return 128
    
    @property
    def embedding_size(self) -> int:
        """Alias for embedding_dim for backward compatibility."""
        return self.embedding_dim
    
    def _initialize(self) -> bool:
        """Lazy initialization of OpenFace model."""
        if self._initialized:
            return self._net is not None
        
        self._initialized = True
        
        model_dir = Path("data/models/face_recognition")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_locations = [
            self._model_path,
            str(model_dir / self.MODEL_FILENAME),
            self.MODEL_FILENAME,
            str(Path.home() / ".face_models" / self.MODEL_FILENAME),
        ]
        
        model_file = None
        for loc in model_locations:
            if loc and Path(str(loc)).exists():
                model_file = str(loc)
                break
        
        if not model_file:
            model_file = self._download_model(model_dir / self.MODEL_FILENAME)
        
        if not model_file:
            logger.warning("OpenFace model not available")
            return False
        
        try:
            self._net = cv2.dnn.readNetFromTorch(model_file)
            logger.info(f"Loaded OpenFace model from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load OpenFace model: {e}")
            return False
    
    def _download_model(self, target_path: Path) -> Optional[str]:
        """Download OpenFace model if not present."""
        try:
            logger.info(f"Downloading OpenFace model to {target_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, str(target_path))
            logger.info("OpenFace model downloaded successfully")
            return str(target_path)
        except Exception as e:
            logger.error(f"Failed to download OpenFace model: {e}")
            return None
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 128D embedding using OpenFace.
        
        Args:
            face_image: BGR face image
            
        Returns:
            128D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            blob = cv2.dnn.blobFromImage(
                face_image,
                scalefactor=1.0 / 255,
                size=(96, 96),
                mean=(0, 0, 0),
                swapRB=True,
                crop=False,
            )
            
            self._net.setInput(blob)
            embedding = self._net.forward()
            
            embedding = embedding.flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"OpenFace embedding extraction failed: {e}")
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
