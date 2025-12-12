"""TFLite face embedding backend."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TFLiteEmbeddingBackend:
    """Face embedding using TFLite model (512D)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize TFLite embedding backend.
        
        Args:
            model_path: Path to TFLite face embedding model
                       If None, will try default locations
        """
        self._interpreter = None
        self._model_path = model_path
        self._initialized = False
        self._input_details = None
        self._output_details = None
    
    @property
    def name(self) -> str:
        return "tflite"
    
    @property
    def embedding_dim(self) -> int:
        return 512
    
    @property
    def embedding_size(self) -> int:
        """Alias for embedding_dim for backward compatibility."""
        return self.embedding_dim
    
    def _initialize(self) -> bool:
        """Lazy initialization of TFLite interpreter."""
        if self._initialized:
            return self._interpreter is not None
        
        self._initialized = True
        
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                logger.warning(
                    "TFLite not installed. Install with: "
                    "pip install tflite-runtime or pip install tensorflow"
                )
                return False
        
        model_locations = [
            self._model_path,
            "face_embedding.tflite",
            "data/models/face_recognition/face_embedding.tflite",
            Path.home() / ".face_models" / "face_embedding.tflite",
        ]
        
        model_file = None
        for loc in model_locations:
            if loc and Path(str(loc)).exists():
                model_file = str(loc)
                break
        
        if not model_file:
            logger.warning(
                "TFLite face embedding model not found. "
                "Please provide a valid model path."
            )
            return False
        
        try:
            self._interpreter = tflite.Interpreter(model_path=model_file)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info(f"Loaded TFLite model from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D embedding using TFLite.
        
        Args:
            face_image: BGR face image
            
        Returns:
            512D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            input_shape = self._input_details[0]["shape"]
            height, width = input_shape[1], input_shape[2]
            
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (width, height))
            
            input_data = resized.astype(np.float32)
            input_data = (input_data - 127.5) / 128.0
            input_data = np.expand_dims(input_data, axis=0)
            
            self._interpreter.set_tensor(
                self._input_details[0]["index"], input_data
            )
            self._interpreter.invoke()
            
            embedding = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )
            
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"TFLite embedding extraction failed: {e}")
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
