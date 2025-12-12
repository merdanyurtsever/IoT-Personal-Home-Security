"""MobileNetV2 face embedding backend."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MobileNetV2EmbeddingBackend:
    """Face embedding using MobileNetV2 (512D)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize MobileNetV2 embedding backend.
        
        Args:
            model_path: Path to MobileNetV2 model file
                       Supports .h5, .keras, .tflite, or SavedModel directory
        """
        self._model = None
        self._model_path = model_path
        self._initialized = False
        self._model_type = None
    
    @property
    def name(self) -> str:
        return "mobilenetv2"
    
    @property
    def embedding_dim(self) -> int:
        return 512
    
    @property
    def embedding_size(self) -> int:
        """Alias for embedding_dim for backward compatibility."""
        return self.embedding_dim
    
    def _initialize(self) -> bool:
        """Lazy initialization of MobileNetV2 model."""
        if self._initialized:
            return self._model is not None
        
        self._initialized = True
        
        model_locations = [
            self._model_path,
            "mobilenetv2_face_embedding.h5",
            "mobilenetv2_face_embedding.keras",
            "mobilenetv2_face_embedding.tflite",
            "data/models/face_recognition/mobilenetv2_face_embedding.h5",
            Path.home() / ".face_models" / "mobilenetv2_face_embedding.h5",
        ]
        
        model_file = None
        for loc in model_locations:
            if loc and Path(str(loc)).exists():
                model_file = str(loc)
                break
        
        if model_file and model_file.endswith(".tflite"):
            return self._load_tflite(model_file)
        elif model_file:
            return self._load_keras(model_file)
        else:
            return self._create_default_model()
    
    def _load_keras(self, model_file: str) -> bool:
        """Load Keras/TensorFlow model."""
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(model_file, compile=False)
            self._model_type = "keras"
            logger.info(f"Loaded MobileNetV2 Keras model from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            return False
    
    def _load_tflite(self, model_file: str) -> bool:
        """Load TFLite model."""
        try:
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            self._model = tflite.Interpreter(model_path=model_file)
            self._model.allocate_tensors()
            self._model_type = "tflite"
            logger.info(f"Loaded MobileNetV2 TFLite model from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False
    
    def _create_default_model(self) -> bool:
        """Create default MobileNetV2 embedding model."""
        try:
            from tensorflow import keras
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
            from tensorflow.keras.models import Model
            
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(224, 224, 3),
            )
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation=None, name="embedding")(x)
            
            self._model = Model(inputs=base_model.input, outputs=x)
            self._model_type = "keras"
            
            logger.info("Created default MobileNetV2 embedding model (ImageNet weights)")
            return True
            
        except Exception as e:
            logger.warning(f"Could not create MobileNetV2 model: {e}")
            return False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D embedding using MobileNetV2.
        
        Args:
            face_image: BGR face image
            
        Returns:
            512D embedding vector or None
        """
        if not self._initialize():
            return None
        
        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            
            input_data = resized.astype(np.float32)
            input_data = input_data / 127.5 - 1.0
            input_data = np.expand_dims(input_data, axis=0)
            
            if self._model_type == "keras":
                embedding = self._model.predict(input_data, verbose=0)
            else:
                input_details = self._model.get_input_details()
                output_details = self._model.get_output_details()
                
                self._model.set_tensor(input_details[0]["index"], input_data)
                self._model.invoke()
                embedding = self._model.get_tensor(output_details[0]["index"])
            
            embedding = embedding.flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"MobileNetV2 embedding extraction failed: {e}")
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
