"""Sound classification."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .preprocessing import AudioPreprocessor, AudioConfig
from .features import FeatureExtractor, FeatureConfig
from ..constants import get_audio_processing_config

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of sound classification."""
    label: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None
    
    @property
    def is_alert(self) -> bool:
        """Check if this is an alert-worthy sound."""
        alert_sounds = {
            "dog_bark", "siren", "glass_breaking", "gunshot",
            "scream", "alarm", "car_horn", "crying_baby",
        }
        return self.label.lower() in alert_sounds


class SoundClassifier:
    """Sound classification using various backends."""
    
    DEFAULT_CLASSES = [
        "dog_bark", "children_playing", "car_horn", "air_conditioner",
        "street_music", "gun_shot", "siren", "engine_idling",
        "jackhammer", "drilling",
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        audio_config: Optional[AudioConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """Initialize sound classifier.
        
        Args:
            model_path: Path to trained model file
            classes: List of class names
            audio_config: Audio preprocessing config
            feature_config: Feature extraction config
        """
        self.classes = classes or self.DEFAULT_CLASSES
        self._model = None
        self._model_type = None
        self._model_path = model_path
        
        self._preprocessor = AudioPreprocessor(audio_config)
        self._feature_extractor = FeatureExtractor(feature_config)
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> bool:
        """Load classification model."""
        path = Path(model_path)
        
        if not path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        suffix = path.suffix.lower()
        
        try:
            if suffix == ".tflite":
                return self._load_tflite(model_path)
            elif suffix in (".h5", ".keras"):
                return self._load_keras(model_path)
            elif suffix == ".onnx":
                return self._load_onnx(model_path)
            elif suffix == ".pkl":
                return self._load_sklearn(model_path)
            else:
                logger.warning(f"Unsupported model format: {suffix}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_tflite(self, model_path: str) -> bool:
        """Load TFLite model."""
        try:
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            self._model = tflite.Interpreter(model_path=model_path)
            self._model.allocate_tensors()
            self._model_type = "tflite"
            logger.info(f"Loaded TFLite model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False
    
    def _load_keras(self, model_path: str) -> bool:
        """Load Keras model."""
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(model_path, compile=False)
            self._model_type = "keras"
            logger.info(f"Loaded Keras model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            return False
    
    def _load_onnx(self, model_path: str) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(model_path)
            self._model_type = "onnx"
            logger.info(f"Loaded ONNX model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def _load_sklearn(self, model_path: str) -> bool:
        """Load sklearn model."""
        try:
            import pickle
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            self._model_type = "sklearn"
            logger.info(f"Loaded sklearn model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            return False
    
    def classify(
        self,
        audio: Union[np.ndarray, str, Path],
        sr: Optional[int] = None,
    ) -> ClassificationResult:
        """Classify a sound.
        
        Args:
            audio: Audio data (numpy array) or path to audio file
            sr: Sample rate (required if audio is numpy array)
            
        Returns:
            Classification result
        """
        if isinstance(audio, (str, Path)):
            audio, sr = self._preprocessor.load(audio)
        
        if sr is None:
            sr = self._preprocessor.config.sample_rate
        
        features = self._extract_features(audio, sr)
        
        if self._model is None:
            return self._dummy_classify(features)
        
        return self._predict(features)
    
    def _extract_features(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Extract features for classification."""
        if self._model_type in ("keras", "tflite"):
            mel_spec = self._feature_extractor.extract_mel_spectrogram(audio, sr)
            return mel_spec[np.newaxis, ..., np.newaxis]
        else:
            return self._feature_extractor.extract_all(audio, sr, aggregate=True)
    
    def _predict(self, features: np.ndarray) -> ClassificationResult:
        """Run model prediction."""
        try:
            if self._model_type == "keras":
                probs = self._model.predict(features, verbose=0)[0]
            
            elif self._model_type == "tflite":
                input_details = self._model.get_input_details()
                output_details = self._model.get_output_details()
                
                features = features.astype(input_details[0]["dtype"])
                self._model.set_tensor(input_details[0]["index"], features)
                self._model.invoke()
                probs = self._model.get_tensor(output_details[0]["index"])[0]
            
            elif self._model_type == "onnx":
                input_name = self._model.get_inputs()[0].name
                outputs = self._model.run(None, {input_name: features.astype(np.float32)})
                probs = outputs[0][0]
            
            elif self._model_type == "sklearn":
                if hasattr(self._model, "predict_proba"):
                    probs = self._model.predict_proba(features.reshape(1, -1))[0]
                else:
                    pred_idx = self._model.predict(features.reshape(1, -1))[0]
                    probs = np.zeros(len(self.classes))
                    probs[pred_idx] = 1.0
            
            else:
                return self._dummy_classify(features)
            
            pred_idx = int(np.argmax(probs))
            label = self.classes[pred_idx] if pred_idx < len(self.classes) else "unknown"
            confidence = float(probs[pred_idx])
            
            all_probs = {
                self.classes[i]: float(probs[i])
                for i in range(min(len(probs), len(self.classes)))
            }
            
            return ClassificationResult(
                label=label,
                confidence=confidence,
                all_probabilities=all_probs,
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ClassificationResult(
                label="error",
                confidence=0.0,
            )
    
    def _dummy_classify(self, features: np.ndarray) -> ClassificationResult:
        """Dummy classification when no model is loaded."""
        audio_config = get_audio_processing_config()
        energy = np.mean(np.abs(features))
        
        if energy > audio_config.dummy_loud_threshold:
            label = "loud_sound"
        elif energy > audio_config.dummy_ambient_threshold:
            label = "ambient"
        else:
            label = "silence"
        
        return ClassificationResult(
            label=label,
            confidence=audio_config.dummy_default_confidence,
            all_probabilities={label: audio_config.dummy_default_confidence},
        )
    
    def classify_file(self, audio_path: Union[str, Path]) -> ClassificationResult:
        """Convenience method to classify an audio file."""
        return self.classify(audio_path)
    
    def classify_stream(
        self,
        audio_chunk: np.ndarray,
        sr: int,
    ) -> ClassificationResult:
        """Classify a streaming audio chunk."""
        return self.classify(audio_chunk, sr)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    @property
    def model_type(self) -> Optional[str]:
        """Get loaded model type."""
        return self._model_type
