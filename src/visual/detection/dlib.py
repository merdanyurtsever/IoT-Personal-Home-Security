"""Dlib face detector."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from .base import BaseFaceDetector
from .types import DetectedFace

logger = logging.getLogger(__name__)


class DlibFaceDetector(BaseFaceDetector):
    """Face detector using dlib library with HOG or CNN model.
    
    Pros: Very accurate, provides 68-point facial landmarks
    Cons: Slower than other methods, requires dlib compilation
    Best for: High-accuracy detection when speed is not critical
    """
    
    def __init__(
        self,
        use_cnn: bool = False,
        upsample_num_times: int = 1,
        cnn_model_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize dlib face detector."""
        self.use_cnn = use_cnn
        self.upsample_num_times = upsample_num_times
        
        try:
            import dlib
            self._dlib = dlib
            
            if use_cnn:
                if cnn_model_path is None:
                    default_path = Path(__file__).parent.parent.parent.parent / "data" / "models" / "face_detection" / "mmod_human_face_detector.dat"
                    if default_path.exists():
                        cnn_model_path = default_path
                    else:
                        raise FileNotFoundError(
                            "CNN model not found. Download from: "
                            "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
                        )
                self.detector = dlib.cnn_face_detection_model_v1(str(cnn_model_path))
                logger.info("Initialized dlib CNN face detector")
            else:
                self.detector = dlib.get_frontal_face_detector()
                logger.info("Initialized dlib HOG face detector")
            
            # Load shape predictor for landmarks (optional)
            self.shape_predictor = None
            predictor_path = Path(__file__).parent.parent.parent.parent / "data" / "models" / "face_detection" / "shape_predictor_68_face_landmarks.dat"
            if predictor_path.exists():
                self.shape_predictor = dlib.shape_predictor(str(predictor_path))
                logger.info("Loaded 68-point shape predictor for landmarks")
                
        except ImportError:
            raise ImportError(
                "dlib is required. Install with: pip install dlib"
            )
    
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using dlib."""
        # dlib expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect faces
        if self.use_cnn:
            detections = self.detector(rgb_image, self.upsample_num_times)
            faces_data = [(d.rect, d.confidence) for d in detections]
        else:
            detections, scores, _ = self.detector.run(
                rgb_image, self.upsample_num_times, 0.0
            )
            faces_data = list(zip(detections, scores))
        
        detected = []
        for rect, confidence in faces_data:
            x = max(0, rect.left())
            y = max(0, rect.top())
            width = rect.width()
            height = rect.height()
            
            if width <= 0 or height <= 0:
                continue
            
            # Extract landmarks if shape predictor is available
            landmarks = None
            if self.shape_predictor is not None:
                try:
                    shape = self.shape_predictor(rgb_image, rect)
                    landmarks = {
                        'left_eye': (
                            (shape.part(36).x + shape.part(39).x) // 2,
                            (shape.part(36).y + shape.part(39).y) // 2
                        ),
                        'right_eye': (
                            (shape.part(42).x + shape.part(45).x) // 2,
                            (shape.part(42).y + shape.part(45).y) // 2
                        ),
                        'nose_tip': (shape.part(30).x, shape.part(30).y),
                        'mouth_left': (shape.part(48).x, shape.part(48).y),
                        'mouth_right': (shape.part(54).x, shape.part(54).y),
                        'mouth_center': (
                            (shape.part(48).x + shape.part(54).x) // 2,
                            (shape.part(51).y + shape.part(57).y) // 2
                        ),
                    }
                except Exception as e:
                    logger.debug(f"Failed to extract landmarks: {e}")
            
            detected.append(DetectedFace(
                x=x, y=y, width=width, height=height,
                confidence=float(confidence), landmarks=landmarks
            ))
        
        return detected
