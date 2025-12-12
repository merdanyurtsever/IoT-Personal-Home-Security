"""Face detection using OpenCV DNN."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class DetectedFace:
    """Represents a detected face."""
    bbox: tuple  # (x, y, w, h)
    confidence: float
    

class FaceDetector:
    """Face detector using OpenCV DNN backend."""
    
    # OpenCV DNN face detector model files
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    def __init__(self, backend: str = "opencv_dnn", confidence_threshold: float = 0.5):
        """Initialize the face detector.
        
        Args:
            backend: Detection backend ("opencv_dnn" or "haar")
            confidence_threshold: Minimum confidence for detection
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self._net = None
        self._cascade = None
        
        if backend == "opencv_dnn":
            self._init_dnn()
        elif backend == "haar":
            self._init_haar()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _get_model_dir(self) -> Path:
        """Get or create model directory."""
        model_dir = Path(__file__).parent.parent.parent / "models" / "face_detection"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL."""
        import urllib.request
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
    
    def _init_dnn(self) -> None:
        """Initialize OpenCV DNN face detector."""
        model_dir = self._get_model_dir()
        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download models if not present
        if not prototxt.exists():
            self._download_file(self.PROTOTXT_URL, prototxt)
        if not caffemodel.exists():
            self._download_file(self.CAFFEMODEL_URL, caffemodel)
        
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    
    def _init_haar(self) -> None:
        """Initialize Haar cascade face detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of DetectedFace objects
        """
        if self.backend == "opencv_dnn":
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_dnn(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces using DNN."""
        h, w = frame.shape[:2]
        
        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self._net.setInput(blob)
        detections = self._net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure valid bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append(DetectedFace(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(confidence)
                ))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            faces.append(DetectedFace(
                bbox=(x, y, w, h),
                confidence=1.0  # Haar doesn't provide confidence
            ))
        
        return faces
