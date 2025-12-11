"""IoT Personal Home Security System.

Simple flat structure:
- face.py: Face detection, recognition, utilities, pipeline, attribute detection
- face_service.py: Database, processing queue, service layer
- audio.py: Sound classification
- sensors.py: Camera, microphone, motion sensor
- alerts.py: Local alarms, notifications
- api.py: FastAPI REST endpoints
"""

# Face detection and recognition
from .face import (
    crop_face,
    align_face,
    preprocess_face,
    compute_face_quality,
    FaceDetector,
    DetectedFace,
    HaarCascadeDetector,
    MediaPipeDetector,
    OpenCVDNNDetector,
    DlibFaceDetector,
    FaceRecognizer,
    RecognitionResult,
    FaceCategory,
    FaceSecurityPipeline,
    FaceEvent,
    DetectionMode,  # Detection mode enum
    # Attribute detection (decorator pattern)
    FaceAttribute,
    AttributeResult,
    AttributeProfile,
    AttributeFilter,
    AttributeFilterChain,
    BaseAttributeDetector,
    HaarAttributeDetector,
    # Embedding backends
    BaseEmbeddingBackend,
    DlibEmbeddingBackend,
    TFLiteEmbeddingBackend,
    MobileNetV2EmbeddingBackend,
    OpenCVDNNEmbeddingBackend,
    EMBEDDING_BACKENDS,
)

# Face service (database + processing)
from .face_service import (
    FaceDatabase,
    FaceRecord,
    FaceStatus,
    FaceType,
    ProcessingJob,
    FaceProcessingQueue,
    ProcessingEvent,
    ProcessingEventType,
    FaceRecognizerService,
)

# Audio
from .audio import SoundClassifier, ClassificationResult

# Sensors
from .sensors import (
    CameraInterface,
    CameraConfig,
    MicrophoneInterface,
    MotionSensor,
    list_cameras,
)

# Alerts
from .alerts import LocalAlarm, NotificationManager, SecurityEvent

# API
from .api import router, create_app, set_face_service, get_face_service

__all__ = [
    "crop_face", "align_face", "preprocess_face", "compute_face_quality",
    "FaceDetector", "DetectedFace", "FaceRecognizer", "RecognitionResult", "FaceCategory",
    "HaarCascadeDetector", "MediaPipeDetector", "OpenCVDNNDetector", "DlibFaceDetector",
    "FaceSecurityPipeline", "FaceEvent",
    "FaceDatabase", "FaceRecord", "FaceStatus", "FaceType", "ProcessingJob",
    "FaceProcessingQueue", "ProcessingEvent", "ProcessingEventType", "FaceRecognizerService",
    "SoundClassifier", "ClassificationResult",
    "CameraInterface", "CameraConfig", "MicrophoneInterface", "MotionSensor", "list_cameras",
    "LocalAlarm", "NotificationManager", "SecurityEvent",
    "router", "create_app", "set_face_service", "get_face_service",
]
