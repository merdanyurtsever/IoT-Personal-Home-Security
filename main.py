#!/usr/bin/env python3
"""Main entry point for IoT Home Security System.

Runs the security system combining face detection, sound classification,
and sensor monitoring. Includes a REST API for face management.
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from src import (
    FaceDetector,
    FaceRecognizer,
    FaceRecognizerService,
    FaceCategory,
    SoundClassifier,
    CameraInterface,
    MicrophoneInterface,
    MotionSensor,
    NotificationManager,
    LocalAlarm,
    crop_face,
    create_app,
)


class SecuritySystem:
    """Main security system class for Raspberry Pi deployment."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the security system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.logger = self._setup_logging()
        
        # Initialize components
        self.camera: Optional[CameraInterface] = None
        self.microphone: Optional[MicrophoneInterface] = None
        self.motion_sensor: Optional[MotionSensor] = None
        self.face_detector: Optional[FaceDetector] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.face_service: Optional[FaceRecognizerService] = None
        self.sound_classifier: Optional[SoundClassifier] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.local_alarm: Optional[LocalAlarm] = None
        
        # API server thread
        self._api_thread: Optional[threading.Thread] = None
        self._api_server = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the security system."""
        logger = logging.getLogger("security_system")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received shutdown signal")
        self.stop()
    
    def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing security system...")
        
        # Initialize camera
        camera_config = self.config.get("camera", {})
        if camera_config.get("enabled", True):
            self.logger.info("Initializing camera...")
            self.camera = CameraInterface(
                use_picamera=self.config.get("raspberry_pi", {}).get("use_picamera", True),
                resolution=tuple(camera_config.get("resolution", [640, 480])),
                fps=camera_config.get("fps", 30),
            )
        
        # Initialize microphone
        mic_config = self.config.get("microphone", {})
        if mic_config.get("enabled", True):
            self.logger.info("Initializing microphone...")
            self.microphone = MicrophoneInterface(
                sample_rate=mic_config.get("sample_rate", 44100),
                channels=mic_config.get("channels", 1),
            )
        
        # Initialize motion sensor
        motion_config = self.config.get("motion_sensor", {})
        if motion_config.get("enabled", True):
            self.logger.info("Initializing motion sensor...")
            self.motion_sensor = MotionSensor(
                pin=motion_config.get("gpio_pin", 17),
                cooldown=motion_config.get("cooldown_seconds", 5),
            )
        
        # Initialize face detection
        face_config = self.config.get("face_detection", {})
        if face_config.get("enabled", True):
            self.logger.info("Initializing face detection...")
            self.face_detector = FaceDetector(
                backend=face_config.get("model", "haar_cascade"),
            )
        
        # Initialize face recognition service (with database and API support)
        recog_config = self.config.get("face_recognition", {})
        if recog_config.get("enabled", True):
            self.logger.info("Initializing face recognition service...")
            
            # Get database path
            db_config = self.config.get("database", {})
            db_path = db_config.get("faces_db", "data/faces.db")
            
            # Get upload directory
            upload_dir = recog_config.get("upload_dir", "data/raw/faces/uploads")
            
            # Get TFLite model path if using TFLite backend
            tflite_model_path = recog_config.get("tflite_model_path")
            
            self.face_service = FaceRecognizerService(
                database_path=db_path,
                upload_dir=upload_dir,
                embedding_backend=recog_config.get("model", "dlib"),
                similarity_threshold=recog_config.get("similarity_threshold", 0.6),
                tflite_model_path=tflite_model_path,
            )
            
            # Keep reference to recognizer for compatibility
            self.face_recognizer = self.face_service.recognizer
        
        # Initialize sound classification
        sound_config = self.config.get("sound_classification", {})
        if sound_config.get("enabled", True):
            self.logger.info("Initializing sound classification...")
            model_path = self.config.get("models", {}).get("sound_classification")
            self.sound_classifier = SoundClassifier(
                model_path=model_path,
                sample_rate=sound_config.get("sample_rate", 44100),
                confidence_threshold=sound_config.get("confidence_threshold", 0.7),
            )
        
        # Initialize local alarm
        alarm_config = self.config.get("alerts", {}).get("local_alarm", {})
        if alarm_config.get("enabled", True):
            self.logger.info("Initializing local alarm...")
            self.local_alarm = LocalAlarm(
                buzzer_pin=alarm_config.get("gpio_pin", 18),
                default_duration=alarm_config.get("duration_seconds", 10),
            )
        
        # Initialize notification manager
        self.notification_manager = NotificationManager()
        
        # Initialize API server
        api_config = self.config.get("api", {})
        if api_config.get("enabled", True) and self.face_service:
            self.logger.info("Initializing face management API...")
            self._init_api_server(api_config)
        
        self.logger.info("Security system initialized")
    
    def _init_api_server(self, api_config: dict) -> None:
        """Initialize the FastAPI server for face management.
        
        Args:
            api_config: API configuration dictionary
        """
        try:
            from src.api import create_app
            import uvicorn
            
            # Create FastAPI app with face service
            cors_origins = api_config.get("cors_origins", ["*"])
            self.app = create_app(
                service=self.face_service,
                cors_origins=cors_origins,
                debug=self.config.get("app", {}).get("debug", False),
            )
            
            # Configure uvicorn
            host = api_config.get("host", "0.0.0.0")
            port = api_config.get("port", 8000)
            
            # Create server config
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=False,
            )
            self._api_server = uvicorn.Server(config)
            
            self.logger.info(f"API server configured on {host}:{port}")
            
        except ImportError as e:
            self.logger.warning(f"API server not available: {e}")
            self.logger.warning("Install with: pip install fastapi uvicorn")
    
    def _start_api_server(self) -> None:
        """Start the API server in a background thread."""
        if self._api_server is None:
            return
        
        def run_server():
            try:
                self._api_server.run()
            except Exception as e:
                self.logger.error(f"API server error: {e}")
        
        self._api_thread = threading.Thread(
            target=run_server,
            name="APIServer",
            daemon=True,
        )
        self._api_thread.start()
        self.logger.info("API server started")
    
    def _stop_api_server(self) -> None:
        """Stop the API server."""
        if self._api_server:
            self._api_server.should_exit = True
            if self._api_thread:
                self._api_thread.join(timeout=5.0)
    
    def start(self) -> None:
        """Start the security system."""
        self.logger.info("Starting security system...")
        self.running = True
        
        # Start camera
        if self.camera:
            self.camera.start()
        
        # Start microphone
        if self.microphone:
            self.microphone.start()
        
        # Register motion callback
        if self.motion_sensor:
            self.motion_sensor.on_motion(self._on_motion_detected)
        
        # Start API server
        self._start_api_server()
        
        self.logger.info("Security system running")
        
        # Main loop
        self._run_loop()
    
    def stop(self) -> None:
        """Stop the security system."""
        self.logger.info("Stopping security system...")
        self.running = False
        
        # Stop API server
        self._stop_api_server()
        
        # Stop face service
        if self.face_service:
            self.face_service.shutdown()
        
        # Stop components
        if self.camera:
            self.camera.stop()
        
        if self.microphone:
            self.microphone.stop()
        
        if self.local_alarm:
            self.local_alarm.stop()
        
        self.logger.info("Security system stopped")
    
    def _run_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            try:
                # Process camera frame
                if self.camera and self.face_detector:
                    self._process_video_frame()
                
                # Process audio
                if self.microphone and self.sound_classifier:
                    self._process_audio()
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
    
    def _process_video_frame(self) -> None:
        """Process a single video frame."""
        try:
            frame = self.camera.capture()
            faces = self.face_detector.detect(frame)
            
            for face in faces:
                self.logger.debug(f"Face detected at {face.bbox}")
                
                # Crop face and recognize
                if self.face_service:
                    face_image = crop_face(frame, face.bbox, margin=0.2)
                    result = self.face_service.recognize(face_image)
                    
                    if result.is_unwanted:
                        # Trigger alert for unwanted person
                        self.logger.warning(
                            f"UNWANTED PERSON DETECTED: {result.identity} "
                            f"(confidence: {result.confidence:.0%})"
                        )
                        self._trigger_alert("threat_detected", {
                            "identity": result.identity,
                            "confidence": result.confidence,
                            "face_id": result.face_id,
                        })
                    elif not result.is_threat:
                        self.logger.debug(
                            f"No threat match (confidence: {result.confidence:.0%})"
                        )
                        
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
    
    def _process_audio(self) -> None:
        """Process audio for sound classification."""
        try:
            # Read audio chunk
            audio = self.microphone.read(duration=1.0)
            
            # Classify
            result = self.sound_classifier.classify(audio)
            
            if self.sound_classifier.is_security_event(result):
                self.logger.warning(
                    f"Security sound detected: {result.label} "
                    f"({result.confidence:.0%})"
                )
                self._trigger_alert("sound_detected", {
                    "label": result.label,
                    "confidence": result.confidence,
                })
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _on_motion_detected(self) -> None:
        """Callback for motion detection."""
        self.logger.info("Motion detected!")
        self._trigger_alert("motion_detected", {})
    
    def _trigger_alert(self, event_type: str, details: dict) -> None:
        """Trigger security alert.
        
        Args:
            event_type: Type of security event
            details: Event details
        """
        self.logger.warning(f"Security alert: {event_type}")
        
        # Trigger local alarm
        if self.local_alarm:
            self.local_alarm.trigger()
        
        # TODO: Send notifications


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IoT Personal Home Security System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the API server (no sensors/camera)",
    )
    
    args = parser.parse_args()
    
    # Create and run security system
    system = SecuritySystem(config_path=args.config)
    
    if args.debug:
        system.logger.setLevel(logging.DEBUG)
    
    try:
        system.initialize()
        
        if args.api_only:
            # Run only API server
            system.logger.info("Running in API-only mode")
            system._start_api_server()
            # Keep running until interrupted
            while True:
                time.sleep(1)
        else:
            system.start()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
