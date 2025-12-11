"""CLI entry point for IoT Home Security system.

This module provides a comprehensive command-line interface for the security system.

Usage:
    iot-security start [--config PATH] [--debug]
    iot-security api [--port PORT]
    iot-security detect [--image PATH] [--camera]
    iot-security recognize [--image PATH]
    iot-security classify [--audio PATH]
    iot-security demo
    iot-security test [--component NAME]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config file not found: {config_path}")
    return {}


def cmd_start(args):
    """Start the security system."""
    logger.info(f"Starting IoT Home Security System")
    logger.info(f"Config: {args.config}")
    
    config = load_config(args.config)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Import here to avoid circular imports
        from .face import FaceDetector, FaceRecognizerService
        from .audio import SoundClassifier
        
        # Initialize face detection
        if config.get("face_detection", {}).get("enabled", True):
            face_config = config.get("face_detection", {})
            detector = FaceDetector(
                backend=face_config.get("model", "haar_cascade")
            )
            logger.info(f"✓ Face detector initialized ({detector.backend_name})")
        
        # Initialize face recognition service
        if config.get("face_recognition", {}).get("enabled", True):
            recog_config = config.get("face_recognition", {})
            db_config = config.get("database", {})
            
            service = FaceRecognizerService(
                database_path=db_config.get("faces_db", "data/faces.db"),
                upload_dir=recog_config.get("upload_dir", "data/raw/faces/uploads"),
                embedding_backend=recog_config.get("model", "dlib"),
                similarity_threshold=recog_config.get("similarity_threshold", 0.6),
            )
            logger.info(f"✓ Face recognition service initialized")
        
        # Initialize sound classification
        if config.get("sound_classification", {}).get("enabled", True):
            sound_config = config.get("sound_classification", {})
            classifier = SoundClassifier(
                sample_rate=sound_config.get("sample_rate", 22050),
                confidence_threshold=sound_config.get("confidence_threshold", 0.7),
            )
            logger.info(f"✓ Sound classifier initialized")
        
        # Check if API should be started
        api_config = config.get("api", {})
        if api_config.get("enabled", True) and not args.no_api:
            logger.info("Starting API server...")
            cmd_api_internal(
                port=api_config.get("port", 8000),
                host=api_config.get("host", "0.0.0.0"),
                service=service if 'service' in dir() else None,
            )
        else:
            logger.info("System started without API. Press Ctrl+C to stop.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Run './run.sh install' to install dependencies")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


def cmd_api_internal(port: int = 8000, host: str = "0.0.0.0", service=None):
    """Start the API server (internal function)."""
    try:
        import uvicorn
        from .api import create_app
        from .face import FaceRecognizerService
        
        if service is None:
            service = FaceRecognizerService(
                database_path="data/faces.db",
                upload_dir="data/raw/faces/uploads",
                embedding_backend="dlib",
            )
        
        app = create_app(service=service)
        
        logger.info(f"API available at http://{host}:{port}")
        logger.info(f"Documentation at http://{host}:{port}/docs")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install fastapi uvicorn")
        sys.exit(1)


def cmd_api(args):
    """Start the Face Management API server."""
    logger.info(f"Starting Face Management API on port {args.port}")
    cmd_api_internal(port=args.port)


def cmd_detect(args):
    """Test face detection."""
    import numpy as np
    
    try:
        from .face import FaceDetector
        
        detector = FaceDetector(backend=args.backend)
        logger.info(f"Face detector initialized ({detector.backend_name})")
        
        if args.image:
            import cv2
            image = cv2.imread(args.image)
            if image is None:
                logger.error(f"Could not load image: {args.image}")
                sys.exit(1)
            
            faces = detector.detect(image)
            logger.info(f"Detected {len(faces)} face(s)")
            
            for i, face in enumerate(faces):
                logger.info(f"  Face {i+1}: ({face.x}, {face.y}) {face.width}x{face.height} conf={face.confidence:.2%}")
            
            if args.output:
                output = detector.draw_detections(image, faces)
                cv2.imwrite(args.output, output)
                logger.info(f"Saved to {args.output}")
                
        elif args.camera:
            import cv2
            from .sensors import CameraInterface
            
            camera = CameraInterface()
            camera.start()
            logger.info("Camera started. Press 'q' to quit.")
            
            try:
                for frame in camera.stream():
                    faces = detector.detect(frame)
                    output = detector.draw_detections(frame, faces)
                    
                    cv2.imshow("Face Detection", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                camera.stop()
                cv2.destroyAllWindows()
        else:
            # Basic test
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = detector.detect(test_image)
            logger.info(f"✓ Detection test passed (found {len(faces)} faces in blank image)")
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_recognize(args):
    """Test face recognition."""
    try:
        from .face import FaceRecognizer
        
        recognizer = FaceRecognizer(model=args.model, threshold=args.threshold)
        logger.info(f"Face recognizer initialized")
        logger.info(f"  Model: {recognizer.model_name}")
        logger.info(f"  Embedding size: {recognizer.embedding_size}")
        logger.info(f"  Threshold: {recognizer.threshold}")
        logger.info(f"  Enrolled faces: {recognizer.get_enrolled_count()}")
        
        if args.image:
            import cv2
            image = cv2.imread(args.image)
            if image is None:
                logger.error(f"Could not load image: {args.image}")
                sys.exit(1)
            
            result = recognizer.recognize(image)
            logger.info(f"Recognition result:")
            logger.info(f"  Identity: {result.identity}")
            logger.info(f"  Confidence: {result.confidence:.2%}")
            logger.info(f"  Category: {result.category.value}")
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_classify(args):
    """Test sound classification."""
    import numpy as np
    
    try:
        from .audio import SoundClassifier
        
        classifier = SoundClassifier(sample_rate=args.sample_rate)
        logger.info(f"Sound classifier initialized")
        logger.info(f"  Sample rate: {classifier.sample_rate}")
        logger.info(f"  Classes: {classifier.classes}")
        
        if args.audio:
            # Load audio file
            try:
                import librosa
                audio, sr = librosa.load(args.audio, sr=classifier.sample_rate)
            except ImportError:
                import soundfile as sf
                audio, sr = sf.read(args.audio)
                if sr != classifier.sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {classifier.sample_rate}")
            
            result = classifier.classify(audio)
            logger.info(f"Classification result:")
            logger.info(f"  Label: {result.label}")
            logger.info(f"  Confidence: {result.confidence:.2%}")
            logger.info(f"  Security event: {result.is_security_event}")
        else:
            # Test with random audio
            audio = np.random.randn(classifier.sample_rate * 5).astype(np.float32)
            result = classifier.classify(audio)
            logger.info(f"✓ Classification test: {result.label} ({result.confidence:.2%})")
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_demo(args):
    """Run a demo of all components."""
    logger.info("=" * 60)
    logger.info("IoT Home Security - Demo")
    logger.info("=" * 60)
    
    # Test face detection
    logger.info("\n[1/3] Testing Face Detection...")
    try:
        from .face import FaceDetector
        import numpy as np
        
        detector = FaceDetector()
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(test_img)
        logger.info("✓ Face detection OK")
    except Exception as e:
        logger.error(f"✗ Face detection failed: {e}")
    
    # Test face recognition
    logger.info("\n[2/3] Testing Face Recognition...")
    try:
        from .face import FaceRecognizer
        
        recognizer = FaceRecognizer(model="dlib")
        logger.info(f"✓ Face recognition OK (embedding size: {recognizer.embedding_size})")
    except Exception as e:
        logger.error(f"✗ Face recognition failed: {e}")
    
    # Test sound classification
    logger.info("\n[3/3] Testing Sound Classification...")
    try:
        from .audio import SoundClassifier
        import numpy as np
        
        classifier = SoundClassifier()
        audio = np.random.randn(22050 * 5).astype(np.float32)
        result = classifier.classify(audio)
        logger.info(f"✓ Sound classification OK ({result.label})")
    except Exception as e:
        logger.error(f"✗ Sound classification failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


def cmd_test(args):
    """Test system components."""
    logger.info(f"Testing component: {args.component}")
    
    if args.component in ("camera", "all"):
        logger.info("\nTesting camera...")
        try:
            from .sensors import CameraInterface
            camera = CameraInterface()
            camera.start()
            frame = camera.capture()
            camera.stop()
            logger.info(f"✓ Camera OK (frame shape: {frame.shape})")
        except Exception as e:
            logger.error(f"✗ Camera failed: {e}")
    
    if args.component in ("microphone", "all"):
        logger.info("\nTesting microphone...")
        try:
            from .sensors import MicrophoneInterface
            mic = MicrophoneInterface()
            mic.start()
            audio = mic.read(duration=1.0)
            mic.stop()
            logger.info(f"✓ Microphone OK (samples: {len(audio)})")
        except Exception as e:
            logger.error(f"✗ Microphone failed: {e}")
    
    if args.component in ("sensors", "all"):
        logger.info("\nTesting motion sensor...")
        try:
            from .sensors import MotionSensor
            sensor = MotionSensor(use_mock=True)
            value = sensor.read()
            logger.info(f"✓ Motion sensor OK (value: {value})")
        except Exception as e:
            logger.error(f"✗ Motion sensor failed: {e}")


def main():
    """Main entry point for the IoT Home Security CLI."""
    parser = argparse.ArgumentParser(
        prog="iot-security",
        description="IoT Personal Home Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  iot-security start                      # Start the system
  iot-security start --debug              # Start with debug output
  iot-security api --port 8080            # Start API on port 8080
  iot-security detect --camera            # Live face detection
  iot-security demo                       # Run demo
  iot-security test --component camera    # Test camera

For more information, see: https://github.com/yourusername/IoT-Personal-Home-Security
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the security system")
    start_parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    start_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output",
    )
    start_parser.add_argument(
        "--no-api",
        action="store_true",
        help="Start without API server",
    )
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the Face Management API")
    api_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Test face detection")
    detect_parser.add_argument(
        "--image", "-i",
        type=str,
        help="Image file to process",
    )
    detect_parser.add_argument(
        "--camera",
        action="store_true",
        help="Use live camera feed",
    )
    detect_parser.add_argument(
        "--backend", "-b",
        type=str,
        default="haar_cascade",
        choices=["haar_cascade", "mediapipe", "opencv_dnn"],
        help="Detection backend",
    )
    detect_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output image path",
    )
    
    # Recognize command
    recognize_parser = subparsers.add_parser("recognize", help="Test face recognition")
    recognize_parser.add_argument(
        "--image", "-i",
        type=str,
        help="Face image to recognize",
    )
    recognize_parser.add_argument(
        "--model", "-m",
        type=str,
        default="dlib",
        choices=["dlib", "tflite"],
        help="Recognition model",
    )
    recognize_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.6,
        help="Recognition threshold",
    )
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Test sound classification")
    classify_parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Audio file to classify",
    )
    classify_parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=22050,
        help="Sample rate",
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo of all components")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument(
        "--component",
        type=str,
        choices=["camera", "microphone", "sensors", "all"],
        default="all",
        help="Which component to test",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to command handler
    commands = {
        "start": cmd_start,
        "api": cmd_api,
        "detect": cmd_detect,
        "recognize": cmd_recognize,
        "classify": cmd_classify,
        "demo": cmd_demo,
        "test": cmd_test,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
