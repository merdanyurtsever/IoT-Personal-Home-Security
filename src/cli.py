"""CLI entry point for IoT Home Security system.

Simplified command-line interface for the security system.

Usage:
    iot-security start [--config PATH] [--debug]
    iot-security detect [--image PATH] [--camera] [--backend NAME]
    iot-security test [--all]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    logger.info("Starting IoT Home Security System")
    
    config = load_config(args.config)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        from .face_service import FaceRecognizerService
        from .api import create_app
        import uvicorn
        
        # Initialize face service
        recog_config = config.get("face_recognition", {})
        db_config = config.get("database", {})
        api_config = config.get("api", {})
        
        service = FaceRecognizerService(
            database_path=db_config.get("faces_db", "data/faces.db"),
            upload_dir=recog_config.get("upload_dir", "data/raw/faces/uploads"),
            embedding_backend=recog_config.get("model", "opencv_dnn"),
            similarity_threshold=recog_config.get("similarity_threshold", 0.6),
        )
        logger.info("✓ Face recognition service initialized")
        
        # Start API server
        app = create_app(service=service)
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)
        
        logger.info(f"API: http://{host}:{port}")
        logger.info(f"Docs: http://{host}:{port}/docs")
        
        uvicorn.run(app, host=host, port=port, log_level="warning")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


def cmd_detect(args):
    """Run face detection on image or camera."""
    import numpy as np
    
    try:
        from .face import FaceDetector
        import cv2
        
        detector = FaceDetector(backend=args.backend)
        logger.info(f"Using {args.backend} backend")
        
        if args.image:
            # Process single image
            image = cv2.imread(args.image)
            if image is None:
                logger.error(f"Could not load: {args.image}")
                sys.exit(1)
            
            faces = detector.detect(image)
            logger.info(f"Found {len(faces)} face(s)")
            
            for i, face in enumerate(faces):
                logger.info(f"  [{i+1}] pos=({face.x},{face.y}) size={face.width}x{face.height} conf={face.confidence:.0%}")
            
            if args.output:
                output = detector.draw_detections(image, faces)
                cv2.imwrite(args.output, output)
                logger.info(f"Saved: {args.output}")
                
        elif args.camera:
            # Live camera detection
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
            # Quick test
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.detect(test_img)
            logger.info("✓ Detection working")
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_test(args):
    """Test system components."""
    import numpy as np
    results = []
    
    # Face detection
    logger.info("[1/4] Face detection...")
    try:
        from .face import FaceDetector
        detector = FaceDetector()
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("  ✓ OK")
        results.append(("Face detection", True))
    except Exception as e:
        logger.info(f"  ✗ {e}")
        results.append(("Face detection", False))
    
    # Face recognition
    logger.info("[2/4] Face recognition...")
    try:
        from .face import FaceRecognizer
        recognizer = FaceRecognizer(model="opencv_dnn")
        logger.info(f"  ✓ OK (embedding: {recognizer.embedding_size}D)")
        results.append(("Face recognition", True))
    except Exception as e:
        logger.info(f"  ✗ {e}")
        results.append(("Face recognition", False))
    
    # Sound classification
    logger.info("[3/4] Sound classification...")
    try:
        from .audio import SoundClassifier
        classifier = SoundClassifier()
        logger.info(f"  ✓ OK ({len(classifier.classes)} classes)")
        results.append(("Sound classification", True))
    except Exception as e:
        logger.info(f"  ✗ {e}")
        results.append(("Sound classification", False))
    
    # Camera (optional)
    if args.all:
        logger.info("[4/4] Camera...")
        try:
            from .sensors import CameraInterface
            camera = CameraInterface()
            camera.start()
            frame = camera.capture()
            camera.stop()
            logger.info(f"  ✓ OK ({frame.shape[1]}x{frame.shape[0]})")
            results.append(("Camera", True))
        except Exception as e:
            logger.info(f"  ✗ {e}")
            results.append(("Camera", False))
    
    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    logger.info(f"\nResult: {passed}/{total} components OK")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="iot-security",
        description="IoT Home Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  iot-security start                  Start API server
  iot-security start --debug          Start with debug logging
  iot-security detect --camera        Live face detection
  iot-security detect -i photo.jpg    Detect faces in image
  iot-security test                   Test all components
        """,
    )
    
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # start
    start_p = subparsers.add_parser("start", help="Start the system")
    start_p.add_argument("-c", "--config", default="config/config.yaml", help="Config file")
    start_p.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    
    # detect
    detect_p = subparsers.add_parser("detect", help="Face detection")
    detect_p.add_argument("-i", "--image", help="Image file")
    detect_p.add_argument("-o", "--output", help="Output file")
    detect_p.add_argument("--camera", action="store_true", help="Use camera")
    detect_p.add_argument("-b", "--backend", default="opencv_dnn",
                          choices=["haar_cascade", "mediapipe", "opencv_dnn", "dlib"],
                          help="Detection backend")
    
    # test
    test_p = subparsers.add_parser("test", help="Test components")
    test_p.add_argument("--all", action="store_true", help="Include hardware tests")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    commands = {
        "start": cmd_start,
        "detect": cmd_detect,
        "test": cmd_test,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
