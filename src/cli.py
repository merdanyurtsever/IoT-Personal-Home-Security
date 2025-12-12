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
        from .face import FaceRecognizer, FaceSecurityPipeline
        from .api import create_app, set_face_service
        import uvicorn
        
        # Initialize face recognizer as the service
        recog_config = config.get("face_recognition", {})
        api_config = config.get("api", {})
        
        # Create a simple service wrapper for the API
        recognizer = FaceRecognizer(
            embedding_backend=recog_config.get("model", "opencv_dnn"),
            detection_backend=recog_config.get("detection_backend", "opencv_dnn"),
            similarity_threshold=recog_config.get("similarity_threshold", 0.6),
        )
        
        # Register faces from watch_list directory if exists
        watch_list_dir = recog_config.get("watch_list_dir", "data/raw/faces/watch_list")
        if Path(watch_list_dir).exists():
            results = recognizer.register_from_directory(watch_list_dir)
            logger.info(f"✓ Registered {sum(results.values())} faces from watch list")
        
        logger.info("✓ Face recognition service initialized")
        
        # Start API server
        app = create_app()
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
                x, y, w, h = face.bbox
                logger.info(f"  [{i+1}] pos=({x},{y}) size={w}x{h} conf={face.confidence:.0%}")
            
            if args.output:
                output = detector.draw_detections(image, faces)
                cv2.imwrite(args.output, output)
                logger.info(f"Saved: {args.output}")
                
        elif args.camera:
            # Live camera detection
            from .sensors.camera import Camera
            
            with Camera() as camera:
                logger.info("Camera started. Press 'q' to quit.")
                
                for frame in camera.stream():
                    faces = detector.detect(frame.image)
                    output = detector.draw_detections(frame.image, faces)
                    cv2.imshow("Face Detection", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
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


def cmd_recognize(args):
    """Run face recognition on image or camera."""
    import numpy as np
    
    try:
        from .face import FaceRecognizer
        import cv2
        
        recognizer = FaceRecognizer(
            embedding_backend=args.embedding_backend,
            detection_backend=args.detection_backend,
            similarity_threshold=args.threshold,
        )
        logger.info(f"Detection: {args.detection_backend}, Embedding: {args.embedding_backend}")
        logger.info(f"Threshold: {args.threshold}, Embedding dim: {recognizer.embedding_dim}D")
        
        # Load watch list
        watch_list_dir = Path("data/raw/faces/watch_list")
        if watch_list_dir.exists():
            results = recognizer.register_from_directory(str(watch_list_dir))
            logger.info(f"Loaded watch list: {results}")
        
        if args.image:
            # Process single image
            image = cv2.imread(args.image)
            if image is None:
                logger.error(f"Could not load: {args.image}")
                sys.exit(1)
            
            result = recognizer.recognize_face(image)
            logger.info(f"Identity: {result.identity}")
            logger.info(f"Category: {result.category}")
            logger.info(f"Confidence: {result.confidence:.1%}")
            logger.info(f"Should alert: {result.should_alert}")
                
        elif args.camera:
            # Live camera recognition
            from .sensors.camera import Camera
            
            with Camera() as camera:
                logger.info("Camera started. Press 'q' to quit.")
                
                for frame in camera.stream():
                    result = recognizer.recognize_face(frame.image)
                    
                    # Draw results
                    output = frame.image.copy()
                    if result.bbox:
                        x, y, w, h = result.bbox
                        color = (0, 0, 255) if result.should_alert else (0, 255, 0)
                        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                        label = f"{result.identity}: {result.confidence:.0%}"
                        cv2.putText(output, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.imshow("Face Recognition", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                cv2.destroyAllWindows()
        else:
            # Quick test
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = recognizer.recognize_face(test_img)
            logger.info(f"✓ Recognition working (embedding: {recognizer.embedding_dim}D)")
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def cmd_compare(args):
    """Compare face recognition models."""
    logger.info("Running model comparison...")
    
    try:
        # Import and run the comparison script
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "scripts/model_comparison.py"],
            check=False,
        )
        sys.exit(result.returncode)
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        sys.exit(1)
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
        recognizer = FaceRecognizer(embedding_backend="opencv_dnn")
        logger.info(f"  ✓ OK (embedding: {recognizer.embedding_dim}D)")
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
            from .sensors.camera import Camera
            with Camera() as camera:
                frame = camera.read()
                if frame:
                    logger.info(f"  ✓ OK ({frame.width}x{frame.height})")
                    results.append(("Camera", True))
                else:
                    logger.info("  ✗ No frame captured")
                    results.append(("Camera", False))
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
    
    # recognize
    recog_p = subparsers.add_parser("recognize", help="Face recognition")
    recog_p.add_argument("-i", "--image", help="Image file")
    recog_p.add_argument("--camera", action="store_true", help="Use camera")
    recog_p.add_argument("-d", "--detection-backend", default="opencv_dnn",
                         choices=["haar_cascade", "mediapipe", "opencv_dnn", "dlib", "dlib_cnn"],
                         help="Face detection backend")
    recog_p.add_argument("-e", "--embedding-backend", default="opencv_dnn",
                         choices=["opencv_dnn", "dlib", "tflite", "mobilenetv2"],
                         help="Face embedding/recognition backend")
    recog_p.add_argument("-t", "--threshold", type=float, default=0.6,
                         help="Recognition similarity threshold (0.0-1.0)")
    
    # compare - model comparison tool
    compare_p = subparsers.add_parser("compare", help="Compare face recognition models")
    compare_p.add_argument("--models", nargs="*", 
                          choices=["opencv_dnn", "dlib", "tflite", "mobilenetv2", "all"],
                          default=["all"],
                          help="Models to compare (default: all available)")
    
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
        "recognize": cmd_recognize,
        "compare": cmd_compare,
        "test": cmd_test,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
