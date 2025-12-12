#!/usr/bin/env python3
"""Standalone CLI for Face Detection & Recognition module.

This module can run independently from the main security system.
Give this folder to someone and they can run face detection/recognition.

Usage:
    python -m src.face detect --image test.jpg
    python -m src.face detect --camera
    python -m src.face recognize --image test.jpg
    python -m src.face register --name "John" --images path/to/images/
    python -m src.face list
    python -m src.face api --port 8000

Examples:
    # Detect faces in an image
    python -m src.face detect --image photo.jpg --output result.jpg

    # Live camera detection
    python -m src.face detect --camera

    # Recognize faces (after registering some)
    python -m src.face recognize --image photo.jpg

    # Register a person's face
    python -m src.face register --name "Alice" --images ./alice_photos/

    # Start API server for mobile app
    python -m src.face api --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_detect(args):
    """Detect faces in image or camera stream."""
    from .detection import FaceDetector
    
    detector = FaceDetector(backend=args.backend)
    logger.info(f"Using {args.backend} backend")
    
    if args.image:
        # Process single image
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Could not load image: {args.image}")
            sys.exit(1)
        
        faces = detector.detect(image)
        logger.info(f"Found {len(faces)} face(s)")
        
        for i, face in enumerate(faces):
            x, y, w, h = face.bbox
            logger.info(f"  [{i+1}] pos=({x},{y}) size={w}x{h} conf={face.confidence:.0%}")
        
        if args.output:
            output = detector.draw_detections(image, faces)
            cv2.imwrite(args.output, output)
            logger.info(f"Saved to: {args.output}")
        
        return len(faces)
            
    elif args.camera:
        # Live camera detection
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {args.camera_id}")
            sys.exit(1)
        
        logger.info("Camera started. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                faces = detector.detect(frame)
                output = detector.draw_detections(frame, faces)
                
                # Add FPS counter
                cv2.putText(output, f"Faces: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Face Detection - Press 'q' to quit", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        # Quick self-test
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(test_img)
        logger.info("✓ Detection module working")


def cmd_recognize(args):
    """Recognize faces in an image."""
    from .recognition import FaceRecognizer
    
    recognizer = FaceRecognizer(
        embedding_backend=args.backend,
        detection_backend="opencv_dnn",
        similarity_threshold=args.threshold,
    )
    
    # Load registered faces if directory exists
    watch_list_dir = Path(args.watch_list)
    if watch_list_dir.exists():
        results = recognizer.register_from_directory(str(watch_list_dir))
        total = sum(results.values())
        logger.info(f"Loaded {total} faces from {len(results)} identities")
    else:
        logger.warning(f"Watch list directory not found: {watch_list_dir}")
    
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Could not load image: {args.image}")
            sys.exit(1)
        
        results = recognizer.recognize_faces(image)
        logger.info(f"Recognition results ({len(results)} faces):")
        
        for i, result in enumerate(results):
            identity = result.identity or "Unknown"
            logger.info(f"  [{i+1}] {identity} (confidence: {result.confidence:.0%})")
    
    elif args.camera:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {args.camera_id}")
            sys.exit(1)
        
        logger.info("Camera started. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = recognizer.recognize_faces(frame)
                
                # Draw results on frame
                for result in results:
                    # This would require bbox info from detection
                    pass
                
                cv2.imshow("Face Recognition - Press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        logger.info(f"Registered identities: {recognizer.get_registered_identities()}")


def cmd_register(args):
    """Register a person's face from images."""
    from .recognition import FaceRecognizer
    
    recognizer = FaceRecognizer(
        embedding_backend=args.backend,
        detection_backend="opencv_dnn",
    )
    
    images_path = Path(args.images)
    if not images_path.exists():
        logger.error(f"Path not found: {images_path}")
        sys.exit(1)
    
    if images_path.is_file():
        # Single image
        image = cv2.imread(str(images_path))
        if image is None:
            logger.error(f"Could not load image: {images_path}")
            sys.exit(1)
        
        success = recognizer.register_face(args.name, image)
        if success:
            logger.info(f"✓ Registered {args.name} from {images_path.name}")
        else:
            logger.error(f"✗ No face found in {images_path.name}")
    else:
        # Directory of images
        count = 0
        for img_path in images_path.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                image = cv2.imread(str(img_path))
                if image is not None and recognizer.register_face(args.name, image):
                    count += 1
                    logger.info(f"  ✓ {img_path.name}")
        
        logger.info(f"Registered {count} face(s) for {args.name}")


def cmd_list(args):
    """List registered identities."""
    from .recognition import FaceRecognizer
    
    recognizer = FaceRecognizer()
    
    # Load from watch list if exists
    watch_list_dir = Path(args.watch_list)
    if watch_list_dir.exists():
        results = recognizer.register_from_directory(str(watch_list_dir))
        
        logger.info("Registered identities:")
        for name, count in results.items():
            logger.info(f"  - {name}: {count} face(s)")
        logger.info(f"Total: {len(results)} identities")
    else:
        logger.info("No watch list directory found")


def cmd_api(args):
    """Start the face recognition API server."""
    try:
        import uvicorn
        from .api import create_app
        
        app = create_app()
        
        logger.info(f"Starting Face Recognition API")
        logger.info(f"  URL: http://{args.host}:{args.port}")
        logger.info(f"  Docs: http://{args.host}:{args.port}/docs")
        
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install uvicorn fastapi")
        sys.exit(1)


def cmd_test(args):
    """Run self-tests on the face module."""
    logger.info("Running face module self-tests...")
    
    # Test detection
    logger.info("[1/3] Testing face detection...")
    try:
        from .detection import FaceDetector
        detector = FaceDetector(backend="opencv_dnn")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(test_img)
        logger.info("  ✓ Detection OK")
    except Exception as e:
        logger.error(f"  ✗ Detection failed: {e}")
        return 1
    
    # Test recognition
    logger.info("[2/3] Testing face recognition...")
    try:
        from .recognition import FaceRecognizer
        recognizer = FaceRecognizer()
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        recognizer.get_embedding(test_img)
        logger.info("  ✓ Recognition OK")
    except Exception as e:
        logger.error(f"  ✗ Recognition failed: {e}")
        return 1
    
    # Test pipeline
    logger.info("[3/3] Testing pipeline...")
    try:
        from .pipeline import FaceSecurityPipeline
        pipeline = FaceSecurityPipeline()
        logger.info("  ✓ Pipeline OK")
    except Exception as e:
        logger.error(f"  ✗ Pipeline failed: {e}")
        return 1
    
    logger.info("All tests passed! ✓")
    return 0


def main():
    """Main entry point for face module CLI."""
    parser = argparse.ArgumentParser(
        prog="face",
        description="Face Detection & Recognition Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.face detect --image photo.jpg
  python -m src.face detect --camera
  python -m src.face recognize --image photo.jpg
  python -m src.face register --name "Alice" --images ./photos/
  python -m src.face api --port 8000
  python -m src.face test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect faces")
    detect_parser.add_argument("--image", "-i", help="Input image path")
    detect_parser.add_argument("--camera", "-c", action="store_true", help="Use camera")
    detect_parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    detect_parser.add_argument("--output", "-o", help="Output image path")
    detect_parser.add_argument("--backend", "-b", default="opencv_dnn",
                               choices=["opencv_dnn", "haar_cascade", "mediapipe", "dlib"],
                               help="Detection backend")
    
    # Recognize command
    recognize_parser = subparsers.add_parser("recognize", help="Recognize faces")
    recognize_parser.add_argument("--image", "-i", help="Input image path")
    recognize_parser.add_argument("--camera", "-c", action="store_true", help="Use camera")
    recognize_parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    recognize_parser.add_argument("--backend", "-b", default="opencv_dnn", help="Embedding backend")
    recognize_parser.add_argument("--threshold", "-t", type=float, default=0.6, help="Similarity threshold")
    recognize_parser.add_argument("--watch-list", default="data/raw/faces/watch_list", help="Watch list directory")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a person's face")
    register_parser.add_argument("--name", "-n", required=True, help="Person's name")
    register_parser.add_argument("--images", "-i", required=True, help="Image or directory of images")
    register_parser.add_argument("--backend", "-b", default="opencv_dnn", help="Embedding backend")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered identities")
    list_parser.add_argument("--watch-list", default="data/raw/faces/watch_list", help="Watch list directory")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run self-tests")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Route to command handler
    commands = {
        "detect": cmd_detect,
        "recognize": cmd_recognize,
        "register": cmd_register,
        "list": cmd_list,
        "api": cmd_api,
        "test": cmd_test,
    }
    
    handler = commands.get(args.command)
    if handler:
        result = handler(args)
        sys.exit(result if result else 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
