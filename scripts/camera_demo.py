#!/usr/bin/env python3
"""Camera demo with face detection and recognition visualization.

Shows live camera feed with:
- RED rectangle: Watch list match (threat detected)
- GREEN rectangle: No match (safe)
- YELLOW rectangle: Threat profile (attribute match)

Press 'q' to quit.
"""

import cv2
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visual import (
    FaceDetector,
    FaceRecognizer,
    FaceCategory,
)

# Colors (BGR format)
COLOR_THREAT = (0, 0, 255)      # Red - Watch list match
COLOR_SAFE = (0, 255, 0)        # Green - No match
COLOR_PROFILE = (0, 255, 255)   # Yellow - Threat profile (attribute match)
COLOR_TEXT_BG = (0, 0, 0)       # Black background for text


def draw_face_box(frame: np.ndarray, box: tuple, label: str, category: FaceCategory, confidence: float):
    """Draw a colored rectangle around a face with label."""
    x, y, w, h = box
    
    # Choose color based on category
    if category == FaceCategory.WATCH_LIST:
        color = COLOR_THREAT
        status = "THREAT"
    elif category == FaceCategory.THREAT_PROFILE:
        color = COLOR_PROFILE
        status = "PROFILE"
    else:
        color = COLOR_SAFE
        status = "SAFE"
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    # Prepare label text
    if label and label != "unknown":
        text = f"{status}: {label} ({confidence:.0%})"
    else:
        text = f"{status} ({confidence:.0%})"
    
    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw text background
    cv2.rectangle(
        frame, 
        (x, y - text_height - 10), 
        (x + text_width + 10, y), 
        color, 
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        frame, 
        text, 
        (x + 5, y - 5), 
        font, 
        font_scale, 
        (255, 255, 255),  # White text
        thickness
    )


def main():
    print("=" * 60)
    print("Face Detection Camera Demo")
    print("=" * 60)
    print()
    print("Colors:")
    print("  RED    = Watch list match (THREAT)")
    print("  GREEN  = No match (SAFE)")
    print("  YELLOW = Threat profile (ATTRIBUTE MATCH)")
    print()
    print("Press 'q' to quit")
    print("=" * 60)
    print()
    
    # Initialize detector with higher confidence threshold
    print("Initializing face detector (OpenCV DNN)...")
    detector = FaceDetector(backend="opencv_dnn", confidence_threshold=0.75)
    
    # Initialize recognizer with proper threshold to avoid false positives
    # Higher threshold = stricter matching, fewer false positives
    # 0.6 = good balance between accuracy and expression tolerance
    print("Initializing face recognizer (OpenCV DNN)...")
    recognizer = FaceRecognizer(
        embedding_backend="opencv_dnn",
        detection_backend="opencv_dnn",
        similarity_threshold=0.6,
    )
    
    # Load watch list faces
    watch_list_dir = project_root / "data" / "raw" / "faces" / "watch_list"
    if watch_list_dir.exists():
        print(f"Loading watch list from: {watch_list_dir}")
        
        results = recognizer.register_from_directory(str(watch_list_dir))
        for name, count in results.items():
            print(f"  Enrolled: {name} ({count} images)")
        
        print(f"Loaded {sum(results.values())} watch list face(s)")
    else:
        print(f"No watch list directory found at: {watch_list_dir}")
    
    print()
    identities = recognizer.get_registered_identities()
    print(f"Total enrolled: {len(identities)} person(s)")
    print(f"Enrolled names: {identities}")
    print()
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Make sure a camera is connected and not in use by another application.")
        return 1
    
    # Don't set resolution - use camera's native resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera native resolution: {actual_width}x{actual_height}")
    
    print("Camera opened successfully!")
    print("Starting face detection loop...")
    print()
    
    frame_count = 0
    last_detection = None  # Track last valid detection for smoothing
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Process each detected face
            for face in faces:
                x, y, w, h = face.bbox
                
                # Extract face region with padding for better recognition
                # Add 30% padding around the face for context
                pad_x = int(w * 0.3)
                pad_y = int(h * 0.3)
                y1 = max(0, y - pad_y)
                y2 = min(frame.shape[0], y + h + pad_y)
                x1 = max(0, x - pad_x)
                x2 = min(frame.shape[1], x + w + pad_x)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # Recognize face
                    result = recognizer.recognize_face(face_img, detect_attributes=False)
                    
                    # Draw box with appropriate color
                    draw_face_box(
                        frame, 
                        face.bbox, 
                        result.identity,
                        result.category,
                        result.confidence
                    )
                    
                    # Log threats
                    if result.should_alert and frame_count % 30 == 0:  # Log every 30 frames
                        print(f"⚠️  THREAT DETECTED: {result.identity} (confidence: {result.confidence:.0%})")
            
            # Show frame
            cv2.imshow("Face Detection - Press 'q' to quit", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
