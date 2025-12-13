#!/usr/bin/env python3
"""Face Recognition Viewfinder.

A real-time face recognition system using ArcFace (InsightFace buffalo_l model).
Can be used standalone or integrated into other applications.

Usage:
    python -m src.face.viewfinder [--watch-list PATH] [--threshold 0.35]
    
Controls:
    B      : Toggle brightness enhancement
    R      : Re-enroll faces from watch list
    +/-    : Adjust recognition threshold
    Q/ESC  : Quit
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def find_watch_list_dir() -> Optional[Path]:
    """Find watch list directory from common locations."""
    # Try paths relative to current working directory
    candidates = [
        Path("watch_list"),
        Path("faces"),
        Path("data/watch_list"),
        Path("data/raw/faces/watch_list"),
    ]
    # Also try relative to this module
    module_dir = Path(__file__).parent
    candidates.extend([
        module_dir.parent.parent / "data" / "raw" / "faces" / "watch_list",
        module_dir.parent.parent / "data" / "watch_list",
    ])
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def enhance_brightness(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE brightness enhancement to image (LAB color space)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def expand_bbox(x: int, y: int, w: int, h: int, 
                frame_shape: Tuple[int, ...], margin: float = 0.25) -> Tuple[int, int, int, int]:
    """Expand face bounding box by margin for better embedding accuracy."""
    frame_h, frame_w = frame_shape[:2]
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    new_x = max(0, x - margin_w)
    new_y = max(0, y - margin_h)
    new_w = min(frame_w - new_x, w + 2 * margin_w)
    new_h = min(frame_h - new_y, h + 2 * margin_h)
    return new_x, new_y, new_w, new_h


class ArcFaceRecognizer:
    """ArcFace-based face recognition using InsightFace.
    
    Uses buffalo_l model (ResNet-50 backbone) for highest accuracy.
    Provides both face detection and 512D embedding extraction.
    """
    
    def __init__(self, model_name: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640)):
        """Initialize ArcFace recognizer.
        
        Args:
            model_name: InsightFace model pack name
                - buffalo_l: Highest accuracy (ResNet-50, ~280MB)
                - buffalo_s: Faster, smaller (~120MB)
                - buffalo_sc: Smallest, fastest (~15MB)
            det_size: Detection input size (width, height)
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace not installed. Install with:\n"
                "  pip install insightface onnxruntime"
            )
        
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.rec_model = self.app.models.get('recognition')
        self.embedding_dim = 512
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of (x, y, width, height) tuples for each detected face
        """
        faces = self.app.get(frame)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D face embedding.
        
        Args:
            face_image: BGR face image (can be cropped or full image)
            
        Returns:
            Normalized 512D embedding vector or None if extraction fails
        """
        try:
            # Try full pipeline (detection + recognition)
            faces = self.app.get(face_image)
            
            if not faces:
                # Try with padding for small/cropped faces
                h, w = face_image.shape[:2]
                if max(h, w) < 200:
                    pad = max(h, w)
                    padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                    padded[pad:pad+h, pad:pad+w] = face_image
                    faces = self.app.get(padded)
            
            if not faces and self.rec_model is not None:
                # Direct extraction on pre-cropped face
                return self._extract_direct(face_image)
            
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            
            return None
        except Exception:
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model (for pre-cropped faces)."""
        try:
            # Preprocess: resize, BGR->RGB, normalize
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
            face_input = (face_input - 127.5) / 127.5
            face_input = np.expand_dims(face_input, axis=0)
            
            embedding = self.rec_model.session.run(
                self.rec_model.output_names,
                {self.rec_model.input_name: face_input}
            )[0][0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception:
            return None


class FaceDatabase:
    """Simple in-memory face database for enrolled faces."""
    
    def __init__(self):
        self.faces: List[Tuple[str, np.ndarray]] = []
    
    def add(self, name: str, embedding: np.ndarray) -> None:
        """Add a face to the database."""
        self.faces.append((name, embedding))
    
    def clear(self) -> None:
        """Clear all enrolled faces."""
        self.faces.clear()
    
    def find_match(self, embedding: np.ndarray, threshold: float = 0.35) -> Tuple[Optional[str], float]:
        """Find the best matching face.
        
        Args:
            embedding: Query embedding
            threshold: Minimum similarity for a match
            
        Returns:
            Tuple of (name, score) or (None, 0.0) if no match
        """
        best_name = None
        best_score = 0.0
        
        for name, enrolled_emb in self.faces:
            score = cosine_similarity(embedding, enrolled_emb)
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score >= threshold:
            return best_name, best_score
        return None, best_score
    
    def __len__(self) -> int:
        return len(self.faces)


def load_watch_list(recognizer: ArcFaceRecognizer, 
                    watch_list_dir: Path,
                    apply_enhancement: bool = True) -> FaceDatabase:
    """Load and enroll faces from watch list directory.
    
    Args:
        recognizer: ArcFace recognizer instance
        watch_list_dir: Directory containing face images
        apply_enhancement: Whether to apply brightness enhancement
        
    Returns:
        FaceDatabase with enrolled faces
    """
    db = FaceDatabase()
    
    if not watch_list_dir.exists():
        print(f"Watch list directory not found: {watch_list_dir}")
        return db
    
    print(f"\nLoading watch list from: {watch_list_dir}")
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in watch_list_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Apply brightness enhancement
            if apply_enhancement:
                img = enhance_brightness(img)
            
            # Detect and extract
            faces = recognizer.detect_faces(img)
            if faces:
                x, y, w, h = faces[0]
                # Expand bbox for better accuracy
                ex, ey, ew, eh = expand_bbox(x, y, w, h, img.shape)
                face_crop = img[ey:ey+eh, ex:ex+ew]
                embedding = recognizer.extract_embedding(face_crop)
            else:
                # Try full image
                embedding = recognizer.extract_embedding(img)
            
            if embedding is not None:
                db.add(img_path.stem, embedding)
                print(f"  ✓ Enrolled: {img_path.name}")
            else:
                print(f"  ✗ Failed: {img_path.name}")
    
    print(f"Enrolled {len(db)} face(s)")
    return db


def run_viewfinder(watch_list_dir: Optional[Path] = None,
                   threshold: float = 0.35,
                   camera_id: int = 0,
                   save_dir: Optional[Path] = None) -> None:
    """Run the face recognition viewfinder.
    
    Args:
        watch_list_dir: Directory containing watch list images (auto-detected if None)
        threshold: Recognition similarity threshold (0.0-1.0)
        camera_id: Camera device ID
        save_dir: Directory to save captured frames (default: ./captures)
    """
    # Find watch list directory if not provided
    if watch_list_dir is None:
        watch_list_dir = find_watch_list_dir()
        if watch_list_dir is None:
            watch_list_dir = Path("watch_list")
            print(f"[WARN] No watch list found. Create '{watch_list_dir}/' with face images.")
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path("captures")
    print("=" * 60)
    print("FACE RECOGNITION VIEWFINDER")
    print("=" * 60)
    print("\nModel: ArcFace (buffalo_l, 512D)")
    
    # Initialize recognizer
    print("\nLoading model...")
    recognizer = ArcFaceRecognizer(model_name="buffalo_l")
    print("  ✓ Model loaded")
    
    # Load watch list
    database = load_watch_list(recognizer, watch_list_dir)
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Get resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera resolution: {width}x{height}")
    
    print("\nControls:")
    print("  B      : Toggle brightness enhancement")
    print("  R      : Re-enroll faces")
    print("  S      : Save current frame")
    print("  SPACE  : Pause/Resume")
    print("  +/-    : Adjust threshold")
    print("  Q/ESC  : Quit")
    print()
    
    # State
    enhance_brightness_enabled = True
    frame_times = []
    paused = False
    last_frame = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        else:
            if last_frame is None:
                continue
            frame = last_frame.copy()
        
        frame_start = time.time()
        
        # Apply brightness enhancement
        if enhance_brightness_enabled:
            frame = enhance_brightness(frame)
        
        # Detect faces
        faces = recognizer.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Expand bbox for embedding
            ex, ey, ew, eh = expand_bbox(x, y, w, h, frame.shape)
            ex2, ey2 = min(frame.shape[1], ex + ew), min(frame.shape[0], ey + eh)
            
            if ex2 <= ex or ey2 <= ey:
                continue
            
            face_crop = frame[ey:ey2, ex:ex2]
            embedding = recognizer.extract_embedding(face_crop)
            
            # Match against database
            if embedding is not None and len(database) > 0:
                match_name, score = database.find_match(embedding, threshold)
                
                if match_name:
                    color = (0, 0, 255)  # Red - threat
                    label = f"THREAT {score:.0%}"
                else:
                    color = (0, 255, 0)  # Green - safe
                    label = f"Safe {score:.0%}"
            else:
                color = (128, 128, 128)  # Gray - no database
                label = "No DB" if len(database) == 0 else "No embed"
            
            # Draw on original detection box
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            
            # Label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS calculation
        frame_time = (time.time() - frame_start) * 1000
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1000 / np.mean(frame_times) if frame_times else 0
        
        # Info overlay
        brightness_status = "ON" if enhance_brightness_enabled else "OFF"
        info_lines = [
            f"ArcFace (512D) | Threshold: {threshold:.2f}",
            f"Brightness: {brightness_status} | FPS: {fps:.1f}",
            f"Enrolled: {len(database)} face(s)",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Face Recognition - Press Q to quit", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('b'):
            enhance_brightness_enabled = not enhance_brightness_enabled
            print(f"Brightness: {'ON' if enhance_brightness_enabled else 'OFF'}")
        elif key == ord('r'):
            print("\nRe-enrolling faces...")
            database = load_watch_list(recognizer, watch_list_dir)
        elif key == ord('+') or key == ord('='):
            threshold = min(1.0, threshold + 0.05)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.0, threshold - 0.05)
            print(f"Threshold: {threshold:.2f}")
        elif key == ord('s'):
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"Saved: {filename}")
        elif key == ord(' '):
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Face Recognition Viewfinder using ArcFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.face.viewfinder
  python -m src.face.viewfinder --watch-list /path/to/faces
  python -m src.face.viewfinder --threshold 0.4
        """
    )
    parser.add_argument(
        "--watch-list", "-w",
        type=Path,
        default=None,
        help="Directory containing watch list face images (auto-detected)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.35,
        help="Recognition threshold (0.0-1.0, default: 0.35)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--save-dir", "-s",
        type=Path,
        default=None,
        help="Directory for saved frames (default: ./captures)"
    )
    
    args = parser.parse_args()
    run_viewfinder(
        watch_list_dir=args.watch_list if args.watch_list else None,
        threshold=args.threshold,
        camera_id=args.camera,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
