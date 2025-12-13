#!/usr/bin/env python3
"""Live Camera Model Comparison.

Runs face recognition models on live camera feed so you can visually assess
their performance in real-time.

Controls:
    1-4: Switch between models
    SPACE: Cycle through models
    R: Re-enroll faces from watch_list
    Q/ESC: Quit
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2


def load_watch_list_images() -> List[Tuple[str, np.ndarray]]:
    """Dynamically load ALL images from watch_list folder."""
    watch_list_dir = project_root / "data" / "raw" / "faces" / "watch_list"
    images = []
    
    if not watch_list_dir.exists():
        print(f"Watch list directory not found: {watch_list_dir}")
        return images
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in watch_list_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img_path.name, img))
                print(f"  Loaded: {img_path.name}")
    
    return images


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE preprocessing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def expand_face_bbox(x, y, w, h, frame_shape, margin=0.25):
    """Expand face bounding box by margin for better embedding accuracy.
    
    Adding margin around the face improves recognition accuracy by:
    1. Including more facial context (forehead, chin, ears)
    2. Reducing edge artifacts from tight crops
    3. Giving alignment algorithms more room to work
    """
    frame_h, frame_w = frame_shape[:2]
    
    # Calculate margin in pixels
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    # Expand bbox with margin, clamp to frame bounds
    new_x = max(0, x - margin_w)
    new_y = max(0, y - margin_h)
    new_w = min(frame_w - new_x, w + 2 * margin_w)
    new_h = min(frame_h - new_y, h + 2 * margin_h)
    
    return new_x, new_y, new_w, new_h


class ModelBackend:
    """Base class for model backends."""
    name: str = "Base"
    embedding_size: int = 0
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError


class OpenFaceBackend(ModelBackend):
    """OpenFace embeddings with RetinaFace detection for stability."""
    name = "OpenFace (128D)"
    embedding_size = 128
    
    def __init__(self):
        from src.face.recognition.embeddings import OpenCVDNNEmbeddingBackend
        self.embedder = OpenCVDNNEmbeddingBackend()
        # Use InsightFace's RetinaFace for more stable detection
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.use_retinaface = True
        except ImportError:
            from src.face.detection import OpenCVDNNDetector
            self.detector = OpenCVDNNDetector()
            self.use_retinaface = False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            return self.embedder.extract(face_image)
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.use_retinaface:
            faces = self.app.get(frame)
            result = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                result.append((x1, y1, x2 - x1, y2 - y1))
            return result
        else:
            faces = self.detector.detect(frame)
            return [(f.x, f.y, f.width, f.height) for f in faces]


class OpenFaceCLAHEBackend(ModelBackend):
    """OpenFace with CLAHE preprocessing and RetinaFace detection."""
    name = "OpenFace+CLAHE (128D)"
    embedding_size = 128
    
    def __init__(self):
        from src.face.recognition.embeddings import OpenCVDNNEmbeddingBackend
        self.embedder = OpenCVDNNEmbeddingBackend()
        # Use InsightFace's RetinaFace for more stable detection
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.use_retinaface = True
        except ImportError:
            from src.face.detection import OpenCVDNNDetector
            self.detector = OpenCVDNNDetector()
            self.use_retinaface = False
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            preprocessed = apply_clahe(face_image)
            return self.embedder.extract(preprocessed)
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.use_retinaface:
            faces = self.app.get(frame)
            result = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                result.append((x1, y1, x2 - x1, y2 - y1))
            return result
        else:
            faces = self.detector.detect(frame)
            return [(f.x, f.y, f.width, f.height) for f in faces]


class FaceNetBackend(ModelBackend):
    """FaceNet via facenet-pytorch."""
    name = "FaceNet (512D)"
    embedding_size = 512
    
    def __init__(self):
        from facenet_pytorch import InceptionResnetV1, MTCNN
        import torch
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.torch = torch
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            face = cv2.resize(face_image, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5
            
            face_tensor = self.torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with self.torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy().flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb)
        
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces


class ArcFaceBackend(ModelBackend):
    """ArcFace via InsightFace - High accuracy model."""
    name = "ArcFace (512D)"
    embedding_size = 512
    
    def __init__(self):
        from insightface.app import FaceAnalysis
        # Use buffalo_l for highest accuracy (ResNet-50 backbone)
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        # Get the recognition model for direct extraction on crops
        self.rec_model = self.app.models.get('recognition')
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            # First try with InsightFace's full pipeline (detection + recognition)
            faces = self.app.get(face_image)
            
            if not faces:
                # If no face detected, image might be a pre-cropped face
                # Try with padding to give detector more context
                h, w = face_image.shape[:2]
                if max(h, w) < 200:
                    pad = max(h, w)
                    padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                    padded[pad:pad+h, pad:pad+w] = face_image
                    faces = self.app.get(padded)
            
            if not faces and self.rec_model is not None:
                # Still no detection - use recognition model directly on crop
                return self._extract_direct(face_image)
            
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            return None
        except:
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model on pre-cropped face."""
        try:
            # Preprocess: resize to 112x112, convert BGR->RGB, normalize
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
            face_input = (face_input - 127.5) / 127.5  # Normalize to [-1, 1]
            face_input = np.expand_dims(face_input, axis=0)
            
            # Run recognition model
            embedding = self.rec_model.session.run(
                self.rec_model.output_names,
                {self.rec_model.input_name: face_input}
            )[0][0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self.app.get(frame)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result


class MobileFaceNetBackend(ModelBackend):
    """MobileFaceNet via InsightFace - optimized for mobile/embedded."""
    name = "MobileFaceNet (512D)"
    embedding_size = 512
    
    def __init__(self):
        from insightface.app import FaceAnalysis
        # buffalo_s uses MobileFaceNet for recognition (smaller, faster than buffalo_l)
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller detection for speed
        self.rec_model = self.app.models.get('recognition')
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = self.app.get(face_image)
            
            if not faces:
                # Try with padding for small cropped faces
                h, w = face_image.shape[:2]
                if max(h, w) < 200:
                    pad = max(h, w)
                    padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                    padded[pad:pad+h, pad:pad+w] = face_image
                    faces = self.app.get(padded)
            
            if not faces and self.rec_model is not None:
                return self._extract_direct(face_image)
            
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            return None
        except:
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model on pre-cropped face."""
        try:
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
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self.app.get(frame)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result


class RetinaFaceBackend(ModelBackend):
    """RetinaFace detection + ArcFace recognition via InsightFace."""
    name = "RetinaFace+ArcFace (512D)"
    embedding_size = 512
    
    def __init__(self):
        from insightface.app import FaceAnalysis
        # buffalo_l uses RetinaFace for detection (highest accuracy)
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.rec_model = self.app.models.get('recognition')
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = self.app.get(face_image)
            
            if not faces:
                h, w = face_image.shape[:2]
                if max(h, w) < 200:
                    pad = max(h, w)
                    padded = np.zeros((h + 2*pad, w + 2*pad, 3), dtype=np.uint8)
                    padded[pad:pad+h, pad:pad+w] = face_image
                    faces = self.app.get(padded)
            
            if not faces and self.rec_model is not None:
                return self._extract_direct(face_image)
            
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            return None
        except:
            return None
    
    def _extract_direct(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from recognition model on pre-cropped face."""
        try:
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
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self.app.get(frame)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result


class YOLOFaceBackend(ModelBackend):
    """YOLOv8 Face detection + InsightFace recognition."""
    name = "YOLO+ArcFace (512D)"
    embedding_size = 512
    
    def __init__(self):
        from ultralytics import YOLO
        from insightface.app import FaceAnalysis
        
        # YOLOv8 face model (download on first use)
        self.detector = YOLO('yolov8n-face.pt')
        
        # ArcFace for recognition
        self.recognizer = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=-1)
    
    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = self.recognizer.get(face_image)
            if faces:
                embedding = faces[0].embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            return None
        except:
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            results = self.detector(frame, verbose=False)
            faces = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
        except:
            return []


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 60)
    print("LIVE CAMERA MODEL COMPARISON")
    print("=" * 60)
    print()
    
    # Load available models
    print("Loading models...")
    models: List[ModelBackend] = []
    
    # Always available: OpenFace variants
    try:
        models.append(OpenFaceBackend())
        print(f"  ✓ {models[-1].name}")
    except Exception as e:
        print(f"  ✗ OpenFace: {e}")
    
    try:
        models.append(OpenFaceCLAHEBackend())
        print(f"  ✓ {models[-1].name}")
    except Exception as e:
        print(f"  ✗ OpenFace+CLAHE: {e}")
    
    # Optional: FaceNet (requires PyTorch)
    try:
        models.append(FaceNetBackend())
        print(f"  ✓ {models[-1].name}")
    except ImportError:
        print("  - FaceNet: not installed (pip install facenet-pytorch)")
    except Exception as e:
        print(f"  ✗ FaceNet: {e}")
    
    # Optional: ArcFace (InsightFace buffalo_sc - balanced)
    try:
        models.append(ArcFaceBackend())
        print(f"  ✓ {models[-1].name}")
    except ImportError:
        print("  - ArcFace: not installed (pip install insightface onnxruntime)")
    except Exception as e:
        print(f"  ✗ ArcFace: {e}")
    
    # Optional: MobileFaceNet (InsightFace buffalo_s - fast, for embedded)
    try:
        models.append(MobileFaceNetBackend())
        print(f"  ✓ {models[-1].name}")
    except ImportError:
        print("  - MobileFaceNet: not installed (pip install insightface onnxruntime)")
    except Exception as e:
        print(f"  ✗ MobileFaceNet: {e}")
    
    # Optional: RetinaFace + ArcFace (InsightFace buffalo_l - highest accuracy)
    try:
        models.append(RetinaFaceBackend())
        print(f"  ✓ {models[-1].name}")
    except ImportError:
        print("  - RetinaFace: not installed (pip install insightface onnxruntime)")
    except Exception as e:
        print(f"  ✗ RetinaFace: {e}")
    
    # Optional: YOLO + ArcFace (Ultralytics YOLO for detection)
    try:
        models.append(YOLOFaceBackend())
        print(f"  ✓ {models[-1].name}")
    except ImportError:
        print("  - YOLO: not installed (pip install ultralytics)")
    except Exception as e:
        print(f"  ✗ YOLO: {e}")
    
    if not models:
        print("ERROR: No models available!")
        return
    
    print(f"\nLoaded {len(models)} model(s)")
    
    # Load watch list (dynamically - any images in the folder)
    print("\nLoading watch list...")
    watch_list_images = load_watch_list_images()
    
    if not watch_list_images:
        print("WARNING: No images in watch_list folder!")
        print("Add photos to: data/raw/faces/watch_list/")
    
    # Current model index
    current_model_idx = 0
    current_model = models[current_model_idx]
    
    # Helper function to enhance brightness of an image
    def enhance_image(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE brightness enhancement to image."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Enroll watch list faces with current model
    def enroll_faces():
        nonlocal watch_list_embeddings
        watch_list_embeddings = []
        print(f"\nEnrolling faces with {current_model.name}...")
        
        for img_name, original_img in watch_list_images:
            # Apply brightness enhancement to watch list images
            img = enhance_image(original_img)
            
            # Detect faces in the image
            faces = current_model.detect_faces(img)
            if faces:
                x, y, w, h = faces[0]
                # Expand bbox for better embedding accuracy (same as live processing)
                ex, ey, ew, eh = expand_face_bbox(x, y, w, h, img.shape, margin=0.25)
                face_crop = img[max(0,ey):ey+eh, max(0,ex):ex+ew]
                emb = current_model.extract(face_crop)
                if emb is not None:
                    watch_list_embeddings.append((img_name, emb))
                    print(f"  ✓ Enrolled: {img_name}")
                else:
                    print(f"  ✗ Could not extract embedding: {img_name}")
            else:
                # Try extracting directly from full image
                emb = current_model.extract(img)
                if emb is not None:
                    watch_list_embeddings.append((img_name, emb))
                    print(f"  ✓ Enrolled (full image): {img_name}")
                else:
                    print(f"  ✗ No face detected: {img_name}")
    
    watch_list_embeddings = []
    enroll_faces()
    
    # Recognition threshold (optimized per model)
    # Lower = more strict (fewer false positives)
    # Higher = more lenient (fewer false negatives)
    thresholds = {
        "OpenFace (128D)": 0.55,          # 128D needs higher threshold
        "OpenFace+CLAHE (128D)": 0.55,    # CLAHE improves slightly
        "FaceNet (512D)": 0.50,           # FaceNet is well-calibrated
        "ArcFace (512D)": 0.35,           # ArcFace is very discriminative
        "MobileFaceNet (512D)": 0.38,     # MobileFaceNet slightly less
        "RetinaFace+ArcFace (512D)": 0.35, # Same as ArcFace
        "YOLO+ArcFace (512D)": 0.35,      # Same as ArcFace
    }
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Get native resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    print()
    print("Controls:")
    print(f"  1-{min(9, len(models))}    : Select model directly")
    print("  SPACE  : Cycle to next model")
    print("  B      : Toggle brightness enhancement")
    print("  R      : Re-enroll faces (after adding new photos)")
    print("  +/-    : Adjust threshold")
    print("  Q/ESC  : Quit")
    print()
    
    # Brightness enhancement flag
    enhance_brightness = True
    
    # Stats
    frame_times = []
    detection_times = []
    recognition_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # Apply brightness enhancement if enabled
        if enhance_brightness:
            # Convert to LAB color space for better brightness adjustment
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Get current threshold
        threshold = thresholds.get(current_model.name, 0.5)
        
        # Detect faces
        detect_start = time.time()
        faces = current_model.detect_faces(frame)
        detect_time = (time.time() - detect_start) * 1000
        detection_times.append(detect_time)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Expand bbox for better embedding accuracy
            ex, ey, ew, eh = expand_face_bbox(x, y, w, h, frame.shape, margin=0.25)
            
            # Ensure bounds are valid
            ex, ey = max(0, ex), max(0, ey)
            ex2, ey2 = min(frame.shape[1], ex + ew), min(frame.shape[0], ey + eh)
            
            if ex2 <= ex or ey2 <= ey:
                continue
            
            # Use expanded crop for embedding extraction
            face_crop = frame[ey:ey2, ex:ex2]
            
            # Extract embedding
            recog_start = time.time()
            embedding = current_model.extract(face_crop)
            recog_time = (time.time() - recog_start) * 1000
            recognition_times.append(recog_time)
            
            # Compare with watch list
            best_match = None
            best_score = 0.0
            
            if embedding is not None and watch_list_embeddings:
                for name, enrolled_emb in watch_list_embeddings:
                    score = cosine_similarity(embedding, enrolled_emb)
                    if score > best_score:
                        best_score = score
                        best_match = name
            
            # Determine color and label
            if best_score >= threshold:
                # MATCH - RED (threat on watch list)
                color = (0, 0, 255)
                label = f"THREAT {best_score:.0%}"
            else:
                # NO MATCH - GREEN (safe)
                color = (0, 255, 0)
                label = f"Safe {best_score:.0%}"
            
            # Draw rectangle using ORIGINAL detection box (x, y, w, h)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate FPS
        frame_time = (time.time() - frame_start) * 1000
        frame_times.append(frame_time)
        
        # Keep only last 30 frames for stats
        if len(frame_times) > 30:
            frame_times.pop(0)
        if len(detection_times) > 30:
            detection_times.pop(0)
        if len(recognition_times) > 30:
            recognition_times.pop(0)
        
        avg_frame_time = np.mean(frame_times)
        fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        avg_detect = np.mean(detection_times) if detection_times else 0
        avg_recog = np.mean(recognition_times) if recognition_times else 0
        
        # Draw info overlay
        brightness_status = "ON" if enhance_brightness else "OFF"
        info_lines = [
            f"Model: {current_model.name}",
            f"Threshold: {threshold:.2f} (+/- to adjust)",
            f"Brightness: {brightness_status} (B to toggle)",
            f"FPS: {fps:.1f}",
            f"Detect: {avg_detect:.0f}ms | Recog: {avg_recog:.0f}ms",
            f"Enrolled: {len(watch_list_embeddings)} face(s)",
        ]
        
        # Draw model selector
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw model list
        y_offset = height - 30 * len(models) - 10
        for i, model in enumerate(models):
            prefix = ">> " if i == current_model_idx else "   "
            color = (0, 255, 255) if i == current_model_idx else (200, 200, 200)
            cv2.putText(frame, f"{prefix}[{i+1}] {model.name}", 
                       (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("Model Comparison - Press Q to quit", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # Space - cycle models
            current_model_idx = (current_model_idx + 1) % len(models)
            current_model = models[current_model_idx]
            print(f"\nSwitched to: {current_model.name}")
            enroll_faces()
            frame_times.clear()
            detection_times.clear()
            recognition_times.clear()
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
            idx = int(chr(key)) - 1
            if idx < len(models):
                current_model_idx = idx
                current_model = models[current_model_idx]
                print(f"\nSwitched to: {current_model.name}")
                enroll_faces()
                frame_times.clear()
                detection_times.clear()
                recognition_times.clear()
        elif key == ord('b'):  # Toggle brightness enhancement
            enhance_brightness = not enhance_brightness
            print(f"\nBrightness enhancement: {'ON' if enhance_brightness else 'OFF'}")
        elif key == ord('r'):  # Re-enroll
            print("\nReloading watch list...")
            watch_list_images = load_watch_list_images()
            enroll_faces()
        elif key == ord('+') or key == ord('='):
            thresholds[current_model.name] = min(1.0, threshold + 0.05)
            print(f"Threshold: {thresholds[current_model.name]:.2f}")
        elif key == ord('-'):
            thresholds[current_model.name] = max(0.0, threshold - 0.05)
            print(f"Threshold: {thresholds[current_model.name]:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
