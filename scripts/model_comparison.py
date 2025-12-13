#!/usr/bin/env python3
"""Face Recognition Model Comparison Pipeline.

Compares different face recognition models to find the best one for your use case.
Tests: OpenFace, MobileNetV2, ArcFace, and optionally others.

Usage:
    1. Add photos to data/raw/faces/test_same_person/ (multiple photos of YOU)
    2. Add photos to data/raw/faces/test_different_people/ (photos of other people)
    3. Run: python scripts/model_comparison.py

The best model will have:
    - HIGH similarity scores for same person (>0.7)
    - LOW similarity scores for different people (<0.4)
    - Large gap between same/different scores
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


@dataclass
class ModelResult:
    """Results for a single model."""
    name: str
    embedding_size: int
    same_person_scores: List[float]
    different_person_scores: List[float]
    extraction_time_ms: float
    load_time_ms: float
    
    @property
    def same_person_avg(self) -> float:
        return np.mean(self.same_person_scores) if self.same_person_scores else 0.0
    
    @property
    def same_person_min(self) -> float:
        return np.min(self.same_person_scores) if self.same_person_scores else 0.0
    
    @property
    def different_person_avg(self) -> float:
        return np.mean(self.different_person_scores) if self.different_person_scores else 0.0
    
    @property
    def different_person_max(self) -> float:
        return np.max(self.different_person_scores) if self.different_person_scores else 0.0
    
    @property
    def separation_gap(self) -> float:
        """Gap between same-person min and different-person max. Higher is better."""
        return self.same_person_min - self.different_person_max
    
    @property
    def recommended_threshold(self) -> float:
        """Recommended threshold (midpoint between distributions)."""
        return (self.same_person_min + self.different_person_max) / 2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_images(folder: Path) -> List[Tuple[str, np.ndarray]]:
    """Load all images from a folder."""
    images = []
    if not folder.exists():
        return images
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in folder.glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img_path.name, img))
    
    return images


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE preprocessing for better lighting invariance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def test_openface_opencv() -> Optional[ModelResult]:
    """Test OpenFace model via OpenCV DNN."""
    print("\n" + "=" * 60)
    print("Testing: OpenFace (OpenCV DNN) - 128D embeddings")
    print("         (No preprocessing)")
    print("=" * 60)
    
    try:
        from src.face import OpenCVDNNEmbeddingBackend
        
        start = time.time()
        backend = OpenCVDNNEmbeddingBackend()
        load_time = (time.time() - start) * 1000
        
        print(f"  ✓ Model loaded in {load_time:.0f}ms")
        print(f"  ✓ Embedding size: {backend.embedding_dim}D")
        
        return _run_model_test("OpenFace-128D", backend, load_time)
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def test_openface_clahe() -> Optional[ModelResult]:
    """Test OpenFace model with CLAHE preprocessing."""
    print("\n" + "=" * 60)
    print("Testing: OpenFace + CLAHE - 128D embeddings")
    print("         (With CLAHE lighting normalization)")
    print("=" * 60)
    
    try:
        from src.face.recognition.embeddings import OpenCVDNNEmbeddingBackend
        
        start = time.time()
        inner_backend = OpenCVDNNEmbeddingBackend()
        load_time = (time.time() - start) * 1000
        
        # Create wrapper with CLAHE preprocessing
        class OpenFaceCLAHEBackend:
            def __init__(self, backend):
                self.backend = backend
                self.embedding_dim = backend.embedding_dim
                self.embedding_size = backend.embedding_dim  # Alias for compatibility
            
            def extract(self, face_image: np.ndarray) -> np.ndarray:
                # Apply CLAHE preprocessing
                preprocessed = apply_clahe(face_image)
                return self.backend.extract(preprocessed)
        
        backend = OpenFaceCLAHEBackend(inner_backend)
        
        print(f"  ✓ Model loaded in {load_time:.0f}ms")
        print(f"  ✓ Embedding size: {backend.embedding_dim}D")
        
        return _run_model_test("OpenFace+CLAHE", backend, load_time)
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def test_mobilenetv2() -> Optional[ModelResult]:
    """Test MobileNetV2 model."""
    print("\n" + "=" * 60)
    print("Testing: MobileNetV2 (Keras) - 512D embeddings")
    print("=" * 60)
    
    try:
        from src.face import MobileNetV2EmbeddingBackend
        
        start = time.time()
        backend = MobileNetV2EmbeddingBackend()
        load_time = (time.time() - start) * 1000
        
        print(f"  ✓ Model loaded in {load_time:.0f}ms")
        print(f"  ✓ Embedding size: {backend.embedding_dim}D")
        
        return _run_model_test("MobileNetV2-512D", backend, load_time)
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_arcface() -> Optional[ModelResult]:
    """Test ArcFace model using insightface or onnx."""
    print("\n" + "=" * 60)
    print("Testing: ArcFace (InsightFace) - 512D embeddings")
    print("=" * 60)
    
    try:
        # Try insightface package first
        import insightface
        from insightface.app import FaceAnalysis
        
        start = time.time()
        app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        load_time = (time.time() - start) * 1000
        
        print(f"  ✓ Model loaded in {load_time:.0f}ms")
        print(f"  ✓ Embedding size: 512D")
        
        # Create wrapper backend
        class ArcFaceBackend:
            embedding_dim = 512
            embedding_size = 512  # Alias for compatibility
            
            def __init__(self, app):
                self.app = app
            
            def extract(self, face_image: np.ndarray) -> np.ndarray:
                faces = self.app.get(face_image)
                if not faces:
                    # If no face detected, try resizing
                    face_image = cv2.resize(face_image, (112, 112))
                    faces = self.app.get(face_image)
                
                if faces:
                    embedding = faces[0].embedding
                    # L2 normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding.astype(np.float32)
                else:
                    raise ValueError("No face detected in image")
        
        backend = ArcFaceBackend(app)
        return _run_model_test("ArcFace-512D", backend, load_time)
        
    except ImportError:
        print("  ✗ InsightFace not installed. Install with:")
        print("    pip install insightface onnxruntime")
        return None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def test_retinaface() -> Optional[ModelResult]:
    """Test RetinaFace for detection + ArcFace for recognition."""
    print("\n" + "=" * 60)
    print("Testing: RetinaFace + ArcFace (via InsightFace)")
    print("=" * 60)
    
    # RetinaFace is a detection model, not recognition
    # It's typically bundled with InsightFace which uses ArcFace for embeddings
    print("  → RetinaFace is a DETECTION model, not recognition.")
    print("  → Using InsightFace (ArcFace) for embeddings instead.")
    
    return test_arcface()  # They use the same embedding model


def test_yolov8_face() -> Optional[ModelResult]:
    """Test YOLOv8-Face for detection + embeddings."""
    print("\n" + "=" * 60)
    print("Testing: YOLOv8-Face")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Check if face model exists
        print("  → YOLOv8 is primarily a DETECTION model.")
        print("  → For face recognition, you'd need a separate embedding model.")
        print("  → Skipping (use YOLOv8 for detection + ArcFace for recognition)")
        return None
        
    except ImportError:
        print("  ✗ Ultralytics not installed. Install with:")
        print("    pip install ultralytics")
        return None


def test_facenet() -> Optional[ModelResult]:
    """Test FaceNet model (alternative to OpenFace)."""
    print("\n" + "=" * 60)
    print("Testing: FaceNet (facenet-pytorch) - 512D embeddings")
    print("=" * 60)
    
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        
        start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        load_time = (time.time() - start) * 1000
        
        print(f"  ✓ Model loaded in {load_time:.0f}ms (device: {device})")
        print(f"  ✓ Embedding size: 512D")
        
        class FaceNetBackend:
            embedding_dim = 512
            embedding_size = 512  # Alias for compatibility
            
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def extract(self, face_image: np.ndarray) -> np.ndarray:
                # Preprocess: resize, convert to RGB, normalize
                face = cv2.resize(face_image, (160, 160))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face.astype(np.float32) / 255.0
                face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
                
                # Convert to tensor
                face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.model(face).cpu().numpy().flatten()
                
                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding.astype(np.float32)
        
        backend = FaceNetBackend(model, device)
        return _run_model_test("FaceNet-512D", backend, load_time)
        
    except ImportError:
        print("  ✗ facenet-pytorch not installed. Install with:")
        print("    pip install facenet-pytorch")
        return None
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def _run_model_test(name: str, backend, load_time: float) -> Optional[ModelResult]:
    """Run the actual model comparison test."""
    
    # Load test images
    same_person_dir = project_root / "data" / "raw" / "faces" / "test_same_person"
    different_people_dir = project_root / "data" / "raw" / "faces" / "test_different_people"
    watch_list_dir = project_root / "data" / "raw" / "faces" / "watch_list"
    
    # Use watch_list as "same person" reference if test_same_person doesn't exist
    if not same_person_dir.exists() or not list(same_person_dir.glob("*")):
        same_person_dir = watch_list_dir
        print(f"  → Using watch_list folder for same-person test")
    
    same_person_images = load_images(same_person_dir)
    different_people_images = load_images(different_people_dir)
    
    if len(same_person_images) < 1:
        print(f"  ✗ No images found in {same_person_dir}")
        print(f"    Add photos of the SAME person to test recognition accuracy")
        return None
    
    print(f"  → Found {len(same_person_images)} same-person images")
    print(f"  → Found {len(different_people_images)} different-people images")
    
    # Extract embeddings
    print(f"  → Extracting embeddings...")
    
    same_person_embeddings = []
    extraction_times = []
    
    for img_name, img in same_person_images:
        try:
            start = time.time()
            emb = backend.extract(img)
            extraction_times.append((time.time() - start) * 1000)
            same_person_embeddings.append((img_name, emb))
        except Exception as e:
            print(f"    ✗ Failed on {img_name}: {e}")
    
    different_people_embeddings = []
    for img_name, img in different_people_images:
        try:
            emb = backend.extract(img)
            different_people_embeddings.append((img_name, emb))
        except Exception as e:
            print(f"    ✗ Failed on {img_name}: {e}")
    
    avg_extraction_time = np.mean(extraction_times) if extraction_times else 0
    print(f"  → Avg extraction time: {avg_extraction_time:.1f}ms")
    
    # Calculate same-person similarities (compare all pairs)
    same_person_scores = []
    if len(same_person_embeddings) >= 2:
        for i in range(len(same_person_embeddings)):
            for j in range(i + 1, len(same_person_embeddings)):
                name_i, emb_i = same_person_embeddings[i]
                name_j, emb_j = same_person_embeddings[j]
                sim = cosine_similarity(emb_i, emb_j)
                same_person_scores.append(sim)
                print(f"    Same person: {name_i} vs {name_j} = {sim:.2%}")
    elif len(same_person_embeddings) == 1:
        # Only one image - can't compare same person
        print(f"    ⚠ Only 1 same-person image. Add more for accurate testing!")
        # Use self-similarity as baseline (should be 1.0)
        _, emb = same_person_embeddings[0]
        same_person_scores.append(cosine_similarity(emb, emb))
    
    # Calculate different-person similarities (compare reference vs others)
    different_person_scores = []
    if same_person_embeddings and different_people_embeddings:
        ref_name, ref_emb = same_person_embeddings[0]
        for name, emb in different_people_embeddings:
            sim = cosine_similarity(ref_emb, emb)
            different_person_scores.append(sim)
            print(f"    Different: {ref_name} vs {name} = {sim:.2%}")
    
    # Get embedding size with fallback for compatibility
    emb_dim = getattr(backend, 'embedding_dim', getattr(backend, 'embedding_size', 128))
    
    return ModelResult(
        name=name,
        embedding_size=emb_dim,
        same_person_scores=same_person_scores,
        different_person_scores=different_person_scores,
        extraction_time_ms=avg_extraction_time,
        load_time_ms=load_time,
    )


def print_results(results: List[ModelResult]):
    """Print comparison results."""
    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    if not results:
        print("No models were successfully tested!")
        return
    
    # Sort by separation gap (higher is better)
    results.sort(key=lambda r: r.separation_gap, reverse=True)
    
    print(f"\n{'Model':<20} {'Same Avg':<10} {'Same Min':<10} {'Diff Max':<10} {'Gap':<10} {'Threshold':<10}")
    print("-" * 70)
    
    for r in results:
        gap_indicator = "✓✓✓" if r.separation_gap > 0.3 else ("✓✓" if r.separation_gap > 0.1 else ("✓" if r.separation_gap > 0 else "✗"))
        print(f"{r.name:<20} {r.same_person_avg:>8.1%}  {r.same_person_min:>8.1%}  {r.different_person_max:>8.1%}  {r.separation_gap:>+7.1%}  {r.recommended_threshold:>8.1%} {gap_indicator}")
    
    print()
    print("LEGEND:")
    print("  Same Avg:   Average similarity between photos of the SAME person (higher = better)")
    print("  Same Min:   Minimum similarity for same person (higher = better)")
    print("  Diff Max:   Maximum similarity for DIFFERENT people (lower = better)")
    print("  Gap:        Same Min - Diff Max (higher = better separation)")
    print("  Threshold:  Recommended recognition threshold")
    print()
    
    best = results[0]
    print(f"🏆 RECOMMENDED MODEL: {best.name}")
    print(f"   - Set recognition threshold to: {best.recommended_threshold:.2f}")
    print(f"   - Expected same-person match rate: >{best.same_person_min:.0%}")
    print(f"   - Expected false positive rate: <{best.different_person_max:.0%}")
    print(f"   - Extraction time: {best.extraction_time_ms:.1f}ms per face")


def main():
    print("=" * 70)
    print("FACE RECOGNITION MODEL COMPARISON")
    print("=" * 70)
    print()
    print("This tool compares different face recognition models to find the best one.")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ REQUIRED TEST DATA:                                                 │")
    print("│                                                                     │")
    print("│ 1. data/raw/faces/watch_list/          - Your face (1+ photos)     │")
    print("│ 2. data/raw/faces/test_same_person/    - MORE photos of YOU        │")
    print("│    (different expressions, angles, lighting - at least 3-5 photos) │")
    print("│ 3. data/raw/faces/test_different_people/ - Photos of OTHER people  │")
    print("│    (friends, family, strangers - at least 3-5 photos)              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    
    # Check for test data
    test_different = project_root / "data" / "raw" / "faces" / "test_different_people"
    if not test_different.exists():
        test_different.mkdir(parents=True, exist_ok=True)
        print(f"Created: {test_different}")
        print("⚠️  Please add photos of DIFFERENT people to this folder!")
        print()
    
    test_same = project_root / "data" / "raw" / "faces" / "test_same_person"
    if not test_same.exists():
        test_same.mkdir(parents=True, exist_ok=True)
        print(f"Created: {test_same}")
        print("⚠️  Please add MORE photos of yourself (different expressions) to this folder!")
        print()
    
    # Run tests
    results = []
    
    # Test OpenFace variants (always available)
    result = test_openface_opencv()
    if result:
        results.append(result)
    
    result = test_openface_clahe()
    if result:
        results.append(result)
    
    # Test optional models
    result = test_mobilenetv2()
    if result:
        results.append(result)
    
    result = test_facenet()
    if result:
        results.append(result)
    
    result = test_arcface()
    if result:
        results.append(result)
    
    # Print results
    print_results(results)
    
    print()
    print("To install additional models (requires ~2GB download):")
    print("  pip install facenet-pytorch    # FaceNet (recommended, smaller)")
    print("  pip install insightface onnxruntime  # ArcFace (most accurate)")
    print()


if __name__ == "__main__":
    main()
