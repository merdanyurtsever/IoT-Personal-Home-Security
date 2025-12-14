
#!/usr/bin/env python3
"""Face Recognition Viewfinder (MobileFaceNet version).

A real-time face recognition system using MobileFaceNet (InsightFace model).
Can be used standalone or integrated into other applications.

Usage:
	python -m src.face-alternative1.viewfinder [--watch-list PATH] [--threshold 0.35]

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
	candidates = [
		Path("watch_list"),
		Path("faces"),
		Path("face-alternative1/watch_list"),
		Path("face-alternative1/faces"),
		Path("data/watch_list"),
		Path("data/raw/faces/watch_list"),
	]
	for path in candidates:
		if path.exists() and path.is_dir():
			return path
	return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def enhance_brightness(image: np.ndarray) -> np.ndarray:
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	l = clahe.apply(l)
	lab = cv2.merge([l, a, b])
	return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def expand_bbox(x: int, y: int, w: int, h: int, frame_shape: Tuple[int, ...], margin: float = 0.25) -> Tuple[int, int, int, int]:
	frame_h, frame_w = frame_shape[:2]
	margin_w = int(w * margin)
	margin_h = int(h * margin)
	new_x = max(0, x - margin_w)
	new_y = max(0, y - margin_h)
	new_w = min(frame_w - new_x, w + 2 * margin_w)
	new_h = min(frame_h - new_y, h + 2 * margin_h)
	return new_x, new_y, new_w, new_h


# Recognizer using RetinaFace for detection and antelopev2 for recognition

# Recognizer using serengil/retinaface for detection and antelopev2 for recognition

# Recognizer using serengil/retinaface for detection and antelopev2 for recognition (direct model)
class MobileFaceNetRecognizer:
	"""Face recognition using serengil/retinaface for detection and antelopev2 for recognition (direct model)."""
	def __init__(self, rec_model: str = "antelopev2"):
		try:
			from retinaface import RetinaFace
		except ImportError:
			raise ImportError(
				"retina-face not installed. Install with:\n  pip install retina-face"
			)
		try:
			from insightface.model_zoo import get_model
		except ImportError:
			raise ImportError(
				"InsightFace not installed. Install with:\n  pip install insightface onnxruntime"
			)
		self.retinaface = RetinaFace
		self.recognizer = get_model(rec_model)
		self.recognizer.prepare(ctx_id=-1)
		self.embedding_dim = 512

	def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
		results = self.retinaface.detect_faces(frame)
		bboxes = []
		if isinstance(results, dict):
			for key in results:
				face = results[key]
				x1, y1, w, h = face['facial_area']
				bboxes.append((x1, y1, w - x1, h - y1))
		return bboxes

	def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
		try:
			# Preprocess: resize, BGR->RGB, normalize
			import cv2
			import numpy as np
			face_resized = cv2.resize(face_image, (112, 112))
			face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
			face_input = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
			face_input = (face_input - 127.5) / 127.5
			face_input = np.expand_dims(face_input, axis=0)
			embedding = self.recognizer.get_embedding(face_input)[0]
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
		self.faces.append((name, embedding))
	def clear(self) -> None:
		self.faces.clear()
	def find_match(self, embedding: np.ndarray, threshold: float = 0.35) -> Tuple[Optional[str], float]:
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

def load_watch_list(recognizer: MobileFaceNetRecognizer, watch_list_dir: Path, apply_enhancement: bool = True) -> FaceDatabase:
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
			if apply_enhancement:
				img = enhance_brightness(img)
			faces = recognizer.detect_faces(img)
			if faces:
				x, y, w, h = faces[0]
				ex, ey, ew, eh = expand_bbox(x, y, w, h, img.shape)
				face_crop = img[ey:ey+eh, ex:ex+ew]
				embedding = recognizer.extract_embedding(face_crop)
			else:
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
				   save_dir: Optional[Path] = None,
				   record: bool = False,
				   record_fps: float = 15.0) -> None:
	if watch_list_dir is None:
		watch_list_dir = find_watch_list_dir()
		if watch_list_dir is None:
			watch_list_dir = Path("watch_list")
			print(f"[WARN] No watch list found. Create '{watch_list_dir}/' with face images.")
	if save_dir is None:
		save_dir = Path("captures")
	video_writer = None
	print("=" * 60)
	print("FACE RECOGNITION VIEWFINDER (MobileFaceNet)")
	print("=" * 60)
	print("\nModel: MobileFaceNet (128D)")
	print("\nLoading model...")
	recognizer = MobileFaceNetRecognizer(rec_model="antelopev2")
	print("  ✓ Model loaded")
	database = load_watch_list(recognizer, watch_list_dir)
	print("\nOpening camera...")
	cap = cv2.VideoCapture(camera_id)
	if not cap.isOpened():
		print("ERROR: Could not open camera")
		return
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f"  Camera resolution: {width}x{height}")
	if record:
		save_dir.mkdir(parents=True, exist_ok=True)
		from datetime import datetime
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		video_path = save_dir / f"recording_{ts}.mp4"
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		video_writer = cv2.VideoWriter(str(video_path), fourcc, record_fps, (width, height))
		print(f"[INFO] Recording video to: {video_path}")
	print("\nControls:")
	print("  B      : Toggle brightness enhancement")
	print("  R      : Re-enroll faces")
	print("  S      : Save current frame")
	print("  SPACE  : Pause/Resume")
	print("  +/-    : Adjust threshold")
	print("  Q/ESC  : Quit")
	print()
	enhance_brightness_enabled = True
	frame_times = []
	paused = False
	last_frame = None
	import threading
	import copy
	frame_count = 0
	model_frame = None
	model_frame_time = 0
	model_lock = threading.Lock()
	stop_event = threading.Event()
	recog_state = {
		'frame': None,
		'result': None,
		'fps': 0.0,
		'enhance': True,
	}
	def recognition_worker():
		while not stop_event.is_set():
			with model_lock:
				frame = recog_state['frame']
				enhance = recog_state['enhance']
			if frame is None:
				time.sleep(0.01)
				continue
			proc_frame = frame.copy()
			t0 = time.time()
			if enhance:
				proc_frame = enhance_brightness(proc_frame)
			faces = recognizer.detect_faces(proc_frame)
			for (x, y, w, h) in faces:
				ex, ey, ew, eh = expand_bbox(x, y, w, h, proc_frame.shape)
				ex2, ey2 = min(proc_frame.shape[1], ex + ew), min(proc_frame.shape[0], ey + eh)
				if ex2 <= ex or ey2 <= ey:
					continue
				face_crop = proc_frame[ey:ey2, ex:ex2]
				embedding = recognizer.extract_embedding(face_crop)
				if embedding is not None and len(database) > 0:
					match_name, score = database.find_match(embedding, threshold)
					if match_name:
						color = (0, 0, 255)
						label = f"THREAT {score:.0%}"
					else:
						color = (0, 255, 0)
						label = f"Safe {score:.0%}"
				else:
					color = (128, 128, 128)
					label = "No DB" if len(database) == 0 else "No embed"
				x2, y2 = min(proc_frame.shape[1], x + w), min(proc_frame.shape[0], y + h)
				cv2.rectangle(proc_frame, (x, y), (x2, y2), color, 2)
				label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
				cv2.rectangle(proc_frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
				cv2.putText(proc_frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
			fps = 1.0 / (time.time() - t0) if faces else 0.0
			brightness_status = "ON" if enhance else "OFF"
			info_lines = [
				f"MobileFaceNet (128D) | Threshold: {threshold:.2f}",
				f"Brightness: {brightness_status} | FPS: {fps:.1f}",
				f"Enrolled: {len(database)} face(s)",
			]
			if video_writer:
				info_lines.append("RECORDING...")
			for i, line in enumerate(info_lines):
				cv2.putText(proc_frame, line, (10, 25 + i * 25),
						   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
			with model_lock:
				recog_state['result'] = proc_frame
				recog_state['fps'] = fps
			time.sleep(0.01)
	recog_thread = threading.Thread(target=recognition_worker, daemon=True)
	recog_thread.start()
	running = True
	try:
		while running:
			if not paused:
				ret, raw_frame = cap.read()
				if not ret:
					break
				last_frame = raw_frame.copy()
			else:
				if last_frame is None:
					continue
				raw_frame = last_frame.copy()
			if video_writer is not None:
				if video_writer.isOpened():
					video_writer.write(raw_frame)
			with model_lock:
				recog_state['frame'] = raw_frame.copy()
				recog_state['enhance'] = enhance_brightness_enabled
			with model_lock:
				model_frame = recog_state['result']
			if model_frame is not None:
				if model_frame.shape != raw_frame.shape:
					model_frame = cv2.resize(model_frame, (raw_frame.shape[1], raw_frame.shape[0]))
				split = np.hstack((raw_frame, model_frame))
			else:
				split = np.hstack((raw_frame, raw_frame))
			cv2.imshow("[Left] Live Video  |  [Right] Face Recognition", split)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q') or key == 27:
				running = False
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
				cv2.imwrite(str(filename), raw_frame)
				print(f"Saved: {filename}")
			elif key == ord(' '):
				paused = not paused
				print(f"{'PAUSED' if paused else 'RESUMED'}")
			frame_count += 1
	finally:
		stop_event.set()
		recog_thread.join(timeout=2.0)
		cap.release()
		if video_writer is not None:
			if video_writer.isOpened():
				video_writer.release()
		cv2.destroyAllWindows()
		time.sleep(0.2)
		print("\nDone!")

def main():
	parser = argparse.ArgumentParser(
		description="Face Recognition Viewfinder using MobileFaceNet",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python -m src.face-alternative1.viewfinder
  python -m src.face-alternative1.viewfinder --watch-list /path/to/faces
  python -m src.face-alternative1.viewfinder --threshold 0.4
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
