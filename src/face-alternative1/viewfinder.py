
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




# Recognizer using OpenCV Haar Cascade for detection and LBPHFaceRecognizer for recognition
class MobileFaceNetRecognizer:
	"""Face recognition using OpenCV Haar Cascade for detection and LBPHFaceRecognizer for recognition."""
	def __init__(self):
		self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.label_map = {}  # label_id -> name
		self.name_map = {}   # name -> label_id
		self.next_label = 0
		self.trained = False

	def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
		return [tuple(face) for face in faces]

	def add_face(self, name: str, face_img: np.ndarray):
		gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
		if name not in self.name_map:
			label = self.next_label
			self.name_map[name] = label
			self.label_map[label] = name
			self.next_label += 1
		else:
			label = self.name_map[name]
		if not hasattr(self, 'train_faces'):
			self.train_faces = []
			self.train_labels = []
		self.train_faces.append(gray)
		self.train_labels.append(label)

	def train(self):
		if hasattr(self, 'train_faces') and len(self.train_faces) > 0:
			self.recognizer.train(self.train_faces, np.array(self.train_labels))
			self.trained = True

	def recognize(self, face_img: np.ndarray, threshold: float = 0.35, max_confidence: float = 200.0):
		"""Returns (is_threat, match_score) where match_score is in [0,1] (higher is better)."""
		if not self.trained:
			return False, None
		gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
		label, confidence = self.recognizer.predict(gray)
		capped = min(confidence, max_confidence)
		match_score = max(0.0, 1.0 - (capped / max_confidence))
		is_threat = match_score >= threshold
		return is_threat, match_score


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
	enrolled = 0
	if not watch_list_dir.exists():
		print(f"Watch list directory not found: {watch_list_dir}")
		return enrolled
	print(f"\nLoading watch list from: {watch_list_dir}")
	for ext in ["*.jpg", "*.jpeg", "*.png"]:
		for img_path in watch_list_dir.glob(ext):
			img = cv2.imread(str(img_path))
			if img is None:
				continue
			if apply_enhancement:
				img = enhance_brightness(img)
			faces = recognizer.detect_faces(img)
			if faces:
				x, y, w, h = faces[0]
				face_crop = img[y:y+h, x:x+w]
			else:
				face_crop = img
			name = img_path.stem
			recognizer.add_face(name, face_crop)
			enrolled += 1
			print(f"  ✓ Enrolled: {img_path.name}")
	recognizer.train()
	print(f"Enrolled {enrolled} face(s)")
	return enrolled

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
	print("FACE RECOGNITION VIEWFINDER (OpenCV Haar+LBPH)")
	print("=" * 60)
	print("\nModel: OpenCV Haar Cascade + LBPHFaceRecognizer")
	print("\nLoading recognizer...")
	recognizer = MobileFaceNetRecognizer()
	print("  ✓ Recognizer ready")
	enrolled = load_watch_list(recognizer, watch_list_dir)
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
		frame_skip = 3  # Only process every 3rd frame for speed
		counter = 0
		while not stop_event.is_set():
			with model_lock:
				frame = recog_state['frame']
				enhance = recog_state['enhance']
			if frame is None:
				time.sleep(0.01)
				continue
			counter += 1
			if counter % frame_skip != 0:
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
				is_threat, match_score = recognizer.recognize(face_crop, threshold)
				if enrolled == 0:
					color = (128, 128, 128)
					label = "No DB"
				elif match_score is None:
					color = (128, 128, 128)
					label = "No embed"
				elif is_threat:
					color = (0, 0, 255)
					label = f"THREAT {match_score:.0%}"
				else:
					color = (0, 255, 0)
					label = f"Safe {match_score:.0%}"
				x2, y2 = min(proc_frame.shape[1], x + w), min(proc_frame.shape[0], y + h)
				cv2.rectangle(proc_frame, (x, y), (x2, y2), color, 2)
				label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
				cv2.rectangle(proc_frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
				cv2.putText(proc_frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
			fps = 1.0 / (time.time() - t0) if faces else 0.0
			brightness_status = "ON" if enhance else "OFF"
			info_lines = [
				f"LBPH | Threshold: {threshold:.2f}",
				f"Brightness: {brightness_status} | FPS: {fps:.1f}",
				f"Enrolled: {enrolled} face(s)",
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
				enrolled = load_watch_list(recognizer, watch_list_dir)
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
