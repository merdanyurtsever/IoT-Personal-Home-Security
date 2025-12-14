
"""Entry point for running face-alternative1 as standalone (MobileFaceNet)."""
from pathlib import Path

def main():
	from .viewfinder import run_viewfinder
	import argparse
	parser = argparse.ArgumentParser(
		description="MobileFaceNet Face Recognition Viewfinder",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Controls:
	q/ESC   - Quit
	r       - Reload watch list
	s       - Save current frame
	b       - Toggle brightness enhancement
	SPACE   - Pause/Resume
	+/-     - Adjust threshold
"""
	)
	parser.add_argument("-w", "--watchlist", type=Path, default=None,
						help="Watch list folder (auto-detected if not provided)")
	parser.add_argument("-t", "--threshold", type=float, default=0.35,
						help="Recognition threshold (default: 0.35)")
	parser.add_argument("-c", "--camera", type=int, default=0,
						help="Camera device index (default: 0)")
	parser.add_argument("-s", "--savedir", type=Path, default=None,
						help="Directory for saved frames (default: ./captures)")
	parser.add_argument("-r", "--record", action="store_true",
						help="Enable video recording to ./captures/recording_<timestamp>.mp4")
	parser.add_argument("--record-fps", type=float, default=15.0,
						help="Recording FPS (default: 15.0)")
	args = parser.parse_args()
	run_viewfinder(
		watch_list_dir=args.watchlist,
		threshold=args.threshold,
		camera_id=args.camera,
		save_dir=args.savedir,
		record=args.record,
		record_fps=args.record_fps,
	)

if __name__ == "__main__":
	main()
