"""Entry point for running face module as standalone.

Usage:
    # Run the live face recognition viewfinder (default)
    python -m src.face
    
    # With optional parameters
    python -m src.face --watchlist path/to/faces --threshold 0.35 --camera 0
"""

import sys


def main():
    """Run the ArcFace viewfinder."""
    from .viewfinder import run_viewfinder
    
    # Parse simple command line args
    import argparse
    parser = argparse.ArgumentParser(
        description="ArcFace Face Recognition Viewfinder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
    q/ESC   - Quit
    r       - Reload watch list
    c       - Clear all matches
    s       - Save current frame
    SPACE   - Pause/Resume
"""
    )
    parser.add_argument("--watchlist", "-w", type=str, default=None,
                        help="Path to watch list folder (default: data/raw/faces/watch_list)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35,
                        help="Recognition threshold (default: 0.35, lower = stricter)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera device index (default: 0)")
    
    args = parser.parse_args()
    
    # Build kwargs, only include watch_list_dir if explicitly provided
    kwargs = {
        "threshold": args.threshold,
        "camera_id": args.camera,
    }
    if args.watchlist:
        from pathlib import Path
        kwargs["watch_list_dir"] = Path(args.watchlist)
    
    run_viewfinder(**kwargs)


if __name__ == "__main__":
    main()
