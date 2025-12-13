"""Entry point for running face module as standalone.

Usage:
    python -m src.face                          # Auto-detect watch list
    python -m src.face -w ./faces               # Custom watch list
    python -m src.face -t 0.4                   # Custom threshold
    python -m src.face -s ./saved               # Custom save directory
"""

from pathlib import Path


def main():
    """Run the ArcFace viewfinder."""
    from .viewfinder import run_viewfinder
    
    import argparse
    parser = argparse.ArgumentParser(
        description="ArcFace Face Recognition Viewfinder",
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
    
    args = parser.parse_args()
    
    run_viewfinder(
        watch_list_dir=args.watchlist,
        threshold=args.threshold,
        camera_id=args.camera,
        save_dir=args.savedir,
    )


if __name__ == "__main__":
    main()
