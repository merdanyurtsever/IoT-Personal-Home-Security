#!/usr/bin/env python3
"""Main entry point for IoT Home Security System.

Simplified entry point that delegates to the CLI module.

Usage:
    python main.py start          # Start the security system with API
    python main.py detect         # Face detection
    python main.py detect --camera # Live camera detection
    python main.py test           # Test components

Or use the CLI directly:
    python -m src.cli start
    python -m src.face detect --camera
"""

import sys


def main():
    """Main entry point - delegates to CLI."""
    # If no arguments, show help
    if len(sys.argv) == 1:
        print(__doc__)
        print("Run 'python main.py --help' for more options")
        sys.exit(0)
    
    # Import and run CLI
    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
