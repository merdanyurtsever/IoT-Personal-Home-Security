"""CLI entry point for IoT Home Security system."""

import argparse
import sys


def main():
    """Main entry point for the IoT Home Security CLI."""
    parser = argparse.ArgumentParser(
        description="IoT Personal Home Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the security system")
    start_parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["face", "audio", "all"],
        default="all",
        help="Which model to train",
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument(
        "--component",
        type=str,
        choices=["camera", "microphone", "sensors", "all"],
        default="all",
        help="Which component to test",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "start":
        print(f"Starting security system with config: {args.config}")
        # TODO: Implement start logic
    elif args.command == "train":
        print(f"Training models: {args.model}")
        # TODO: Implement training logic
    elif args.command == "test":
        print(f"Testing components: {args.component}")
        # TODO: Implement test logic


if __name__ == "__main__":
    main()
