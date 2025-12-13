#!/bin/bash
#
# IoT Home Security - Run Script
# ================================
# Simple script to run the security system
#
# Usage:
#   ./run.sh start          # Start API server
#   ./run.sh detect         # Test face detection
#   ./run.sh detect --camera # Live camera detection
#   ./run.sh test           # Run tests
#   ./run.sh help           # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# ============================================================
# HELP
# ============================================================

show_help() {
    echo "IoT Home Security - Run Script"
    echo "================================"
    echo ""
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start             Start API server (default)"
    echo "  detect            Test face detection"
    echo "  detect --camera   Live camera detection"
    echo "  detect -i FILE    Detect faces in image"
    echo "  recognize         Test face recognition"
    echo "  test              Run tests"
    echo "  setup             Setup Python environment"
    echo "  help              Show this help"
    echo ""
    echo "Face module (standalone):"
    echo "  face detect       Face detection"
    echo "  face api          Face API server"
    echo "  face test         Test face module"
    echo ""
    echo "Options:"
    echo "  --camera, -c      Use camera input"
    echo "  --image, -i FILE  Process image file"
    echo ""
    echo "Examples:"
    echo "  ./run.sh start"
    echo "  ./run.sh detect --camera"
    echo "  ./run.sh face api --port 8000"
}

# ============================================================
# SETUP
# ============================================================

setup_env() {
    print_info "Setting up Python environment..."
    
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
        print_success "Created virtual environment"
    fi
    
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# ============================================================
# COMMANDS
# ============================================================

run_cmd() {
    local cmd="${1:-start}"
    shift 2>/dev/null || true
    
    # Activate venv if available
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
    
    case "$cmd" in
        start|api)
            python -m src.cli start "$@"
            ;;
        detect)
            python -m src.cli detect "$@"
            ;;
        recognize)
            python -m src.cli recognize "$@"
            ;;
        face)
            python -m src.face "$@"
            ;;
        test)
            python -m pytest tests/ -v "$@"
            ;;
        *)
            python -m src.cli "$cmd" "$@"
            ;;
    esac
}

# ============================================================
# MAIN
# ============================================================

main() {
    local cmd=""
    local args=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            help|--help|-h)
                show_help
                exit 0
                ;;
            setup)
                setup_env
                exit 0
                ;;
            *)
                if [[ -z "$cmd" ]]; then
                    cmd="$1"
                else
                    args+=("$1")
                fi
                shift
                ;;
        esac
    done
    
    # Default command
    [[ -z "$cmd" ]] && cmd="start"
    
    run_cmd "$cmd" "${args[@]}"
}

main "$@"
