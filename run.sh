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
#
# Docker mode (default if Docker available):
#   ./run.sh docker start   # Run in container
#
# Local mode:
#   ./run.sh --local start  # Use local Python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
    echo "  help              Show this help"
    echo ""
    echo "Face module (standalone):"
    echo "  face detect       Face detection"
    echo "  face api          Face API server"
    echo "  face test         Test face module"
    echo ""
    echo "Docker:"
    echo "  docker build      Build container"
    echo "  docker start      Run in container"
    echo "  docker shell      Shell in container"
    echo ""
    echo "Options:"
    echo "  --local           Use local Python (not Docker)"
    echo "  --camera, -c      Use camera input"
    echo "  --image, -i FILE  Process image file"
    echo ""
    echo "Examples:"
    echo "  ./run.sh start"
    echo "  ./run.sh detect --camera"
    echo "  ./run.sh face api --port 8000"
    echo "  ./run.sh docker start"
}

# ============================================================
# DOCKER COMMANDS
# ============================================================

docker_build() {
    print_info "Building Docker image..."
    docker build -f docker/Dockerfile -t iot-home-security .
    print_success "Image built: iot-home-security"
}

docker_run() {
    local cmd="${1:-start}"
    shift 2>/dev/null || true
    
    local docker_opts="-it --rm"
    local mount_opts="-v $SCRIPT_DIR/src:/app/src:ro -v $SCRIPT_DIR/data:/app/data -v $SCRIPT_DIR/config:/app/config:ro"
    
    # Check for camera flag
    if [[ "$*" == *"--camera"* ]] || [[ "$*" == *"-c"* ]]; then
        if [[ -e /dev/video0 ]]; then
            docker_opts="$docker_opts --device /dev/video0 --privileged"
            print_info "Camera device attached"
        fi
        if [[ -n "$DISPLAY" ]]; then
            docker_opts="$docker_opts -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro"
            xhost +local:docker 2>/dev/null || true
        fi
    fi
    
    # Port for API
    if [[ "$cmd" == "start" ]] || [[ "$cmd" == "api" ]]; then
        docker_opts="$docker_opts -p 8000:8000"
    fi
    
    case "$cmd" in
        start|api)
            docker run $docker_opts $mount_opts iot-home-security \
                python -m src.cli start "$@"
            ;;
        detect)
            docker run $docker_opts $mount_opts iot-home-security \
                python -m src.cli detect "$@"
            ;;
        recognize)
            docker run $docker_opts $mount_opts iot-home-security \
                python -m src.cli recognize "$@"
            ;;
        face)
            docker run $docker_opts $mount_opts iot-home-security \
                python -m src.face "$@"
            ;;
        test)
            docker run $docker_opts $mount_opts iot-home-security \
                python -m pytest tests/ -v "$@"
            ;;
        shell)
            docker run $docker_opts $mount_opts iot-home-security bash
            ;;
        *)
            docker run $docker_opts $mount_opts iot-home-security "$cmd" "$@"
            ;;
    esac
}

# ============================================================
# LOCAL COMMANDS (Python directly)
# ============================================================

run_local() {
    local cmd="${1:-start}"
    shift 2>/dev/null || true
    
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
    local use_docker=false
    local use_local=false
    local cmd=""
    local args=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --local)
                use_local=true
                shift
                ;;
            docker)
                use_docker=true
                shift
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            build)
                docker_build
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
    
    # Run the command
    if [[ "$use_docker" == true ]]; then
        docker_run "$cmd" "${args[@]}"
    elif [[ "$use_local" == true ]]; then
        run_local "$cmd" "${args[@]}"
    else
        # Default: try local first, fall back to docker
        if command -v python &> /dev/null && python -c "import src" 2>/dev/null; then
            run_local "$cmd" "${args[@]}"
        elif command -v docker &> /dev/null; then
            print_info "Using Docker..."
            docker_run "$cmd" "${args[@]}"
        else
            print_error "Python or Docker required"
            exit 1
        fi
    fi
}

main "$@"
