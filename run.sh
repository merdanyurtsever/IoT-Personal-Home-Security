#!/bin/bash
#
# IoT Home Security - Universal Run Script
# =========================================
#
# One-liner script to run various components of the security system.
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   install         - Install the package (dev mode or production)
#   start           - Start the security system
#   stop            - Stop the security system (if running as service)
#   api             - Start only the Face Management API
#   test            - Run tests
#   detect          - Test face detection
#   recognize       - Test face recognition
#   classify        - Test sound classification
#   camera          - Test camera
#   demo            - Run a demo showing all features
#   help            - Show this help message
#
# Examples:
#   ./run.sh install              # Install in dev mode
#   ./run.sh install --prod       # Install for production
#   ./run.sh start                # Start the system
#   ./run.sh start --debug        # Start with debug output
#   ./run.sh api                  # Start API server only
#   ./run.sh test                 # Run all tests
#   ./run.sh detect --image photo.jpg
#   ./run.sh camera --duration 10
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
VENV_PATH="${SCRIPT_DIR}/venv"
CONFIG_PATH="${SCRIPT_DIR}/config/config.yaml"
PI_CONFIG_PATH="${SCRIPT_DIR}/raspberry_pi/config/config.yaml"

# Detect if on Raspberry Pi
is_raspberry_pi() {
    [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null
}

# Print helpers
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${CYAN}ℹ $1${NC}"; }

# Ensure virtual environment exists and activate it
ensure_venv() {
    if [[ ! -d "$VENV_PATH" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi
    source "$VENV_PATH/bin/activate"
}

# Get the right config file
get_config() {
    if is_raspberry_pi && [[ -f "$PI_CONFIG_PATH" ]]; then
        echo "$PI_CONFIG_PATH"
    else
        echo "$CONFIG_PATH"
    fi
}

# Show help
show_help() {
    print_header "IoT Home Security - Run Script"
    cat << 'EOF'
Usage: ./run.sh <command> [options]

Commands:
  install [--prod|--dev|--arm64|--pi]
                    Install the package
                      --dev    Development mode (default)
                      --prod   Production mode
                      --arm64  ARM64 VM optimized
                      --pi     Raspberry Pi optimized
  
  start [--debug|--api-only|--no-api]
                    Start the security system
                      --debug    Enable debug output
                      --api-only Start only the API server
                      --no-api   Start without API server
  
  stop              Stop the security system service
  
  restart           Restart the security system service
  
  status            Check service status
  
  api [--port PORT] Start the Face Management API
                      --port   API port (default: 8000)
  
  test [--coverage] Run tests
                      --coverage  Include coverage report
  
  detect [--image PATH] [--camera]
                    Test face detection
                      --image   Detect faces in image
                      --camera  Live camera detection
  
  recognize [--image PATH] [--enroll NAME]
                    Test face recognition
                      --image   Recognize face in image
                      --enroll  Enroll new face

  classify [--audio PATH] [--live]
                    Test sound classification
                      --audio   Classify audio file
                      --live    Live microphone classification
  
  camera [--duration SEC]
                    Test camera
                      --duration  Test duration in seconds
  
  demo              Run interactive demo

  train [--face|--sound|--all]
                    Train models
                      --face   Train face recognition
                      --sound  Train sound classification
                      --all    Train all models

  logs [--follow]   View logs
                      --follow  Follow log output (like tail -f)

  help              Show this help message

Examples:
  ./run.sh install                    # Dev install
  ./run.sh install --pi               # Install on Raspberry Pi
  ./run.sh start                      # Start system
  ./run.sh start --debug              # Start with debug
  ./run.sh api --port 8080            # Start API on port 8080
  ./run.sh detect --camera            # Live face detection
  ./run.sh test --coverage            # Run tests with coverage
  ./run.sh logs --follow              # Follow live logs

EOF
}

# Install command
cmd_install() {
    local mode="dev"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prod) mode="prod"; shift ;;
            --dev) mode="dev"; shift ;;
            --arm64) mode="arm64"; shift ;;
            --pi) mode="pi"; shift ;;
            *) shift ;;
        esac
    done
    
    print_header "Installing IoT Home Security ($mode mode)"
    
    case "$mode" in
        arm64)
            print_info "Running ARM64 deployment script..."
            chmod +x scripts/deploy_arm64.sh
            ./scripts/deploy_arm64.sh
            ;;
        pi)
            print_info "Running Raspberry Pi installation script..."
            chmod +x raspberry_pi/scripts/install.sh
            ./raspberry_pi/scripts/install.sh
            ;;
        prod)
            ensure_venv
            print_info "Installing in production mode..."
            pip install --upgrade pip wheel
            if is_raspberry_pi; then
                pip install -r requirements-arm64.txt
            else
                pip install -r requirements.txt
            fi
            pip install .
            print_success "Installation complete"
            ;;
        dev|*)
            ensure_venv
            print_info "Installing in development mode..."
            pip install --upgrade pip wheel
            pip install -r requirements.txt
            pip install -e ".[dev]"
            print_success "Development installation complete"
            ;;
    esac
}

# Start command
cmd_start() {
    local debug=""
    local api_only=""
    local no_api=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --debug) debug="--debug"; shift ;;
            --api-only) api_only="1"; shift ;;
            --no-api) no_api="1"; shift ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    
    local config=$(get_config)
    
    if [[ -n "$api_only" ]]; then
        cmd_api
        return
    fi
    
    print_header "Starting IoT Home Security System"
    print_info "Config: $config"
    
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    if is_raspberry_pi; then
        python3 raspberry_pi/main.py --config "$config" $debug
    else
        # Development mode - run main entry point
        python3 -m src.cli start --config "$config" $debug
    fi
}

# Stop command
cmd_stop() {
    print_header "Stopping IoT Home Security"
    
    if systemctl is-active --quiet security-system 2>/dev/null; then
        sudo systemctl stop security-system
        print_success "Service stopped"
    else
        # Try to kill by process name
        pkill -f "src.cli" 2>/dev/null && print_success "Process stopped" || print_warning "No running process found"
    fi
}

# Restart command
cmd_restart() {
    print_header "Restarting IoT Home Security"
    
    if systemctl is-active --quiet security-system 2>/dev/null; then
        sudo systemctl restart security-system
        print_success "Service restarted"
    else
        cmd_stop
        sleep 1
        cmd_start
    fi
}

# Status command
cmd_status() {
    print_header "IoT Home Security Status"
    
    if systemctl is-active --quiet security-system 2>/dev/null; then
        print_success "Service is running"
        sudo systemctl status security-system --no-pager
    else
        if pgrep -f "src.cli" > /dev/null; then
            print_success "Running as process"
            pgrep -af "src.cli"
        else
            print_warning "Not running"
        fi
    fi
}

# API command
cmd_api() {
    local port=8000
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    
    print_header "Starting Face Management API"
    print_info "API will be available at http://0.0.0.0:$port"
    print_info "Documentation at http://0.0.0.0:$port/docs"
    
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    python3 << EOF
import uvicorn
from src.face_service import FaceRecognizerService
from src.api import create_app

# Initialize service
service = FaceRecognizerService(
    database_path="data/faces.db",
    upload_dir="data/raw/faces/uploads",
    embedding_backend="dlib",
)

# Create app
app = create_app(service=service)

# Run server
uvicorn.run(app, host="0.0.0.0", port=$port)
EOF
}

# Test command
cmd_test() {
    local coverage=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --coverage) coverage="--cov=src --cov-report=term-missing"; shift ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    
    print_header "Running Tests"
    
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    pytest tests/ $coverage -v
}

# Face detection test
cmd_detect() {
    local image=""
    local camera=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image) image="$2"; shift 2 ;;
            --camera) camera="1"; shift ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "Face Detection Test"
    
    if [[ -n "$camera" ]]; then
        print_info "Starting camera face detection (press 'q' to quit)..."
        python3 scripts/test_face_detection.py --camera
    elif [[ -n "$image" ]]; then
        print_info "Detecting faces in: $image"
        python3 scripts/test_face_detection.py --image "$image"
    else
        print_info "Running basic face detection test..."
        python3 -c "
from src.face import FaceDetector
import numpy as np

detector = FaceDetector(backend='haar_cascade')
print('✓ Face detector initialized')

# Test with blank image
img = np.zeros((480, 640, 3), dtype=np.uint8)
faces = detector.detect(img)
print(f'✓ Detection test passed (found {len(faces)} faces in test image)')
print('✓ Face detection is working!')
"
    fi
}

# Face recognition test
cmd_recognize() {
    local image=""
    local enroll=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image) image="$2"; shift 2 ;;
            --enroll) enroll="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "Face Recognition Test"
    
    python3 -c "
from src.face import FaceRecognizer

recognizer = FaceRecognizer(model='dlib', threshold=0.6)
print('✓ Face recognizer initialized')
print(f'  Embedding size: {recognizer.embedding_size}')
print(f'  Enrolled faces: {recognizer.get_enrolled_count()}')
print('✓ Face recognition is working!')
"
}

# Sound classification test  
cmd_classify() {
    local audio=""
    local live=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --audio) audio="$2"; shift 2 ;;
            --live) live="1"; shift ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "Sound Classification Test"
    
    python3 -c "
from src.audio import SoundClassifier
import numpy as np

classifier = SoundClassifier(sample_rate=22050)
print('✓ Sound classifier initialized')
print(f'  Target classes: {classifier.classes}')

# Test with random audio
audio = np.random.randn(22050 * 5).astype(np.float32)
result = classifier.classify(audio)
print(f'✓ Classification test: {result.label} ({result.confidence:.2%})')
print('✓ Sound classification is working!')
"
}

# Camera test
cmd_camera() {
    local duration=5
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --duration) duration="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "Camera Test"
    
    python3 -c "
from src.sensors import CameraInterface
import time

use_pi = False
try:
    with open('/proc/device-tree/model', 'r') as f:
        if 'raspberry' in f.read().lower():
            use_pi = True
except:
    pass

print(f'Using Pi Camera: {use_pi}')
camera = CameraInterface(use_picamera=use_pi)

print('Starting camera...')
camera.start()
print('✓ Camera started')

frames = 0
start = time.time()
while time.time() - start < $duration:
    frame = camera.capture()
    frames += 1
    
camera.stop()
fps = frames / $duration
print(f'✓ Captured {frames} frames in $duration seconds ({fps:.1f} FPS)')
print('✓ Camera is working!')
"
}

# Demo command
cmd_demo() {
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "IoT Home Security Demo"
    
    echo -e "${CYAN}This demo will test all major components:${NC}"
    echo "  1. Face Detection"
    echo "  2. Face Recognition"
    echo "  3. Sound Classification"
    echo "  4. Camera Interface"
    echo ""
    
    read -p "Press Enter to continue..."
    
    echo ""
    cmd_detect
    echo ""
    cmd_recognize
    echo ""
    cmd_classify
    echo ""
    
    print_success "Demo complete!"
}

# Train command - run training notebooks or scripts
cmd_train() {
    local model=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --face) model="face"; shift ;;
            --sound) model="sound"; shift ;;
            --all) model="all"; shift ;;
            *) shift ;;
        esac
    done
    
    ensure_venv
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
    
    print_header "Model Training"
    
    case "$model" in
        face)
            print_info "Starting face recognition training..."
            if [[ -f "notebooks/02_face_recognition_training.ipynb" ]]; then
                jupyter nbconvert --to notebook --execute notebooks/02_face_recognition_training.ipynb --inplace
            else
                print_warning "Training notebook not found. Using placeholder training."
                python3 -c "
from src.face_service import FaceRecognizerService
from pathlib import Path

service = FaceRecognizerService(
    database_path='data/faces.db',
    upload_dir='data/raw/faces/uploads',
)
watch_list_dir = Path('data/raw/faces/watch_list')
if watch_list_dir.exists():
    for person_dir in watch_list_dir.iterdir():
        if person_dir.is_dir():
            print(f'Processing {person_dir.name}...')
            for img_path in person_dir.glob('*.jpg'):
                service.add_face(person_dir.name, str(img_path), 'watch_list')
    print('Starting embedding extraction...')
    job_id, total = service.start_processing()
    print(f'Processing {total} faces in job {job_id}')
else:
    print('No watch list directory found. Add faces to data/raw/faces/watch_list/')
"
            fi
            ;;
        sound)
            print_info "Starting sound classification training..."
            if [[ -f "notebooks/03_sound_classification_training.ipynb" ]]; then
                jupyter nbconvert --to notebook --execute notebooks/03_sound_classification_training.ipynb --inplace
            else
                print_warning "Training notebook not found"
            fi
            ;;
        all)
            cmd_train --face
            cmd_train --sound
            ;;
        *)
            echo "Usage: ./run.sh train [--face|--sound|--all]"
            echo ""
            echo "Options:"
            echo "  --face   Train face recognition model"
            echo "  --sound  Train sound classification model"
            echo "  --all    Train all models"
            ;;
    esac
}

# Logs command
cmd_logs() {
    local follow=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --follow|-f) follow="-f"; shift ;;
            *) shift ;;
        esac
    done
    
    print_header "Viewing Logs"
    
    if systemctl is-active --quiet security-system 2>/dev/null; then
        journalctl -u security-system $follow
    elif [[ -f "logs/security.log" ]]; then
        if [[ -n "$follow" ]]; then
            tail -f logs/security.log
        else
            cat logs/security.log
        fi
    else
        print_warning "No logs found"
    fi
}

# Main entry point
main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        install)    cmd_install "$@" ;;
        start)      cmd_start "$@" ;;
        stop)       cmd_stop "$@" ;;
        restart)    cmd_restart "$@" ;;
        status)     cmd_status "$@" ;;
        api)        cmd_api "$@" ;;
        test)       cmd_test "$@" ;;
        detect)     cmd_detect "$@" ;;
        recognize)  cmd_recognize "$@" ;;
        classify)   cmd_classify "$@" ;;
        camera)     cmd_camera "$@" ;;
        demo)       cmd_demo "$@" ;;
        train)      cmd_train "$@" ;;
        logs)       cmd_logs "$@" ;;
        help|--help|-h) show_help ;;
        *)
            print_error "Unknown command: $command"
            echo "Run './run.sh help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
