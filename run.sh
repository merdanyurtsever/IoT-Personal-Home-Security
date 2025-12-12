#!/bin/bash
#
# IoT Home Security - Universal Run Script (Container-First)
# ===========================================================
#
# All Python commands run inside Docker containers by default.
# Use --local flag to run with local venv (for development without Docker).
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   build           - Build Docker image
#   start           - Start the security system (container)
#   stop            - Stop containers
#   api             - Start the Face Management API
#   test            - Run tests
#   recognize       - Test face recognition
#   classify        - Test sound classification
#   camera          - Test camera with face detection
#   demo            - Run demo showing all features
#   shell           - Open shell in container
#   logs            - View container logs
#   help            - Show this help message
#
# Options:
#   --local         Run with local venv instead of Docker
#   --rebuild       Force rebuild Docker image
#
# Examples:
#   ./run.sh build                  # Build container
#   ./run.sh start                  # Start system in container
#   ./run.sh api                    # Start API server
#   ./run.sh camera                 # Live camera with face detection
#   ./run.sh shell                  # Debug inside container
#   ./run.sh start --local          # Use local venv (no Docker)
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
DOCKER_IMAGE="iot-home-security:latest"
DOCKER_COMPOSE="docker compose"
VENV_PATH="${SCRIPT_DIR}/.venv"
CONFIG_PATH="${SCRIPT_DIR}/config/config.yaml"
PI_CONFIG_PATH="${SCRIPT_DIR}/raspberry_pi/config/config.yaml"

# Mode flags
USE_LOCAL=""
FORCE_REBUILD=""

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

# ============================================================
# DOCKER FUNCTIONS (Primary Mode)
# ============================================================

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Install Docker or use --local flag."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        print_error "Docker daemon not running. Start Docker or use --local flag."
        exit 1
    fi
}

# Build Docker image if needed
ensure_image() {
    if [[ -n "$FORCE_REBUILD" ]] || ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        print_info "Building Docker image..."
        docker build -t "$DOCKER_IMAGE" .
        print_success "Image built: $DOCKER_IMAGE"
    fi
}

# Run command in container
docker_run() {
    local interactive=""
    local devices=""
    local display_args=""
    local port_args=""
    local user_args=""
    
    # Check for camera access
    if [[ "$*" == *"camera"* ]] || [[ "$*" == *"--camera"* ]] || [[ "$*" == *"video"* ]]; then
        if [[ -e /dev/video0 ]]; then
            # Add video devices and run as root for device access
            devices="--device /dev/video0 --device /dev/video1 --privileged --security-opt label=disable"
            user_args="--user root"
            print_info "Camera devices attached (/dev/video0, /dev/video1)"
        else
            print_warning "No camera device found at /dev/video0"
        fi
    fi
    
    # Enable X11 for display (for camera preview windows)
    if [[ -n "$DISPLAY" ]]; then
        xhost +local:docker 2>/dev/null || true
        display_args="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro"
    fi
    
    # Interactive mode for TTY
    if [[ -t 0 ]] && [[ -t 1 ]]; then
        interactive="-it"
    fi
    
    docker run --rm $interactive \
        $user_args \
        -v "$SCRIPT_DIR/src:/app/src:ro" \
        -v "$SCRIPT_DIR/config:/app/config:ro" \
        -v "$SCRIPT_DIR/data:/app/data" \
        -v "$SCRIPT_DIR/logs:/app/logs" \
        -v "$SCRIPT_DIR/scripts:/app/scripts:ro" \
        -v "$SCRIPT_DIR/tests:/app/tests:ro" \
        $devices \
        $display_args \
        "$DOCKER_IMAGE" \
        "$@"
}

# Run command in container with port binding (for API)
docker_run_with_port() {
    local port="${1:-8000}"
    shift
    
    local interactive=""
    
    # Interactive mode for TTY
    if [[ -t 0 ]] && [[ -t 1 ]]; then
        interactive="-it"
    fi
    
    docker run --rm $interactive \
        -v "$SCRIPT_DIR/src:/app/src:ro" \
        -v "$SCRIPT_DIR/config:/app/config:ro" \
        -v "$SCRIPT_DIR/data:/app/data" \
        -v "$SCRIPT_DIR/logs:/app/logs" \
        -p "$port:8000" \
        "$DOCKER_IMAGE" \
        "$@"
}

# ============================================================
# LOCAL VENV FUNCTIONS (Fallback Mode)
# ============================================================

# Ensure virtual environment exists and activate it
ensure_venv() {
    if [[ ! -d "$VENV_PATH" ]]; then
        print_info "Creating virtual environment at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip wheel
        if is_raspberry_pi; then
            pip install -r requirements-pi.txt
        else
            pip install -r requirements.txt
        fi
        pip install -e .
    else
        source "$VENV_PATH/bin/activate"
    fi
    export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
}

# Get the right config file
get_config() {
    if is_raspberry_pi && [[ -f "$PI_CONFIG_PATH" ]]; then
        echo "$PI_CONFIG_PATH"
    else
        echo "$CONFIG_PATH"
    fi
}

# ============================================================
# COMMAND IMPLEMENTATIONS
# ============================================================

# Show help
show_help() {
    print_header "IoT Home Security - Container-First Run Script"
    cat << 'EOF'
Usage: ./run.sh <command> [options]

All commands run inside Docker containers by default.
Use --local flag to run with local Python virtual environment.

COMMANDS:
  build             Build the Docker image
  start             Start the security system
  stop              Stop all containers
  restart           Restart the system
  status            Check service status

  api [--port PORT] Start the Face Management API
  
  recognize [--camera] [--image PATH] [--detection-backend NAME] [--embedding-backend NAME]
                    Test face recognition with model selection
                    Detection backends: haar_cascade, mediapipe, opencv_dnn, dlib, dlib_cnn
                    Embedding backends: opencv_dnn, dlib, tflite, mobilenetv2
  
  compare           Compare face recognition models
                    Outputs performance metrics for all available models
  
  classify          Test sound classification
  camera            Test camera interface
  demo              Run interactive demo

  test [--coverage] Run tests
  train [--face|--sound|--all]
                    Train models

  shell             Open shell in container
  logs [--follow]   View logs

  install [--dev|--prod|--pi]
                    Install locally (for --local mode)

  help              Show this help message

OPTIONS:
  --local           Run with local venv instead of Docker
  --rebuild         Force rebuild Docker image before running

MODEL OPTIONS (for recognize command):
  --detection-backend, -d    Face detection model
                             haar_cascade  - Fast, less accurate
                             mediapipe     - Good balance (requires GPU)
                             opencv_dnn    - Default, works everywhere
                             dlib          - HOG-based detector
                             dlib_cnn      - CNN detector (requires GPU)
  
  --embedding-backend, -e    Face embedding/recognition model
                             opencv_dnn    - Default, OpenFace 128D
                             dlib          - dlib ResNet 128D
                             tflite        - TFLite optimized 512D
                             mobilenetv2   - MobileNetV2 512D
  
  --threshold, -t            Recognition similarity threshold (0.0-1.0, default: 0.6)

EXAMPLES:
  ./run.sh build                  # Build Docker image
  ./run.sh start                  # Start in container (default)
  ./run.sh api                    # API server in container
  ./run.sh camera                 # Live camera with face detection
  ./run.sh recognize --camera     # Camera recognition with default models
  ./run.sh recognize --camera --detection-backend opencv_dnn --embedding-backend dlib
  ./run.sh compare                # Compare all face recognition models
  ./run.sh shell                  # Debug shell in container
  ./run.sh test                   # Run tests in container

  ./run.sh start --local          # Start with local venv
  ./run.sh api --local --port 8080   # Local API on port 8080

EOF
}

# Build command
cmd_build() {
    check_docker
    print_header "Building Docker Image"
    
    local dockerfile="Dockerfile"
    if is_raspberry_pi; then
        dockerfile="Dockerfile.pi"
        print_info "Detected Raspberry Pi - using $dockerfile"
    fi
    
    docker build -f "$dockerfile" -t "$DOCKER_IMAGE" .
    print_success "Image built: $DOCKER_IMAGE"
}

# Start command
cmd_start() {
    local debug=""
    local api_only=""
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --debug) debug="--debug"; shift ;;
            --api-only) api_only="1"; shift ;;
            *) shift ;;
        esac
    done
    
    if [[ -n "$USE_LOCAL" ]]; then
        # Local mode
        ensure_venv
        local config=$(get_config)
        print_header "Starting IoT Home Security System (Local)"
        print_info "Config: $config"
        
        if is_raspberry_pi; then
            python3 raspberry_pi/main.py --config "$config" $debug
        else
            python3 -m src.cli start --config "$config" $debug
        fi
    else
        # Docker mode (default)
        check_docker
        ensure_image
        print_header "Starting IoT Home Security System (Container)"
        print_info "Starting with docker compose..."
        $DOCKER_COMPOSE up -d
        print_success "System started"
        print_info "API: http://localhost:8000"
        print_info "Docs: http://localhost:8000/docs"
        print_info "Logs: ./run.sh logs --follow"
    fi
}

# Stop command
cmd_stop() {
    print_header "Stopping IoT Home Security"
    
    if [[ -n "$USE_LOCAL" ]]; then
        # Stop local process
        if systemctl is-active --quiet security-system 2>/dev/null; then
            sudo systemctl stop security-system
            print_success "Service stopped"
        else
            pkill -f "src.cli" 2>/dev/null && print_success "Process stopped" || print_warning "No running process found"
        fi
    else
        # Stop containers
        check_docker
        $DOCKER_COMPOSE down
        print_success "Containers stopped"
    fi
}

# Restart command
cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start "$@"
}

# Status command
cmd_status() {
    print_header "IoT Home Security Status"
    
    if [[ -n "$USE_LOCAL" ]]; then
        if systemctl is-active --quiet security-system 2>/dev/null; then
            print_success "Service is running"
            sudo systemctl status security-system --no-pager
        elif pgrep -f "src.cli" > /dev/null; then
            print_success "Running as process"
            pgrep -af "src.cli"
        else
            print_warning "Not running"
        fi
    else
        check_docker
        echo "Container Status:"
        docker ps --filter "ancestor=$DOCKER_IMAGE" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        $DOCKER_COMPOSE ps 2>/dev/null || true
    fi
}

# API command
cmd_api() {
    local port=8000
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -n "$USE_LOCAL" ]]; then
        # Local mode
        ensure_venv
        print_header "Starting Face Management API (Local)"
        print_info "API: http://0.0.0.0:$port"
        print_info "Docs: http://0.0.0.0:$port/docs"
        
        python3 << EOF
import uvicorn
from src.api import create_app
from src import FaceRecognizer

# Initialize recognizer
recognizer = FaceRecognizer(
    embedding_backend="opencv_dnn",
    detection_backend="opencv_dnn",
    similarity_threshold=0.6,
)

# Create app with recognizer
app = create_app()

# Run server
uvicorn.run(app, host="0.0.0.0", port=$port)
EOF
    else
        # Docker mode (default)
        check_docker
        ensure_image
        print_header "Starting Face Management API (Container)"
        print_info "API: http://localhost:$port"
        print_info "Docs: http://localhost:$port/docs"
        
        docker_run_with_port "$port" python -m src.cli start
    fi
}

# Test command
cmd_test() {
    local coverage=""
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --coverage) coverage="--cov=src --cov-report=term-missing"; shift ;;
            *) shift ;;
        esac
    done
    
    if [[ -n "$USE_LOCAL" ]]; then
        ensure_venv
        print_header "Running Tests (Local)"
        pytest tests/ $coverage -v
    else
        check_docker
        ensure_image
        print_header "Running Tests (Container)"
        docker_run pytest tests/ $coverage -v
    fi
}

# Face detection test (camera mode is default)
cmd_detect() {
    local image=""
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --camera) shift ;;  # Accepted but ignored (camera is default)
            --image) image="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    # Image mode or camera mode (default)
    if [[ -n "$image" ]]; then
        cmd_camera --image "$image"
    else
        cmd_camera
    fi
}

# Face recognition test with model selection
cmd_recognize() {
    local image=""
    local camera=""
    local detection_backend="opencv_dnn"
    local embedding_backend="opencv_dnn"
    local threshold="0.6"
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --image|-i) image="$2"; shift 2 ;;
            --camera) camera="1"; shift ;;
            --detection-backend|-d) detection_backend="$2"; shift 2 ;;
            --embedding-backend|-e) embedding_backend="$2"; shift 2 ;;
            --threshold|-t) threshold="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -n "$USE_LOCAL" ]]; then
        ensure_venv
        print_header "Face Recognition Test (Local)"
        print_info "Detection: $detection_backend, Embedding: $embedding_backend, Threshold: $threshold"
        
        if [[ -n "$camera" ]]; then
            print_info "Starting camera face recognition (press 'q' to quit)..."
            python3 -c "
import sys
sys.path.insert(0, '.')
from src.visual import FaceRecognizer
from src.sensors.camera import Camera
from pathlib import Path
import cv2

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print(f'Detection: $detection_backend, Embedding: $embedding_backend ({recognizer.embedding_dim}D)')

# Load watch list
watch_list = Path('data/raw/faces/watch_list')
if watch_list.exists():
    results = recognizer.register_from_directory(str(watch_list))
    print(f'Loaded watch list: {results}')

with Camera() as camera:
    print('Camera started. Press q to quit.')
    for frame in camera.stream():
        result = recognizer.recognize_face(frame.image)
        output = frame.image.copy()
        if result.bbox:
            x, y, w, h = result.bbox
            color = (0, 0, 255) if result.should_alert else (0, 255, 0)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label = f'{result.identity}: {result.confidence:.0%}'
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Face Recognition', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
"
        elif [[ -n "$image" ]]; then
            print_info "Recognizing faces in: $image"
            python3 -c "
import sys
sys.path.insert(0, '.')
from src.visual import FaceRecognizer
import cv2

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print(f'Detection: $detection_backend, Embedding: $embedding_backend ({recognizer.embedding_dim}D)')

image = cv2.imread('$image')
if image is None:
    print(f'Error: Could not load $image')
    exit(1)

result = recognizer.recognize_face(image)
print(f'Identity: {result.identity}')
print(f'Category: {result.category}')
print(f'Confidence: {result.confidence:.1%}')
print(f'Should alert: {result.should_alert}')
"
        else
            print_info "Running basic face recognition test..."
            python3 -c "
import sys
sys.path.insert(0, '.')
from src.visual import FaceRecognizer

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print('✓ Face recognizer initialized')
print(f'  Detection backend: $detection_backend')
print(f'  Embedding backend: $embedding_backend')
print(f'  Embedding size: {recognizer.embedding_dim}D')
print(f'  Threshold: $threshold')
print(f'  Registered faces: {len(recognizer.get_registered_identities())}')
print('✓ Face recognition is working!')
"
        fi
    else
        check_docker
        ensure_image
        print_header "Face Recognition Test (Container)"
        print_info "Detection: $detection_backend, Embedding: $embedding_backend, Threshold: $threshold"
        
        if [[ -n "$camera" ]]; then
            print_info "Starting camera face recognition (press 'q' to quit)..."
            docker_run python -c "
from src.visual import FaceRecognizer
from src.sensors.camera import Camera
from pathlib import Path
import cv2

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print(f'Detection: $detection_backend, Embedding: $embedding_backend ({recognizer.embedding_dim}D)')

# Load watch list
watch_list = Path('data/raw/faces/watch_list')
if watch_list.exists():
    results = recognizer.register_from_directory(str(watch_list))
    print(f'Loaded watch list: {results}')

with Camera() as camera:
    print('Camera started. Press q to quit.')
    for frame in camera.stream():
        result = recognizer.recognize_face(frame.image)
        output = frame.image.copy()
        if result.bbox:
            x, y, w, h = result.bbox
            color = (0, 0, 255) if result.should_alert else (0, 255, 0)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label = f'{result.identity}: {result.confidence:.0%}'
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Face Recognition', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
"
        elif [[ -n "$image" ]]; then
            print_info "Recognizing faces in: $image"
            docker_run python -c "
from src.visual import FaceRecognizer
import cv2

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print(f'Detection: $detection_backend, Embedding: $embedding_backend ({recognizer.embedding_dim}D)')

image = cv2.imread('$image')
if image is None:
    print(f'Error: Could not load $image')
    exit(1)

result = recognizer.recognize_face(image)
print(f'Identity: {result.identity}')
print(f'Category: {result.category}')
print(f'Confidence: {result.confidence:.1%}')
print(f'Should alert: {result.should_alert}')
"
        else
            print_info "Running basic face recognition test..."
            docker_run python -c "
from src.visual import FaceRecognizer

recognizer = FaceRecognizer(
    embedding_backend='$embedding_backend',
    detection_backend='$detection_backend',
    similarity_threshold=$threshold,
)
print('✓ Face recognizer initialized')
print(f'  Detection backend: $detection_backend')
print(f'  Embedding backend: $embedding_backend')
print(f'  Embedding size: {recognizer.embedding_dim}D')
print(f'  Threshold: $threshold')
print(f'  Registered faces: {len(recognizer.get_registered_identities())}')
print('✓ Face recognition is working!')
"
        fi
    fi
}

# Model comparison command
cmd_compare() {
    print_header "Face Recognition Model Comparison"
    print_info "Comparing all available face recognition models..."
    print_info "Make sure you have test images in:"
    print_info "  - data/raw/faces/watch_list/ (your face)"
    print_info "  - data/raw/faces/test_same_person/ (more of your face)"
    print_info "  - data/raw/faces/test_different_people/ (other people)"
    echo ""
    
    if [[ -n "$USE_LOCAL" ]]; then
        ensure_venv
        python3 scripts/model_comparison.py
    else
        check_docker
        ensure_image
        docker_run python scripts/model_comparison.py
    fi
}

# Sound classification test
cmd_classify() {
    local test_code="
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
    
    if [[ -n "$USE_LOCAL" ]]; then
        ensure_venv
        print_header "Sound Classification Test (Local)"
        python3 -c "$test_code"
    else
        check_docker
        ensure_image
        print_header "Sound Classification Test (Container)"
        docker_run python -c "$test_code"
    fi
}

# Camera test
cmd_camera() {
    local duration=5
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --duration) duration="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    local test_code="
from src.sensors.camera import Camera, CameraConfig, CameraBackend
import time

use_pi = False
try:
    with open('/proc/device-tree/model', 'r') as f:
        if 'raspberry' in f.read().lower():
            use_pi = True
except:
    pass

print(f'Using Pi Camera: {use_pi}')
backend = CameraBackend.PICAMERA if use_pi else CameraBackend.OPENCV
camera = Camera(CameraConfig(backend=backend))

print('Starting camera...')
camera.open()
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
    
    if [[ -n "$USE_LOCAL" ]]; then
        ensure_venv
        print_header "Camera Test (Local)"
        python3 -c "$test_code"
    else
        check_docker
        ensure_image
        print_header "Camera Test (Container)"
        docker_run python -c "$test_code"
    fi
}

# Demo command
cmd_demo() {
    print_header "IoT Home Security Demo"
    
    echo -e "${CYAN}This demo will test all major components:${NC}"
    echo "  1. Face Detection"
    echo "  2. Face Recognition"
    echo "  3. Sound Classification"
    echo ""
    
    if [[ -t 0 ]]; then
        read -p "Press Enter to continue..."
    fi
    echo ""
    
    # Run tests based on mode
    local args=""
    [[ -n "$USE_LOCAL" ]] && args="--local"
    
    cmd_camera
    echo ""
    cmd_recognize
    echo ""
    cmd_classify
    echo ""
    
    print_success "Demo complete!"
}

# Train command
cmd_train() {
    local model=""
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --face) model="face"; shift ;;
            --sound) model="sound"; shift ;;
            --all) model="all"; shift ;;
            *) shift ;;
        esac
    done
    
    print_header "Model Training"
    
    local train_face_code="
from src import FaceRecognizer
from pathlib import Path
import cv2

recognizer = FaceRecognizer(
    embedding_backend='opencv_dnn',
    detection_backend='opencv_dnn',
)

watch_list_dir = Path('data/raw/faces/watch_list')
if watch_list_dir.exists():
    results = recognizer.register_from_directory(str(watch_list_dir))
    print(f'Registered faces: {results}')
    print('✓ Face registration complete!')
else:
    print('No watch list directory found. Add faces to data/raw/faces/watch_list/')
"
    
    case "$model" in
        face)
            print_info "Starting face recognition training..."
            if [[ -n "$USE_LOCAL" ]]; then
                ensure_venv
                python3 -c "$train_face_code"
            else
                check_docker
                ensure_image
                docker_run python -c "$train_face_code"
            fi
            ;;
        sound)
            print_info "Sound training not yet implemented in container mode"
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

# Shell command (Docker only)
cmd_shell() {
    check_docker
    ensure_image
    print_header "Container Shell"
    print_info "Opening interactive shell in container..."
    docker_run /bin/bash
}

# Logs command
cmd_logs() {
    local follow=""
    
    # Parse remaining args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --follow|-f) follow="-f"; shift ;;
            *) shift ;;
        esac
    done
    
    print_header "Viewing Logs"
    
    if [[ -n "$USE_LOCAL" ]]; then
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
    else
        check_docker
        $DOCKER_COMPOSE logs $follow
    fi
}

# Install command (for local mode)
cmd_install() {
    local mode="dev"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prod) mode="prod"; shift ;;
            --dev) mode="dev"; shift ;;
            --pi) mode="pi"; shift ;;
            --arm64) mode="arm64"; shift ;;
            *) shift ;;
        esac
    done
    
    print_header "Installing IoT Home Security ($mode mode)"
    
    # Delegate to the unified install script
    if [[ -f "scripts/install.sh" ]]; then
        chmod +x scripts/install.sh
        ./scripts/install.sh "--$mode" --dir "$(pwd)"
    else
        print_error "scripts/install.sh not found"
        print_info "Falling back to basic installation..."
        ensure_venv
        pip install --upgrade pip wheel
        pip install -r requirements.txt
        pip install -e ".[dev]"
    fi
    
    print_success "Installation complete"
}

# ============================================================
# MAIN ENTRY POINT
# ============================================================

parse_global_flags() {
    local new_args=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --local)
                USE_LOCAL="1"
                shift
                ;;
            --rebuild)
                FORCE_REBUILD="1"
                shift
                ;;
            *)
                new_args+=("$1")
                shift
                ;;
        esac
    done
    
    # Return cleaned args
    echo "${new_args[@]}"
}

main() {
    # Parse global flags first
    local args
    args=$(parse_global_flags "$@")
    set -- $args
    
    local command="${1:-help}"
    shift || true
    
    # Auto-detect: if on Raspberry Pi without Docker, use local mode
    if is_raspberry_pi && ! command -v docker &> /dev/null; then
        print_info "Raspberry Pi detected without Docker - using local mode"
        USE_LOCAL="1"
    fi
    
    case "$command" in
        build)      cmd_build "$@" ;;
        start)      cmd_start "$@" ;;
        stop)       cmd_stop "$@" ;;
        restart)    cmd_restart "$@" ;;
        status)     cmd_status "$@" ;;
        api)        cmd_api "$@" ;;
        test)       cmd_test "$@" ;;
        detect)     cmd_detect "$@" ;;
        recognize)  cmd_recognize "$@" ;;
        compare)    cmd_compare "$@" ;;
        classify)   cmd_classify "$@" ;;
        camera)     cmd_camera "$@" ;;
        demo)       cmd_demo "$@" ;;
        train)      cmd_train "$@" ;;
        shell)      cmd_shell "$@" ;;
        logs)       cmd_logs "$@" ;;
        install)    cmd_install "$@" ;;
        help|--help|-h) show_help ;;
        *)
            print_error "Unknown command: $command"
            echo "Run './run.sh help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
