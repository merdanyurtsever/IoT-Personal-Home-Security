#!/bin/bash
#
# IoT Home Security - Quick Install Script
# ========================================
#
# One-liner installation for different environments.
#
# Usage:
#   curl -sSL <url>/install.sh | bash                    # Auto-detect
#   curl -sSL <url>/install.sh | bash -s -- --dev        # Development
#   curl -sSL <url>/install.sh | bash -s -- --pi         # Raspberry Pi
#   curl -sSL <url>/install.sh | bash -s -- --arm64      # ARM64 VM
#
# Or locally:
#   ./scripts/install.sh [--dev|--prod|--pi|--arm64]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${CYAN}ℹ $1${NC}"; }

# Detect environment
detect_environment() {
    # Check for Raspberry Pi
    if [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
        echo "pi"
        return
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
        echo "arm64"
        return
    fi
    
    echo "dev"
}

# Install system dependencies
install_system_deps() {
    local env_type="$1"
    
    print_header "Installing System Dependencies"
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        
        case "$env_type" in
            pi)
                sudo apt-get install -y \
                    python3-pip python3-venv python3-dev \
                    libopencv-dev python3-opencv \
                    libatlas-base-dev libhdf5-dev \
                    portaudio19-dev libffi-dev libssl-dev \
                    python3-picamera2 python3-libcamera \
                    cmake build-essential git
                ;;
            arm64)
                sudo apt-get install -y \
                    python3-pip python3-venv python3-dev \
                    libopencv-dev libatlas-base-dev libhdf5-dev \
                    cmake build-essential v4l-utils libv4l-dev git
                ;;
            *)
                sudo apt-get install -y \
                    python3-pip python3-venv python3-dev \
                    libopencv-dev cmake build-essential git
                ;;
        esac
        print_success "System dependencies installed"
        
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y \
            python3-pip python3-devel \
            opencv-devel cmake gcc-c++ git
        print_success "System dependencies installed"
        
    elif command -v brew &> /dev/null; then
        brew install python cmake opencv
        print_success "System dependencies installed"
        
    else
        print_warning "Unknown package manager. Please install dependencies manually."
    fi
}

# Setup Python environment
setup_python_env() {
    local project_dir="$1"
    local env_type="$2"
    
    print_header "Setting Up Python Environment"
    
    cd "$project_dir"
    
    # Create virtual environment (use .venv for consistency with run.sh)
    if [[ ! -d ".venv" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install requirements
    print_info "Installing Python packages..."
    
    case "$env_type" in
        pi)
            pip install -r requirements-arm64.txt 2>/dev/null || pip install -r requirements.txt
            pip install RPi.GPIO gpiozero picamera2 tflite-runtime 2>/dev/null || true
            ;;
        arm64)
            pip install -r requirements-arm64.txt 2>/dev/null || pip install -r requirements.txt
            ;;
        dev)
            pip install -r requirements.txt
            pip install -e ".[dev]"
            ;;
        *)
            pip install -r requirements.txt
            pip install -e .
            ;;
    esac
    
    print_success "Python environment ready"
}

# Setup directories
setup_directories() {
    local project_dir="$1"
    
    print_header "Setting Up Directories"
    
    cd "$project_dir"
    
    mkdir -p data/raw/faces/watch_list
    mkdir -p data/raw/faces/uploads
    mkdir -p data/processed
    mkdir -p data/models/face_detection
    mkdir -p data/models/face_recognition
    mkdir -p data/models/sound_classification
    mkdir -p logs
    
    print_success "Directories created"
}

# Setup systemd service (for Pi)
setup_service() {
    local project_dir="$1"
    
    print_header "Setting Up Systemd Service"
    
    # Create service file
    cat > /tmp/security-system.service << EOF
[Unit]
Description=IoT Home Security System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$project_dir
Environment=PYTHONPATH=$project_dir/src
ExecStart=$project_dir/venv/bin/python3 $project_dir/raspberry_pi/main.py --config $project_dir/raspberry_pi/config/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/security-system.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable security-system
    
    print_success "Systemd service configured"
    print_info "Start with: sudo systemctl start security-system"
}

# Main installation
main() {
    local env_type=""
    local project_dir=""
    local skip_deps=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dev) env_type="dev"; shift ;;
            --prod) env_type="prod"; shift ;;
            --pi) env_type="pi"; shift ;;
            --arm64) env_type="arm64"; shift ;;
            --dir) project_dir="$2"; shift 2 ;;
            --skip-deps) skip_deps="1"; shift ;;
            *) shift ;;
        esac
    done
    
    # Auto-detect environment if not specified
    if [[ -z "$env_type" ]]; then
        env_type=$(detect_environment)
        print_info "Auto-detected environment: $env_type"
    fi
    
    # Determine project directory
    if [[ -z "$project_dir" ]]; then
        if [[ -f "run.sh" ]]; then
            project_dir="$(pwd)"
        elif [[ -f "../run.sh" ]]; then
            project_dir="$(cd .. && pwd)"
        else
            project_dir="/home/$USER/security"
        fi
    fi
    
    print_header "IoT Home Security Installation"
    echo -e "Environment: ${CYAN}$env_type${NC}"
    echo -e "Directory:   ${CYAN}$project_dir${NC}"
    echo ""
    
    # Install system dependencies
    if [[ -z "$skip_deps" ]]; then
        install_system_deps "$env_type"
    fi
    
    # Setup Python environment
    setup_python_env "$project_dir" "$env_type"
    
    # Setup directories
    setup_directories "$project_dir"
    
    # Setup service for Pi
    if [[ "$env_type" == "pi" ]]; then
        setup_service "$project_dir"
    fi
    
    # Make run script executable
    chmod +x "$project_dir/run.sh" 2>/dev/null || true
    
    print_header "Installation Complete!"
    
    echo -e "To get started:"
    echo -e "  ${CYAN}cd $project_dir${NC}"
    echo -e "  ${CYAN}./run.sh help${NC}         # Show all commands"
    echo -e "  ${CYAN}./run.sh demo${NC}         # Run demo"
    echo -e "  ${CYAN}./run.sh start${NC}        # Start system"
    
    if [[ "$env_type" == "pi" ]]; then
        echo ""
        echo -e "For Raspberry Pi:"
        echo -e "  ${CYAN}sudo systemctl start security-system${NC}   # Start service"
        echo -e "  ${CYAN}sudo systemctl status security-system${NC}  # Check status"
    fi
    
    echo ""
    print_success "Ready to use!"
}

main "$@"
