#!/bin/bash
# Download and install antelopev2 model for InsightFace
set -e
MODEL_DIR="$HOME/.insightface/models/antelopev2"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Download ONNX model
MODEL_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_person_2.5g.onnx"
if [ ! -f scrfd_person_2.5g.onnx ]; then
    echo "Downloading scrfd_person_2.5g.onnx..."
    wget "$MODEL_URL" -O scrfd_person_2.5g.onnx
else
    echo "scrfd_person_2.5g.onnx already exists."
fi

echo "scrfd_person_2.5g model is ready in $MODEL_DIR"