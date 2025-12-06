# Model Training Guide

This guide covers training the machine learning models for the IoT Home Security system.

## Overview

The system uses two main ML models:
1. **Face Detection/Recognition** - Identifies people
2. **Sound Classification** - Detects security-relevant sounds

## Prerequisites

### Hardware Requirements

- GPU recommended for training (NVIDIA with CUDA support)
- Minimum 16GB RAM
- 50GB+ storage for datasets

### Software Requirements

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install training dependencies
pip install -r requirements.txt
```

## Face Recognition Training

### 1. Prepare Dataset

Organize face images in the following structure:

```
data/raw/faces/known/
├── person_1/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── person_2/
│   ├── img001.jpg
│   └── ...
└── ...
```

**Requirements per person:**
- Minimum 5-10 images
- Various lighting conditions
- Different angles (frontal preferred)
- Clear, focused images
- Minimum 100x100 pixels for face region

### 2. Training with Notebook

Open and run `notebooks/02_face_recognition_training.ipynb`:

```python
# Key steps in the notebook:
# 1. Load face images
# 2. Extract embeddings using FaceNet/ArcFace
# 3. Train SVM classifier on embeddings
# 4. Evaluate accuracy
# 5. Export model
```

### 3. Model Architectures

#### Option A: FaceNet (Recommended for RPi)
- 512-dimensional embeddings
- Good balance of accuracy and speed
- Works well with TFLite

#### Option B: ArcFace
- State-of-the-art accuracy
- Higher computational cost
- Better for larger face databases

### 4. Training Tips

```python
# Data augmentation for better generalization
augmentations = [
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomResizedCrop(160, scale=(0.8, 1.0)),
]
```

## Sound Classification Training

### 1. ESC-50 Dataset

The project includes the ESC-50 dataset for environmental sound classification:

```
ESC-50-master/
├── audio/          # 2000 audio clips (5 seconds each)
└── meta/
    └── esc50.csv   # Metadata with labels
```

### 2. Security-Relevant Classes

Focus training on these classes:

| Class | ESC-50 Category | Priority |
|-------|-----------------|----------|
| glass_breaking | 38 | High |
| door_wood_knock | 2 | High |
| dog | 0 | Medium |
| siren | 39 | High |
| crying_baby | 20 | Medium |
| gunshot | 37 | High |
| footsteps | 35 | Low |

### 3. Training with Notebook

Open and run `notebooks/03_sound_classification_training.ipynb`:

```python
# Key training parameters
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
N_MELS = 128
EPOCHS = 50
BATCH_SIZE = 32
```

### 4. Model Architecture

```python
# Recommended CNN architecture for RPi
model = tf.keras.Sequential([
    # Input: Mel spectrogram (128 x 216 x 1)
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### 5. Data Augmentation

```python
# Audio augmentation techniques
def augment_audio(audio, sr):
    # Time stretching
    audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9, 1.1))
    
    # Pitch shifting
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.randint(-2, 2))
    
    # Add noise
    noise = np.random.randn(len(audio)) * 0.005
    audio = audio + noise
    
    return audio
```

## Model Optimization

### 1. TensorFlow Lite Conversion

```python
# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. Full Integer Quantization (Best for RPi)

```python
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 128, 216, 1).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

### 3. Benchmarking

```python
# Test inference speed
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Warmup
for _ in range(10):
    interpreter.invoke()

# Benchmark
import time
start = time.time()
for _ in range(100):
    interpreter.invoke()
print(f"Average inference: {(time.time() - start) / 100 * 1000:.2f} ms")
```

## Expected Results

### Face Recognition
- Training accuracy: >95%
- Validation accuracy: >90%
- Inference time on RPi 4: <100ms

### Sound Classification
- ESC-50 accuracy: >70% (security subset)
- False positive rate: <5%
- Inference time on RPi 4: <200ms

## Common Issues

### Out of Memory During Training
- Reduce batch size
- Use mixed precision training
- Use gradient checkpointing

### Poor Accuracy
- Add more training data
- Increase data augmentation
- Use transfer learning from pretrained models

### Slow Inference on RPi
- Use INT8 quantization
- Reduce model size
- Consider Coral USB Accelerator
