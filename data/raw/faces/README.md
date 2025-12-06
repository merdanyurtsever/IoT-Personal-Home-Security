# Face Images Directory

This directory contains face images for training and testing the face recognition system.

## Structure

```
faces/
├── known/           # Known faces for recognition (enrolled users)
│   ├── person_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── person_2/
│   │   └── ...
│   └── ...
├── unknown/         # Unknown/test faces
└── training/        # Training dataset
```

## Adding New Faces

1. Create a folder with the person's name under `known/`
2. Add multiple images of that person (5-10 images recommended)
3. Images should have clear, frontal views of the face
4. Varying lighting conditions improve recognition accuracy

## Image Requirements

- Format: JPEG or PNG
- Minimum resolution: 100x100 pixels for face region
- Clear, well-lit images preferred
- Multiple angles and expressions improve accuracy
