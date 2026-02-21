
# Real-time 2D Object Recognition

**CS 5330 - Pattern Recognition & Computer Vision**  
**Project 3**  
**Author:** Akash Shridhar Shetty | Skandhan Madhusudhana

## Overview

A real-time 2D object recognition system that identifies objects placed against a white background. The system uses a pipeline of computer vision techniques to threshold, clean, segment, extract features, and classify objects using nearest-neighbor matching with scaled Euclidean distance.

The system can differentiate at least 5 objects by their 2D shape, in a translation, scale, and rotation invariant manner.

## Pipeline

1. **Thresholding** (FROM SCRATCH) - ISODATA algorithm (k-means K=2) with darkness score combining HSV value and saturation
2. **Morphological Cleanup** - Closing then opening to fill holes and remove noise
3. **Segmentation** - Connected components analysis with colored region visualization
4. **Feature Extraction** - Oriented bounding box, axis of least central moment, percent filled, aspect ratio, Hu moments
5. **Training** - Interactive labeling of objects, stored in CSV database
6. **Classification** - Nearest-neighbor with scaled Euclidean distance
7. **Evaluation** - Confusion matrix across multiple test images
8. **Embeddings** - ResNet18 one-shot classification (coming soon)

## Setup

- **OS:** macOS (M4 Mac)
- **Language:** C++17
- **Dependencies:** OpenCV 4 (via Homebrew)
- **IDE:** VS Code
- **Build:** Makefile with pkg-config

## Building

```bash
make clean
make
```

## Usage

```bash
# Live camera mode
./objrec

# Single image mode
./objrec path/to/image.jpg

# Directory mode (navigate with n/b keys)
./objrec dev_images/

# Batch mode (auto-process all images for report)
./objrec --batch dev_images/
```

## Key Controls

| Key | Action |
|-----|--------|
| `t` | Toggle threshold view |
| `m` | Toggle morphology view |
| `s` | Toggle segmentation view |
| `f` | Toggle features overlay |
| `c` | Toggle classification mode |
| `l` | Label/train current object |
| `p` | Save screenshots to task folders |
| `n` | Next image (directory mode) |
| `b` | Previous image (directory mode) |
| `q` / `ESC` | Quit |

## Project Structure

```
real-time-2d-object-recognition/
├── Makefile
├── README.md
├── include/
│   ├── vision.h          # Common includes and data structures
│   ├── threshold.h        # Thresholding (from scratch)
│   ├── morphology.h       # Morphological filtering
│   ├── segment.h          # Connected components
│   ├── features.h         # Feature extraction
│   ├── classify.h         # Training DB and classification
│   ├── batch.h            # Batch processing
│   └── embeddings.h       # ResNet18 embeddings
├── src/
│   ├── main.cpp           # Main loop and UI
│   ├── threshold.cpp      # ISODATA thresholding (FROM SCRATCH)
│   ├── morphology.cpp     # Closing + opening cleanup
│   ├── segment.cpp        # Connected components with color map
│   ├── features.cpp       # Moments, oriented bbox, Hu moments
│   ├── classify.cpp       # Scaled Euclidean NN classifier
│   ├── batch.cpp          # Batch report image generation
│   └── utilities.cpp      # Prof's embedding utilities
├── dev_images/            # Development image set
├── models/
│   └── or2d-normmodel-007.onnx  # ResNet18 model
└── data/
    ├── task1_threshold/   # Saved threshold images
    ├── task2_morphology/  # Saved morphology images
    ├── task3_segmentation/# Saved segment images
    ├── task4_features/    # Saved feature overlays + vectors
    ├── task5_training/    # Training DB (CSV)
    ├── task6_classification/ # Classification results
    ├── task7_evaluation/  # Confusion matrix
    └── task8_embeddings/  # Embedding results
```

## From Scratch Implementation

**Thresholding** (`src/threshold.cpp`) is implemented entirely from scratch without using OpenCV's threshold functions:
- Converts image to HSV color space
- Computes a "darkness score" combining value and saturation channels
- Samples 1/16 of pixels for efficiency
- Runs ISODATA algorithm (iterative k-means with K=2) to find optimal threshold
- Applies threshold to separate foreground objects from background

## Acknowledgements

- Professor Bruce Maxwell - CS 5330 course materials and utilities code
- OpenCV library for image processing functions
