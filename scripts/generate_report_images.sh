#!/bin/bash
# Name: Akash Shridhar Shetty
# Date: February 2025
# File: scripts/generate_report_images.sh
#
# Purpose:
# Automatically processes dev images through the pipeline
# and saves output images organized by task for the report.
#
# Usage:
#   chmod +x scripts/generate_report_images.sh
#   ./scripts/generate_report_images.sh

echo "=== Generating Report Images ==="

# Create task directories
mkdir -p data/task1_threshold
mkdir -p data/task2_morphology
mkdir -p data/task3_segmentation
mkdir -p data/task4_features
mkdir -p data/task5_training
mkdir -p data/task6_classification
mkdir -p data/task7_evaluation
mkdir -p data/task8_embeddings

# Check if dev_images directory exists
if [ ! -d "dev_images" ]; then
    echo "Error: dev_images/ directory not found."
    echo "Place the development image set in project3/dev_images/"
    exit 1
fi

# Check if objrec binary exists
if [ ! -f "objrec" ]; then
    echo "Building objrec..."
    make
fi

# Count images
count=0

# Process each image in dev_images
for img in dev_images/*.{jpg,jpeg,png}; do
    # Skip if no files match the glob
    [ -f "$img" ] || continue

    # Extract filename without extension for naming
    filename=$(basename "$img")
    name="${filename%.*}"

    echo "Processing: $filename"

    # Use objrec in single image mode with auto-save
    # We'll create a small C++ tool for batch processing instead
    # For now, copy originals and we'll process them with a batch tool

    # Copy original to task1
    cp "$img" "data/task1_threshold/orig_${name}.png"

    count=$((count + 1))
done

echo ""
echo "Copied $count original images."
echo "Now run the batch processor to generate pipeline outputs:"
echo "  ./objrec_batch dev_images/"
echo ""