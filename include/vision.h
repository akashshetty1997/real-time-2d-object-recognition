/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: include/vision.h
 *
 * Purpose:
 * Common includes, definitions, and data structures shared across
 * the entire 2D object recognition pipeline. All modules include
 * this header for shared types and OpenCV access.
 */

#ifndef VISION_H
#define VISION_H

// === Standard library includes ===
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <map>

// === OpenCV includes ===
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

// === Region properties struct ===
//
// Holds all computed properties for a single segmented region.
// Populated by computeRegionProps() in features.cpp.
//
// Fields:
//   label          - region ID from connected components
//   area           - number of pixels in the region
//   cx, cy         - centroid coordinates
//   theta          - orientation angle of primary axis (radians)
//   minE1, maxE1   - extent along primary axis (min should be negative)
//   minE2, maxE2   - extent along secondary axis (min should be negative)
//   percentFilled  - ratio of region pixels to oriented bounding box area
//   bboxAspectRatio - height/width ratio of oriented bounding box
//   huMoments[7]   - 7 Hu moments (rotation/scale/translation invariant)
//   orientedBBox   - OpenCV RotatedRect for the oriented bounding box
struct RegionProps {
    int label;
    int area;
    int cx, cy;
    float theta;
    float minE1, maxE1;
    float minE2, maxE2;
    float percentFilled;
    float bboxAspectRatio;
    double huMoments[7];
    cv::RotatedRect orientedBBox;
};

// === Training entry struct ===
//
// Holds a single labeled feature vector for the training database.
// Used by both hand-crafted features and embedding-based classification.
//
// Fields:
//   label    - human-readable object name (e.g., "chisel", "wrench")
//   features - feature vector (size depends on feature type)
struct TrainingEntry {
    std::string label;
    std::vector<float> features;
};

#endif // VISION_H