/*
 * Name: Akash Shridhar Shetty , Skandhan Madhusudhana
 * Date: February 2025
 * File: include/vision.h
 *
 * Purpose:
 * Common includes, definitions, and data structures shared across
 * the entire 2D object recognition pipeline.
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
#include <set>
#include <iomanip>

// === OpenCV includes ===
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

// === Region properties struct ===
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
struct TrainingEntry {
    std::string label;
    std::vector<float> features;
};

#endif // VISION_H