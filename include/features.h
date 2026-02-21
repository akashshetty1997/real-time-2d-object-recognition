/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: include/features.h
 *
 * Purpose:
 * Header for feature extraction and region analysis functions.
 * Computes properties for segmented regions including:
 *   - Centroid (cx, cy)
 *   - Axis of least central moment (orientation angle theta)
 *   - Oriented bounding box
 *   - Percent filled (region area / oriented bbox area)
 *   - Bounding box aspect ratio (height / width)
 *   - Hu moments (7 rotation/scale/translation invariant moments)
 *
 * All features are translation, scale, and rotation invariant
 * so they work regardless of object position/orientation in frame.
 */

#ifndef FEATURES_H
#define FEATURES_H

#include "vision.h"

/*
 * computeRegionProps
 *
 * Compute all properties for a single region given a region map and label.
 * Uses OpenCV moments() to compute spatial and central moments, then
 * derives orientation, bounding box, and invariant features.
 *
 * @param regionMap  Labeled region map (32SC1) from segmentRegions()
 * @param regionID   The label of the region to analyze (e.g., 1 for largest)
 * @param props      Output RegionProps struct with all computed properties
 * @return           0 on success, -1 on error
 */
int computeRegionProps(const cv::Mat &regionMap, int regionID, RegionProps &props);

/*
 * buildFeatureVector
 *
 * Extract a feature vector from RegionProps for classification.
 * Features (all invariant to translation, scale, rotation):
 *   [0] percentFilled      - how much of the oriented bbox is filled
 *   [1] bboxAspectRatio    - height/width ratio of oriented bbox
 *   [2-8] huMoments[0..6]  - 7 Hu moments (log-transformed)
 *
 * @param props     Input region properties
 * @param features  Output feature vector (9 elements)
 * @return          0 on success, -1 on error
 */
int buildFeatureVector(const RegionProps &props, std::vector<float> &features);

/*
 * drawFeatures
 *
 * Draw feature overlay on an image: oriented bounding box, primary axis,
 * centroid dot, and optional text label.
 *
 * @param frame  Image to draw on (modified in place)
 * @param props  Region properties to visualize
 * @param label  Optional text label to display near the object
 */
void drawFeatures(cv::Mat &frame, const RegionProps &props, const std::string &label = "");

#endif // FEATURES_H