/*
 * Name: Akash Shridhar Shetty , Skandhan Madhusudhana
 * Date: February 2025
 * File: include/threshold.h
 *
 * Purpose:
 * Header for thresholding functions - IMPLEMENTED FROM SCRATCH.
 * No OpenCV threshold functions are used.
 * Separates dark objects from a light/white background.
 *
 * Approach:
 * Uses ISODATA algorithm (k-means with K=2) on sampled pixels to
 * dynamically find the optimal threshold. Also considers saturation
 * so that colorful objects (like the yellow triangle) are detected
 * even if their intensity/value is high.
 */

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include "vision.h"

/*
 * dynamicThreshold
 *
 * Thresholds an image using the ISODATA algorithm (K=2 k-means)
 * on a sampled subset of pixels. Combines HSV value and saturation
 * channels for robust foreground/background separation.
 *
 * IMPLEMENTED FROM SCRATCH - no cv::threshold used.
 *
 * @param src  Input BGR image (8UC3)
 * @param dst  Output binary image (8UC1, 0=background, 255=foreground)
 * @return     The computed threshold value
 *
 * Algorithm:
 *   1. Convert BGR to HSV (pixel-by-pixel, from scratch)
 *   2. Compute a "darkness score" for each pixel:
 *      score = (255 - V) + S * 0.5
 *      This makes dark pixels AND saturated/colorful pixels score high
 *   3. Sample 1/16 of pixels from the score image
 *   4. Run ISODATA (iterative k-means, K=2) on sampled scores:
 *      a. Initialize two means: mean1 = min, mean2 = max
 *      b. Assign each sample to nearest mean
 *      c. Recompute means from assignments
 *      d. Repeat until convergence (means stop changing)
 *   5. Threshold = midpoint between the two converged means
 *   6. Apply threshold to full score image:
 *      pixel >= threshold  =>  255 (foreground/object)
 *      pixel <  threshold  =>  0   (background)
 */
int dynamicThreshold(const cv::Mat &src, cv::Mat &dst);

#endif // THRESHOLD_H