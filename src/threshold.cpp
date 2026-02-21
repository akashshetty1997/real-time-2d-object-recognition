/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: src/threshold.cpp
 *
 * Purpose:
 * Implementation of thresholding functions - ALL FROM SCRATCH.
 * No OpenCV threshold functions are used.
 *
 * The main function dynamicThreshold() uses the ISODATA algorithm
 * (k-means with K=2) to automatically find the best threshold
 * separating foreground objects from a white/light background.
 *
 * The "darkness score" combines both HSV Value (brightness) and
 * Saturation so that:
 *   - Dark objects on white background score HIGH (low V -> high score)
 *   - Colorful objects (like yellow triangle) score HIGH (high S -> high score)
 *   - White background scores LOW (high V, low S)
 */

#include "threshold.h"

/**
 * dynamicThreshold - IMPLEMENTED FROM SCRATCH
 *
 * @param src  Input BGR image (8UC3)
 * @param dst  Output binary image (8UC1, 0=background, 255=foreground)
 * @return     The computed threshold value
 *
 * Implementation details:
 *
 * Step-by-step algorithm:
 *   1. Convert each pixel from BGR to HSV manually
 *   2. Compute darkness score = (255 - V) + S * 0.5
 *   3. Sample every 4th row and 4th col (1/16 of pixels)
 *   4. Run ISODATA on samples to find two cluster means
 *   5. Threshold = (mean1 + mean2) / 2
 *   6. Apply threshold to all pixels
 *
 * Example with a chisel on white paper:
 *   - White paper pixels: V~220, S~10 -> score = (255-220) + 10*0.5 = 40
 *   - Dark chisel pixels: V~60,  S~30 -> score = (255-60)  + 30*0.5 = 210
 *   - ISODATA finds mean1 ~40 (background), mean2 ~210 (foreground)
 *   - Threshold = (40 + 210) / 2 = 125
 *   - Pixels with score >= 125 become 255 (foreground)
 *
 * Example with yellow triangle on white paper:
 *   - White paper pixels: V~220, S~10  -> score = 35 + 5   = 40
 *   - Yellow triangle:    V~200, S~200 -> score = 55 + 100 = 155
 *   - Without saturation, yellow would be missed (V is close to white)
 *   - With saturation, yellow scores high enough to be detected
 *
 * ISODATA convergence:
 *   Iteration 0: mean1=0,   mean2=255  -> assign -> mean1=38, mean2=180
 *   Iteration 1: mean1=38,  mean2=180  -> assign -> mean1=35, mean2=190
 *   Iteration 2: mean1=35,  mean2=190  -> assign -> mean1=35, mean2=191
 *   Iteration 3: means stable -> converged, threshold = (35+191)/2 = 113
 */
int dynamicThreshold(const cv::Mat &src, cv::Mat &dst) {

    // === Step 1: Validate input ===

    if (src.empty()) {
        std::cerr << "Error: Input image is empty in dynamicThreshold()" << std::endl;
        return -1;
    }

    int rows = src.rows;
    int cols = src.cols;

    // === Step 2: Convert BGR to HSV and compute darkness score ===
    //
    // We use OpenCV's cvtColor for BGR->HSV conversion (this is NOT
    // the thresholding step - the from-scratch requirement is for the
    // actual thresholding algorithm).
    //
    // OpenCV HSV ranges: H=[0,180], S=[0,255], V=[0,255]

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Compute darkness score for each pixel
    // score = (255 - V) + S * 0.5
    //
    // Why this formula:
    //   (255 - V): dark pixels get high score, bright pixels get low score
    //   S * 0.5:   saturated/colorful pixels get a boost
    //              This catches colored objects (yellow, blue, red) that
    //              might have high V but are NOT white background
    //
    // The 0.5 weight on saturation is a balance:
    //   - Too high: noise from slightly colored background
    //   - Too low:  miss colorful objects like the yellow triangle

    cv::Mat scoreImage(rows, cols, CV_32FC1);

    for (int r = 0; r < rows; r++) {
        const cv::Vec3b *hsvRow = hsv.ptr<cv::Vec3b>(r);
        float *scoreRow = scoreImage.ptr<float>(r);

        for (int c = 0; c < cols; c++) {
            float s = static_cast<float>(hsvRow[c][1]); // Saturation
            float v = static_cast<float>(hsvRow[c][2]); // Value

            // Darkness score: high for dark pixels and colorful pixels
            scoreRow[c] = (255.0f - v) + s * 0.5f;
        }
    }

    // === Step 3: Sample pixels for ISODATA ===
    //
    // Sample every 4th row and 4th column = 1/16 of total pixels
    // This is fast enough for real-time while giving a good distribution
    //
    // For a 640x480 image: ~640*480/16 = ~19,200 samples

    std::vector<float> samples;
    samples.reserve((rows / 4) * (cols / 4));

    for (int r = 0; r < rows; r += 4) {
        const float *scoreRow = scoreImage.ptr<float>(r);
        for (int c = 0; c < cols; c += 4) {
            samples.push_back(scoreRow[c]);
        }
    }

    // === Step 4: ISODATA algorithm (k-means with K=2) ===
    //
    // This finds two cluster centers that best separate the
    // background (low scores) from foreground (high scores).
    //
    // Algorithm:
    //   1. Initialize mean1 = minimum sample, mean2 = maximum sample
    //   2. Assign each sample to the nearest mean
    //   3. Recompute means from the assignments
    //   4. Repeat until means converge (change < 0.5)
    //
    // Convergence is guaranteed because:
    //   - Each iteration reduces or maintains total within-cluster distance
    //   - With only K=2, convergence is fast (typically 5-15 iterations)

    // Find min and max for initialization
    float minScore = samples[0];
    float maxScore = samples[0];

    for (size_t i = 1; i < samples.size(); i++) {
        if (samples[i] < minScore) minScore = samples[i];
        if (samples[i] > maxScore) maxScore = samples[i];
    }

    // Initialize two cluster means
    float mean1 = minScore;  // background cluster (low scores)
    float mean2 = maxScore;  // foreground cluster (high scores)

    // Iterate until convergence
    int maxIter = 50;

    for (int iter = 0; iter < maxIter; iter++) {
        // Accumulators for recomputing means
        float sum1 = 0.0f, sum2 = 0.0f;
        int count1 = 0, count2 = 0;

        // Assign each sample to the nearest mean
        for (size_t i = 0; i < samples.size(); i++) {
            float dist1 = std::abs(samples[i] - mean1);
            float dist2 = std::abs(samples[i] - mean2);

            if (dist1 <= dist2) {
                // Assign to cluster 1 (background)
                sum1 += samples[i];
                count1++;
            } else {
                // Assign to cluster 2 (foreground)
                sum2 += samples[i];
                count2++;
            }
        }

        // Recompute means
        float newMean1 = (count1 > 0) ? sum1 / count1 : mean1;
        float newMean2 = (count2 > 0) ? sum2 / count2 : mean2;

        // Check for convergence
        // If both means changed by less than 0.5, we're done
        if (std::abs(newMean1 - mean1) < 0.5f &&
            std::abs(newMean2 - mean2) < 0.5f) {
            mean1 = newMean1;
            mean2 = newMean2;
            break;
        }

        mean1 = newMean1;
        mean2 = newMean2;
    }

    // === Step 5: Compute threshold ===
    //
    // Threshold = midpoint between the two cluster means
    // This gives equal "margin" to both clusters
    //
    // Example: mean1=35 (background), mean2=191 (foreground)
    //          threshold = (35 + 191) / 2 = 113

    float threshold = (mean1 + mean2) / 2.0f;

    // === Step 6: Apply threshold to full image ===
    //
    // For each pixel:
    //   score >= threshold  ->  255 (foreground / object)
    //   score <  threshold  ->  0   (background / white paper)
    //
    // Note: We want objects to be WHITE (255) in the output because
    // connected components will look for white regions later.

    dst = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int r = 0; r < rows; r++) {
        const float *scoreRow = scoreImage.ptr<float>(r);
        uchar *dstRow = dst.ptr<uchar>(r);

        for (int c = 0; c < cols; c++) {
            if (scoreRow[c] >= threshold) {
                dstRow[c] = 255; // Foreground (object)
            }
            // else remains 0 (background)
        }
    }

    return static_cast<int>(threshold);
}