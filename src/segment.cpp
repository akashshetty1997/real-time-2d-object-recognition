/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: src/segment.cpp
 *
 * Purpose:
 * Implementation of connected components segmentation.
 * Uses OpenCV's connectedComponentsWithStats to find and label regions.
 *
 * Takes a cleaned binary image and produces:
 *   1. A labeled region map (each pixel has its region ID)
 *   2. A colored visualization (each region shown in a unique color)
 *
 * Filters out:
 *   - Regions smaller than minRegionSize (noise)
 *   - Background region (label 0)
 *   - Keeps only the largest N regions
 *
 * Strategy for selecting the "main" object:
 *   - Sort valid regions by area (largest first)
 *   - Relabel so label 1 = largest, label 2 = second largest, etc.
 *   - Later stages (features, classification) use label 1 as the main object
 *
 * Example with eraser on whiteboard:
 *   connectedComponents finds: label 0 (whiteboard), label 1 (eraser),
 *     label 2 (small shadow speck), label 3 (tiny noise)
 *   After filtering (area >= 500): label 1 (eraser) survives
 *   After relabeling: eraser = label 1
 *   Color map: eraser = red, everything else = black
 */

#include "segment.h"

/**
 * segmentRegions - Run connected components and return colored region map
 *
 * @param src           Input binary image (8UC1, 255 = foreground)
 * @param regionMap     Output labeled region map (32SC1), relabeled 1..N
 * @param colorMap      Output colored visualization (8UC3)
 * @param minRegionSize Ignore regions smaller than this (default 500)
 * @param maxRegions    Max number of regions to keep (default 5)
 * @return              Number of valid regions found
 *
 * Implementation details:
 *
 * OpenCV connectedComponentsWithStats returns:
 *   - labels:    each pixel labeled 0 (background) or 1..N (regions)
 *   - stats:     for each label, [left, top, width, height, area]
 *                accessed via cv::CC_STAT_LEFT, CC_STAT_TOP, etc.
 *   - centroids: for each label, [cx, cy]
 *
 * Step-by-step:
 *   1. Run connectedComponentsWithStats on binary image
 *   2. Collect regions that pass the minimum size filter
 *   3. Sort valid regions by area (largest first)
 *   4. Keep only the top maxRegions
 *   5. Build a relabeled region map (1 = largest, 2 = second, etc.)
 *   6. Generate colored visualization with a fixed color palette
 *
 * Example walkthrough:
 *   Input: binary image with eraser (area=15000), shadow (area=800),
 *          noise speck (area=50)
 *   Step 1: connectedComponents finds 4 labels (0=bg, 1, 2, 3)
 *   Step 2: filter by area >= 500:
 *           label 1 (area=15000) -> passes
 *           label 2 (area=800)   -> passes
 *           label 3 (area=50)    -> filtered out
 *   Step 3: sort by area: [label 1 (15000), label 2 (800)]
 *   Step 4: keep top 5 -> both kept
 *   Step 5: relabel: label 1 -> new label 1, label 2 -> new label 2
 *   Step 6: color: label 1 = red, label 2 = green
 */
int segmentRegions(const cv::Mat &src, cv::Mat &regionMap, cv::Mat &colorMap,
                   int minRegionSize, int maxRegions) {

    // === Step 1: Validate input ===

    if (src.empty()) {
        std::cerr << "Error: Input image is empty in segmentRegions()" << std::endl;
        return 0;
    }

    // === Step 2: Run connected components with stats ===
    //
    // connectedComponentsWithStats gives us:
    //   nLabels:   total number of labels (including background = 0)
    //   labels:    same size as input, each pixel = its label (0..nLabels-1)
    //   stats:     nLabels x 5 matrix (int32):
    //              row i = [left, top, width, height, area] for label i
    //   centroids: nLabels x 2 matrix (double):
    //              row i = [cx, cy] for label i

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // === Step 3: Filter regions by size ===
    //
    // Skip label 0 (background - this is always the largest "region")
    // Only keep regions with area >= minRegionSize
    //
    // We store each valid region's info for sorting

    struct RegionInfo {
        int area;           // number of pixels in this region
        int originalLabel;  // label from connectedComponents
        double cx, cy;      // centroid coordinates
    };

    std::vector<RegionInfo> validRegions;

    for (int i = 1; i < nLabels; i++) { // skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area >= minRegionSize) {
            RegionInfo info;
            info.area = area;
            info.originalLabel = i;
            info.cx = centroids.at<double>(i, 0);
            info.cy = centroids.at<double>(i, 1);
            validRegions.push_back(info);
        }
    }

    // === Step 4: Sort by area (largest first) ===
    //
    // This ensures label 1 in our relabeled map is always the biggest
    // region, which is most likely the main object.

    std::sort(validRegions.begin(), validRegions.end(),
              [](const RegionInfo &a, const RegionInfo &b) {
                  return a.area > b.area;
              });

    // === Step 5: Keep only top maxRegions ===

    if (static_cast<int>(validRegions.size()) > maxRegions) {
        validRegions.resize(maxRegions);
    }

    int numValid = static_cast<int>(validRegions.size());

    // === Step 6: Build relabeled region map ===
    //
    // Create a mapping: original_label -> new_label (1..numValid)
    // 0 remains background
    //
    // This ensures:
    //   new label 1 = largest region (main object)
    //   new label 2 = second largest region
    //   etc.
    //
    // We use a map for O(log N) lookup per pixel, which is fast
    // enough since N is small (maxRegions <= 5 typically)

    std::map<int, int> labelMapping;
    for (int i = 0; i < numValid; i++) {
        labelMapping[validRegions[i].originalLabel] = i + 1; // 1-indexed
    }

    // Create the output region map (32-bit signed int, same as OpenCV labels)
    regionMap = cv::Mat::zeros(labels.size(), CV_32SC1);

    for (int r = 0; r < labels.rows; r++) {
        const int *labelRow = labels.ptr<int>(r);
        int *regionRow = regionMap.ptr<int>(r);

        for (int c = 0; c < labels.cols; c++) {
            auto it = labelMapping.find(labelRow[c]);
            if (it != labelMapping.end()) {
                regionRow[c] = it->second;
            }
            // else remains 0 (background or filtered-out region)
        }
    }

    // === Step 7: Generate colored visualization ===
    //
    // Use a fixed color palette so colors are consistent across frames.
    // Background = black, each region gets a distinct bright color.
    //
    // Palette has 8 colors (index 0 = background):
    //   0: black (background)
    //   1: red       (largest region / main object)
    //   2: green     (second largest)
    //   3: blue
    //   4: yellow
    //   5: magenta
    //   6: cyan
    //   7: orange
    //
    // Colors are in BGR format (OpenCV convention)

    std::vector<cv::Vec3b> palette = {
        cv::Vec3b(0, 0, 0),       // 0: background (black)
        cv::Vec3b(0, 0, 255),     // 1: red
        cv::Vec3b(0, 255, 0),     // 2: green
        cv::Vec3b(255, 0, 0),     // 3: blue
        cv::Vec3b(0, 255, 255),   // 4: yellow
        cv::Vec3b(255, 0, 255),   // 5: magenta
        cv::Vec3b(255, 255, 0),   // 6: cyan
        cv::Vec3b(0, 128, 255)    // 7: orange
    };

    // Create the colored output image
    colorMap = cv::Mat::zeros(regionMap.size(), CV_8UC3);

    for (int r = 0; r < regionMap.rows; r++) {
        const int *regionRow = regionMap.ptr<int>(r);
        cv::Vec3b *colorRow = colorMap.ptr<cv::Vec3b>(r);

        for (int c = 0; c < regionMap.cols; c++) {
            int regionID = regionRow[c];

            // Look up color from palette
            // Use modulo to wrap around if more regions than colors
            if (regionID > 0 && regionID < static_cast<int>(palette.size())) {
                colorRow[c] = palette[regionID];
            } else if (regionID > 0) {
                // More regions than palette colors -> wrap around
                colorRow[c] = palette[(regionID % (palette.size() - 1)) + 1];
            }
            // else regionID == 0 -> stays black (background)
        }
    }

    return numValid;
}