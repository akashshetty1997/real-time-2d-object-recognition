/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
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
 *   - Regions that touch the image boundary (cardboard/table edges)
 *   - Keeps only the largest N regions
 *
 * Strategy for selecting the "main" object:
 *   - Filter out boundary-touching regions (these are background/table)
 *   - Sort valid regions by area (largest first)
 *   - Relabel so label 1 = largest, label 2 = second largest, etc.
 *   - Later stages (features, classification) use label 1 as the main object
 *
 * Why boundary filtering matters:
 *   When shooting objects on white paper, the cardboard/table edges
 *   around the paper appear as large foreground regions. Without
 *   boundary filtering, these dominate and the actual object (label 1)
 *   ends up being the cardboard corner, not the tool/object.
 *
 * Example with T-tool on white paper:
 *   connectedComponents finds:
 *     label 0 (white paper background)
 *     label 1 (cardboard right edge, area=50000, touches right boundary)
 *     label 2 (cardboard left corner, area=30000, touches left boundary)
 *     label 3 (T-tool, area=8000, does NOT touch boundary)
 *     label 4 (tiny noise speck, area=50)
 *   After boundary + size filtering:
 *     label 3 (T-tool) is the only valid region
 *   After relabeling: T-tool = label 1
 *   Color map: T-tool = red
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
 * Key change from original:
 *   Added boundary touching check - regions whose bounding box touches
 *   within 2 pixels of any image edge are discarded. This removes the
 *   cardboard/table edges that surround the white paper workspace.
 */
int segmentRegions(const cv::Mat &src, cv::Mat &regionMap, cv::Mat &colorMap,
                   int minRegionSize, int maxRegions) {

    // === Step 1: Validate input ===

    if (src.empty()) {
        std::cerr << "Error: Input image is empty in segmentRegions()" << std::endl;
        return 0;
    }

    // === Step 2: Run connected components with stats ===

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);

    // === Step 3: Filter regions by size AND boundary contact ===
    //
    // Two filters applied here:
    //
    // Filter A - Minimum size:
    //   Removes tiny noise specks that survived morphological cleanup.
    //   Default threshold: 500 pixels.
    //
    // Filter B - Boundary touching:
    //   Removes regions whose bounding box touches the image edge.
    //   The cardboard/table surrounding the white paper always reaches
    //   the image boundary, so this cleanly separates background clutter
    //   from the actual object placed on the paper.
    //
    //   We use a 2-pixel margin to handle minor edge effects:
    //     left   <= 2             (touches left edge)
    //     top    <= 2             (touches top edge)
    //     right  >= cols - 2      (touches right edge)
    //     bottom >= rows - 2      (touches bottom edge)
    //
    //   Example: cardboard corner in top-right
    //     left=600, top=0, width=40, height=80
    //     top=0 <= 2  ->  REJECTED (touches top boundary)
    //
    //   Example: T-tool in center of paper
    //     left=280, top=300, width=120, height=200
    //     None of the edges touch boundary  ->  ACCEPTED

    struct RegionInfo {
        int area;
        int originalLabel;
        double cx, cy;
    };

    std::vector<RegionInfo> validRegions;
    int boundaryMargin = 2; // pixels from edge to consider "touching"

    for (int i = 1; i < nLabels; i++) { // skip background (label 0)

        int area  = stats.at<int>(i, cv::CC_STAT_AREA);
        int left  = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top   = stats.at<int>(i, cv::CC_STAT_TOP);
        int w     = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h     = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        int right  = left + w;
        int bottom = top  + h;

        // Filter A: minimum size
        if (area < minRegionSize) continue;

        // Filter B: reject regions touching the image boundary
        // These are almost always cardboard/table edges, not the object
        bool touchesBoundary = (left   <= boundaryMargin          ||
                                top    <= boundaryMargin           ||
                                right  >= src.cols - boundaryMargin ||
                                bottom >= src.rows - boundaryMargin);

        if (touchesBoundary) continue;

        RegionInfo info;
        info.area          = area;
        info.originalLabel = i;
        info.cx            = centroids.at<double>(i, 0);
        info.cy            = centroids.at<double>(i, 1);
        validRegions.push_back(info);
    }

    // === Step 4: Sort by area (largest first) ===
    //
    // After boundary filtering, the remaining regions are actual objects
    // on the paper. Sorting by area puts the main object at label 1.

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

    std::map<int, int> labelMapping;
    for (int i = 0; i < numValid; i++) {
        labelMapping[validRegions[i].originalLabel] = i + 1;
    }

    regionMap = cv::Mat::zeros(labels.size(), CV_32SC1);

    for (int r = 0; r < labels.rows; r++) {
        const int *labelRow  = labels.ptr<int>(r);
        int       *regionRow = regionMap.ptr<int>(r);

        for (int c = 0; c < labels.cols; c++) {
            auto it = labelMapping.find(labelRow[c]);
            if (it != labelMapping.end()) {
                regionRow[c] = it->second;
            }
        }
    }

    // === Step 7: Generate colored visualization ===
    //
    // Fixed color palette (BGR format):
    //   1: red    (main object / largest non-boundary region)
    //   2: green  (second object)
    //   3: blue
    //   4: yellow
    //   5: magenta
    //   6: cyan
    //   7: orange

    std::vector<cv::Vec3b> palette = {
        cv::Vec3b(0,   0,   0  ),  // 0: background (black)
        cv::Vec3b(0,   0,   255),  // 1: red
        cv::Vec3b(0,   255, 0  ),  // 2: green
        cv::Vec3b(255, 0,   0  ),  // 3: blue
        cv::Vec3b(0,   255, 255),  // 4: yellow
        cv::Vec3b(255, 0,   255),  // 5: magenta
        cv::Vec3b(255, 255, 0  ),  // 6: cyan
        cv::Vec3b(0,   128, 255)   // 7: orange
    };

    colorMap = cv::Mat::zeros(regionMap.size(), CV_8UC3);

    for (int r = 0; r < regionMap.rows; r++) {
        const int   *regionRow = regionMap.ptr<int>(r);
        cv::Vec3b   *colorRow  = colorMap.ptr<cv::Vec3b>(r);

        for (int c = 0; c < regionMap.cols; c++) {
            int regionID = regionRow[c];
            if (regionID > 0 && regionID < static_cast<int>(palette.size())) {
                colorRow[c] = palette[regionID];
            } else if (regionID > 0) {
                colorRow[c] = palette[(regionID % (palette.size() - 1)) + 1];
            }
        }
    }

    return numValid;
}