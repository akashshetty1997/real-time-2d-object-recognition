/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: include/segment.h
 *
 * Purpose:
 * Header for connected components segmentation functions.
 * Takes a cleaned binary image and produces labeled regions
 * with a colored visualization.
 *
 * Uses OpenCV's connectedComponentsWithStats to find regions,
 * then filters by size and relabels (1 = largest region).
 */

#ifndef SEGMENT_H
#define SEGMENT_H

#include "vision.h"

/*
 * segmentRegions
 *
 * Run connected components analysis on a binary image.
 * Filters out small regions, keeps the largest N, and produces
 * both a labeled region map and a colored visualization.
 *
 * @param src           Input binary image (8UC1, 255 = foreground)
 * @param regionMap     Output labeled region map (32SC1), relabeled 1..N
 * @param colorMap      Output colored visualization (8UC3)
 * @param minRegionSize Ignore regions smaller than this (default 500)
 * @param maxRegions    Max number of regions to keep (default 5)
 * @return              Number of valid regions found
 */
int segmentRegions(const cv::Mat &src, cv::Mat &regionMap, cv::Mat &colorMap,
                   int minRegionSize = 500, int maxRegions = 5);

#endif // SEGMENT_H