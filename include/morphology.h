/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: include/morphology.h
 *
 * Purpose:
 * Header for morphological filtering functions.
 * Cleans up binary images by filling holes and removing noise.
 *
 * Strategy: Closing followed by Opening
 *   - Closing (dilate then erode): fills small holes inside objects
 *   - Opening (erode then dilate): removes small noise specks
 */

#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include "vision.h"

/*
 * cleanBinary
 *
 * Clean up a binary image using morphological operations.
 * Performs closing (fill holes) then opening (remove noise).
 *
 * @param src        Input binary image (8UC1, 0 or 255)
 * @param dst        Output cleaned binary image (8UC1)
 * @param kernelSize Size of structuring element (default 5)
 * @return           0 on success, -1 on error
 */
int cleanBinary(const cv::Mat &src, cv::Mat &dst, int kernelSize = 5);

#endif // MORPHOLOGY_H