/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: src/morphology.cpp
 *
 * Purpose:
 * Implementation of morphological filtering to clean up binary images.
 * Uses OpenCV morphological operations (erode, dilate).
 *
 * Strategy:
 * After thresholding, binary images typically have two problems:
 *   1. Small noise specks (false positives in the background)
 *   2. Small holes inside the object (false negatives in the foreground)
 *
 * Solution: Closing followed by Opening
 *
 *   Closing (dilate then erode):
 *     - Fills small holes and gaps inside the object
 *     - Bridges narrow breaks in the foreground
 *     - The dilation grows the white regions, filling holes
 *     - The erosion shrinks them back to roughly original size
 *
 *   Opening (erode then dilate):
 *     - Removes small noise specks in the background
 *     - The erosion removes tiny white regions (noise)
 *     - The dilation restores the remaining regions to original size
 *
 * Order matters:
 *   Closing FIRST -> fills holes before we remove noise
 *   Opening SECOND -> removes noise without re-opening filled holes
 *
 * Example with phone on white wall:
 *   Before cleanup:
 *     - Phone has small black holes (camera lens, card slot)
 *     - A few white noise specks near the edges
 *   After closing:
 *     - Phone interior is more solid white
 *   After opening:
 *     - Noise specks removed, phone shape is clean
 */

#include "morphology.h"

/**
 * cleanBinary - Clean up a binary image using morphological operations
 *
 * @param src        Input binary image (8UC1, 0 or 255)
 * @param dst        Output cleaned binary image (8UC1)
 * @param kernelSize Size of the structuring element (default 5)
 * @return           0 on success, -1 on error
 *
 * Implementation details:
 *
 * Step-by-step:
 *   1. Validate input image
 *   2. Create a cross-shaped structuring element
 *   3. Apply closing: dilate then erode (fills holes)
 *   4. Apply opening: erode then dilate (removes noise)
 *
 * Why cross-shaped kernel:
 *   - Less aggressive than a full square kernel
 *   - Preserves corners and thin features better
 *   - Still effective at filling small holes and removing noise
 *   - Good balance for objects with both thick and thin parts
 *     (e.g., pen cap vs pen body, key teeth vs key handle)
 *
 * Kernel size choice (default 5):
 *   - Size 3: too small, doesn't fill holes well
 *   - Size 5: good balance for 640x480 images
 *   - Size 7+: too aggressive, can merge nearby objects
 *
 * Example with kernel size 5:
 *   Cross kernel:     0 0 1 0 0
 *                     0 0 1 0 0
 *                     1 1 1 1 1
 *                     0 0 1 0 0
 *                     0 0 1 0 0
 */
int cleanBinary(const cv::Mat &src, cv::Mat &dst, int kernelSize) {

    // === Step 1: Validate input ===

    if (src.empty()) {
        std::cerr << "Error: Input image is empty in cleanBinary()" << std::endl;
        return -1;
    }

    if (src.type() != CV_8UC1) {
        std::cerr << "Error: Input must be single-channel 8-bit in cleanBinary()" << std::endl;
        return -1;
    }

    // === Step 2: Create structuring element ===
    //
    // MORPH_CROSS creates a cross-shaped kernel:
    //   For size 5: a "+" shape that is 5 pixels wide and 5 pixels tall
    //
    // Alternative shapes:
    //   MORPH_RECT: full square (more aggressive)
    //   MORPH_ELLIPSE: oval shape (good for round objects)

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_CROSS,
        cv::Size(kernelSize, kernelSize)
    );

    // === Step 3: Closing (dilate then erode) ===
    //
    // Purpose: Fill small holes inside the object
    //
    // How it works:
    //   Dilate: each white pixel expands outward by kernel radius
    //           -> small black holes get swallowed by surrounding white
    //   Erode:  each white pixel shrinks inward by kernel radius
    //           -> object returns to roughly original size
    //           -> but filled holes stay filled (they were already white)
    //
    // Example: a 3x3 hole inside a large white region
    //   Before: ...111101111...  (hole at position 6)
    //   Dilate: ...111111111...  (hole filled)
    //   Erode:  ...111111111...  (hole stays filled, edges shrink back)

    cv::Mat closed;
    cv::dilate(src, closed, kernel);
    cv::erode(closed, closed, kernel);

    // === Step 4: Opening (erode then dilate) ===
    //
    // Purpose: Remove small noise specks in the background
    //
    // How it works:
    //   Erode:  each white pixel shrinks inward by kernel radius
    //           -> tiny white specks disappear entirely
    //           -> large objects just get slightly smaller
    //   Dilate: each white pixel expands outward by kernel radius
    //           -> large objects return to original size
    //           -> removed specks stay removed (nothing to expand)
    //
    // Example: a 2x2 white noise speck
    //   Before: ...000011000...  (noise at positions 5-6)
    //   Erode:  ...000000000...  (noise removed, too small to survive)
    //   Dilate: ...000000000...  (noise stays removed)

    cv::erode(closed, dst, kernel);
    cv::dilate(dst, dst, kernel);

    return 0;
}