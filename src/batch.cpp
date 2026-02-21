/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: src/batch.cpp
 *
 * Purpose:
 * Batch processing tool for generating report images.
 * Processes all images in a directory through the full pipeline
 * and saves outputs organized by task folder.
 *
 * No GUI, no keyboard input needed - fully automatic.
 *
 * Usage:
 *   ./objrec --batch dev_images/
 *
 * Output structure:
 *   data/task1_threshold/orig_<name>.png      - original image
 *   data/task1_threshold/thresh_<name>.png    - thresholded image
 *   data/task2_morphology/morph_<name>.png    - morphologically cleaned
 *   data/task3_segmentation/seg_<name>.png    - colored region map
 *   data/task4_features/feat_<name>.png       - features overlay image
 *   data/task4_features/features_<name>.txt   - feature vector values
 */

#include "batch.h"
#include "threshold.h"
#include "morphology.h"
#include "segment.h"
#include "features.h"

/**
 * extractFilename - Get just the name part from a full path
 *
 * @param path  Full file path (e.g., "dev_images/chisel.jpg")
 * @return      Name without path or extension (e.g., "chisel")
 *
 * Example:
 *   Input:  "dev_images/chisel.jpg"
 *   Step 1: find last '/' -> "chisel.jpg"
 *   Step 2: find last '.' -> "chisel"
 *   Output: "chisel"
 */
std::string extractFilename(const std::string &path) {
    // Find last slash to remove directory path
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash != std::string::npos)
                            ? path.substr(lastSlash + 1)
                            : path;

    // Remove extension
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        filename = filename.substr(0, lastDot);
    }

    return filename;
}

/**
 * runBatchProcessing - Process all images in a directory
 *
 * @param dirPath  Path to directory containing images
 * @return         0 on success, -1 on error
 *
 * For each image:
 *   1. Load the image
 *   2. Pre-process (Gaussian blur)
 *   3. Threshold (ISODATA from scratch)
 *   4. Morphological cleanup
 *   5. Connected components segmentation
 *   6. Feature extraction (moments, oriented bbox, Hu moments)
 *   7. Save all outputs to task-specific folders
 */
int runBatchProcessing(const std::string &dirPath) {

    // === Step 1: Create output directories ===

    system("mkdir -p data/task1_threshold data/task2_morphology data/task3_segmentation");
    system("mkdir -p data/task4_features data/task5_training data/task6_classification");
    system("mkdir -p data/task7_evaluation data/task8_embeddings");

    // === Step 2: Get list of image files ===

    std::vector<cv::String> cvFiles;
    std::vector<std::string> imageFiles;

    // Search for .jpg files
    cv::glob(dirPath + "/*.jpg", cvFiles, false);
    for (auto &f : cvFiles) imageFiles.push_back(f);

    // Search for .jpeg files
    cvFiles.clear();
    cv::glob(dirPath + "/*.jpeg", cvFiles, false);
    for (auto &f : cvFiles) imageFiles.push_back(f);

    // Search for .png files
    cvFiles.clear();
    cv::glob(dirPath + "/*.png", cvFiles, false);
    for (auto &f : cvFiles) imageFiles.push_back(f);

    std::sort(imageFiles.begin(), imageFiles.end());

    if (imageFiles.empty()) {
        std::cerr << "Error: No image files found in " << dirPath << std::endl;
        return -1;
    }

    std::cout << "=== Batch Processing ===" << std::endl;
    std::cout << "Found " << imageFiles.size() << " images in " << dirPath << std::endl;
    std::cout << std::endl;

    // === Step 3: Process each image ===

    int processed = 0;

    for (const auto &imgPath : imageFiles) {
        std::string name = extractFilename(imgPath);
        std::cout << "Processing: " << name << std::endl;

        // Load the image
        cv::Mat frame = cv::imread(imgPath);
        if (frame.empty()) {
            std::cerr << "  Warning: Cannot read " << imgPath << ", skipping." << std::endl;
            continue;
        }

        // Pre-process: Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        // Task 1: Threshold (FROM SCRATCH)
        cv::Mat threshed;
        int threshVal = dynamicThreshold(blurred, threshed);
        std::cout << "  Threshold value: " << threshVal << std::endl;

        // Task 2: Morphological cleanup
        cv::Mat cleaned;
        cleanBinary(threshed, cleaned);

        // Task 3: Connected components segmentation
        cv::Mat regionMap, colorMap;
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);
        std::cout << "  Regions found: " << nRegions << std::endl;

        // === Save outputs for Tasks 1-3 ===

        // Task 1: Original + Threshold
        cv::imwrite("data/task1_threshold/orig_" + name + ".png", frame);
        cv::imwrite("data/task1_threshold/thresh_" + name + ".png", threshed);

        // Task 2: Morphology
        cv::imwrite("data/task2_morphology/morph_" + name + ".png", cleaned);

        // Task 3: Segmentation
        if (nRegions > 0) {
            cv::imwrite("data/task3_segmentation/seg_" + name + ".png", colorMap);
        }

        // === Task 4: Feature extraction ===
        //
        // Compute features for the largest region (label 1).
        // Save the features overlay image and feature vector text file.

        if (nRegions > 0) {
            RegionProps mainRegion;

            if (computeRegionProps(regionMap, 1, mainRegion) == 0) {
                // Save features overlay image
                // Shows oriented bounding box (green), primary axis (red),
                // centroid (blue dot), and feature values
                cv::Mat featDisplay = frame.clone();
                drawFeatures(featDisplay, mainRegion);
                cv::imwrite("data/task4_features/feat_" + name + ".png", featDisplay);

                // Build feature vector
                std::vector<float> featureVec;
                buildFeatureVector(mainRegion, featureVec);

                // Save feature vector to text file
                // This makes it easy to include in the report
                std::ofstream fout("data/task4_features/features_" + name + ".txt");
                fout << "Object: " << name << std::endl;
                fout << "Area: " << mainRegion.area << " pixels" << std::endl;
                fout << "Centroid: (" << mainRegion.cx << ", " << mainRegion.cy << ")" << std::endl;
                fout << "Orientation: " << mainRegion.theta << " rad ("
                     << mainRegion.theta * 180.0f / M_PI << " deg)" << std::endl;
                fout << std::endl;
                fout << "Feature vector:" << std::endl;
                fout << "  percentFilled:  " << featureVec[0] << std::endl;
                fout << "  aspectRatio:    " << featureVec[1] << std::endl;
                for (int i = 0; i < 7; i++) {
                    fout << "  huMoment[" << i << "]:   " << featureVec[i + 2] << std::endl;
                }
                fout.close();

                // Print summary to console
                std::cout << "  Features: fill=" << featureVec[0]
                          << " ar=" << featureVec[1]
                          << " hu0=" << featureVec[2]
                          << " hu1=" << featureVec[3] << std::endl;
            }
        }

        processed++;
        std::cout << "  Saved." << std::endl;
    }

    // === Summary ===

    std::cout << std::endl;
    std::cout << "=== Batch Processing Complete ===" << std::endl;
    std::cout << "Processed " << processed << "/" << imageFiles.size() << " images" << std::endl;
    std::cout << std::endl;
    std::cout << "Output locations:" << std::endl;
    std::cout << "  Task 1 (Threshold):     data/task1_threshold/" << std::endl;
    std::cout << "  Task 2 (Morphology):    data/task2_morphology/" << std::endl;
    std::cout << "  Task 3 (Segmentation):  data/task3_segmentation/" << std::endl;
    std::cout << "  Task 4 (Features):      data/task4_features/" << std::endl;

    return 0;
}