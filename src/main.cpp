/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: src/main.cpp
 *
 * Purpose:
 * Main entry point for the real-time 2D object recognition system.
 *
 * Current pipeline:
 *   Step 1: Camera capture or image load
 *   Step 2: Pre-process (Gaussian blur)
 *   Step 3: Threshold (ISODATA - from scratch)
 *   Step 4: Morphological cleanup (closing + opening)
 *   Step 5: Connected components segmentation
 *   Step 6: Feature extraction (moments, oriented bbox, Hu moments)
 *   Step 7: Classification (nearest-neighbor, scaled Euclidean)
 *
 * Key controls:
 *   q / ESC  - quit
 *   t        - toggle threshold view
 *   m        - toggle morphology view
 *   s        - toggle segmentation view
 *   f        - toggle features overlay view
 *   c        - toggle classification mode
 *   l        - TRAINING: label current object and save to DB
 *   p        - save screenshots
 *   n        - next image (directory mode)
 *   b        - previous image (directory mode)
 *
 * Usage:
 *   ./objrec                          # live camera
 *   ./objrec dev_images/chisel.jpg    # single image
 *   ./objrec dev_images/              # directory of images
 *   ./objrec --batch dev_images/      # batch process for report
 */

#include <sys/stat.h>
#include "vision.h"
#include "threshold.h"
#include "morphology.h"
#include "segment.h"
#include "features.h"
#include "classify.h"
#include "batch.h"

const std::string TRAINING_DB_PATH = "data/task5_training/training_db.csv";

/**
 * isDirectory - Check if a path is a directory
 */
bool isDirectory(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

/**
 * getImageFiles - Get sorted list of image files from a directory
 */
std::vector<std::string> getImageFiles(const std::string &dirPath) {
    std::vector<std::string> files;
    std::vector<cv::String> cvFiles;

    cv::glob(dirPath + "/*.jpg", cvFiles, false);
    for (auto &f : cvFiles) files.push_back(f);

    cvFiles.clear();
    cv::glob(dirPath + "/*.jpeg", cvFiles, false);
    for (auto &f : cvFiles) files.push_back(f);

    cvFiles.clear();
    cv::glob(dirPath + "/*.png", cvFiles, false);
    for (auto &f : cvFiles) files.push_back(f);

    std::sort(files.begin(), files.end());
    return files;
}

int main(int argc, char *argv[]) {

    // === Check for batch mode ===
    if (argc > 2 && std::string(argv[1]) == "--batch") {
        return runBatchProcessing(argv[2]);
    }

    // === Step 1: Determine input mode ===

    bool liveMode = true;
    bool dirMode = false;
    cv::VideoCapture cap;
    cv::Mat staticImage;
    std::vector<std::string> imageFiles;
    int imageIndex = 0;

    if (argc > 1) {
        std::string inputPath = argv[1];

        if (isDirectory(inputPath)) {
            dirMode = true;
            liveMode = false;
            imageFiles = getImageFiles(inputPath);

            if (imageFiles.empty()) {
                std::cerr << "Error: No image files found in " << inputPath << std::endl;
                return -1;
            }

            std::cout << "Directory mode: found " << imageFiles.size() << " images" << std::endl;
            staticImage = cv::imread(imageFiles[0]);

            if (staticImage.empty()) {
                std::cerr << "Error: Cannot read " << imageFiles[0] << std::endl;
                return -1;
            }

            std::cout << "Loaded: " << imageFiles[0] << std::endl;

        } else {
            liveMode = false;
            staticImage = cv::imread(inputPath);

            if (staticImage.empty()) {
                std::cerr << "Error: Cannot read image " << inputPath << std::endl;
                return -1;
            }

            std::cout << "Single image mode: " << inputPath << std::endl;
        }
    } else {
        cap.open(0);

        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera." << std::endl;
            return -1;
        }

        std::cout << "Live camera mode" << std::endl;
    }

    // === Create output directories ===

    system("mkdir -p data/task1_threshold data/task2_morphology data/task3_segmentation");
    system("mkdir -p data/task4_features data/task5_training data/task6_classification");
    system("mkdir -p data/task7_evaluation data/task8_embeddings");

    // === Load training database ===
    //
    // The training DB is a CSV file that persists between runs.
    // If no DB exists yet, we start fresh (empty).

    std::vector<TrainingEntry> trainingDB;
    loadTrainingDB(TRAINING_DB_PATH, trainingDB);

    // === Step 2: Initialize state ===

    cv::Mat frame, blurred, threshed, cleaned, regionMap, colorMap;
    bool showThresh = false;
    bool showMorph = false;
    bool showSegment = false;
    bool showFeatures = false;
    bool classifyOn = false;
    int saveCount = 0;

    std::cout << "=== Real-time 2D Object Recognition ===" << std::endl;
    std::cout << "Keys: t=threshold, m=morphology, s=segment, f=features" << std::endl;
    std::cout << "      c=classify, l=label/train, p=save, q=quit" << std::endl;
    if (dirMode) {
        std::cout << "      n=next image, b=previous image" << std::endl;
    }

    // === Step 3: Main loop ===

    while (true) {
        // --- Get frame based on mode ---
        if (liveMode) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Empty frame received." << std::endl;
                break;
            }
        } else {
            frame = staticImage.clone();
        }

        // --- Pre-process: Gaussian blur ---
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        // --- Task 1: Threshold (FROM SCRATCH) ---
        int threshVal = dynamicThreshold(blurred, threshed);

        // --- Task 2: Morphological cleanup ---
        cleanBinary(threshed, cleaned);

        // --- Task 3: Connected components segmentation ---
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);

        // --- Task 4: Compute features for largest region ---
        RegionProps mainRegion;
        std::vector<float> featureVec;
        bool hasRegion = false;

        if (nRegions > 0) {
            if (computeRegionProps(regionMap, 1, mainRegion) == 0) {
                buildFeatureVector(mainRegion, featureVec);
                hasRegion = true;
            }
        }

        // --- Task 6: Classification ---
        std::string classLabel = "";
        float classDist = 0.0f;

        if (classifyOn && hasRegion && !trainingDB.empty()) {
            classifyNN(featureVec, trainingDB, classLabel, classDist);
        }

        // --- Display ---

        cv::Mat display = frame.clone();

        // Draw features overlay if enabled
        if (showFeatures && hasRegion) {
            drawFeatures(display, mainRegion);
        }

        // Draw classification result if enabled
        if (classifyOn && hasRegion && !classLabel.empty()) {
            // Draw label on the object
            drawFeatures(display, mainRegion, classLabel);

            // Show classification info at top of screen
            cv::putText(display, "Class: " + classLabel,
                        cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 255), 2);
            cv::putText(display, "Dist: " + std::to_string(classDist).substr(0, 5),
                        cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 200, 200), 2);
        }

        // Info overlay
        cv::putText(display, "Thresh: " + std::to_string(threshVal),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Regions: " + std::to_string(nRegions),
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "DB: " + std::to_string(trainingDB.size()) + " entries",
                    cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 0), 2);

        if (dirMode) {
            std::string filename = imageFiles[imageIndex];
            size_t lastSlash = filename.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                filename = filename.substr(lastSlash + 1);
            }

            cv::putText(display, "Image " + std::to_string(imageIndex + 1) + "/" +
                        std::to_string(imageFiles.size()) + ": " + filename,
                        cv::Point(10, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Video", display);

        // Toggle views
        if (showThresh) {
            cv::imshow("Threshold", threshed);
        }
        if (showMorph) {
            cv::imshow("Morphology", cleaned);
        }
        if (showSegment && nRegions > 0) {
            cv::imshow("Segments", colorMap);
        }

        // --- Key handling ---
        int waitTime = liveMode ? 10 : 50;
        int key = cv::waitKey(waitTime) & 0xFF;

        if (key == 'q' || key == 27) {
            std::cout << "Exiting..." << std::endl;
            break;
        }

        switch (key) {
            case 't':
                showThresh = !showThresh;
                std::cout << "Threshold view: " << (showThresh ? "ON" : "OFF") << std::endl;
                if (!showThresh) cv::destroyWindow("Threshold");
                break;

            case 'm':
                showMorph = !showMorph;
                std::cout << "Morphology view: " << (showMorph ? "ON" : "OFF") << std::endl;
                if (!showMorph) cv::destroyWindow("Morphology");
                break;

            case 's':
                showSegment = !showSegment;
                std::cout << "Segment view: " << (showSegment ? "ON" : "OFF") << std::endl;
                if (!showSegment) cv::destroyWindow("Segments");
                break;

            case 'f':
                showFeatures = !showFeatures;
                std::cout << "Features view: " << (showFeatures ? "ON" : "OFF") << std::endl;
                break;

            case 'c':
                classifyOn = !classifyOn;
                std::cout << "Classification: " << (classifyOn ? "ON" : "OFF") << std::endl;
                if (classifyOn && trainingDB.empty()) {
                    std::cout << "  Warning: Training DB is empty. Use 'l' to label objects first." << std::endl;
                }
                break;

            case 'l': {
                // === Task 5: Training mode ===
                // Label the current object and save its features to the DB
                if (!hasRegion) {
                    std::cout << "No region detected. Place an object first." << std::endl;
                    break;
                }

                std::cout << "Enter label for current object: ";
                std::string label;
                std::getline(std::cin, label);

                if (!label.empty()) {
                    addTrainingEntry(trainingDB, label, featureVec);
                    saveTrainingDB(TRAINING_DB_PATH, trainingDB);
                    std::cout << "Saved '" << label << "' to training DB. ("
                              << trainingDB.size() << " entries total)" << std::endl;

                    // Print the feature vector for reference
                    std::cout << "  Features: fill=" << featureVec[0]
                              << " ar=" << featureVec[1]
                              << " hu0=" << featureVec[2]
                              << " hu1=" << featureVec[3] << std::endl;
                } else {
                    std::cout << "Empty label, skipping." << std::endl;
                }
                break;
            }

            case 'n':
                if (dirMode && imageIndex < static_cast<int>(imageFiles.size()) - 1) {
                    imageIndex++;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                    std::cout << "Loaded: " << imageFiles[imageIndex] << std::endl;
                } else if (dirMode) {
                    std::cout << "Already at last image." << std::endl;
                }
                break;

            case 'b':
                if (dirMode && imageIndex > 0) {
                    imageIndex--;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                    std::cout << "Loaded: " << imageFiles[imageIndex] << std::endl;
                } else if (dirMode) {
                    std::cout << "Already at first image." << std::endl;
                }
                break;

            case 'p': {
                // Task 1: Original + Threshold
                std::string t1Orig = "data/task1_threshold/orig_" + std::to_string(saveCount) + ".png";
                std::string t1Thresh = "data/task1_threshold/thresh_" + std::to_string(saveCount) + ".png";
                cv::imwrite(t1Orig, frame);
                cv::imwrite(t1Thresh, threshed);

                // Task 2: Morphology
                std::string t2Morph = "data/task2_morphology/morph_" + std::to_string(saveCount) + ".png";
                cv::imwrite(t2Morph, cleaned);

                // Task 3: Segmentation
                if (nRegions > 0) {
                    std::string t3Seg = "data/task3_segmentation/seg_" + std::to_string(saveCount) + ".png";
                    cv::imwrite(t3Seg, colorMap);
                }

                // Task 4: Features overlay
                if (hasRegion) {
                    cv::Mat featDisplay = frame.clone();
                    drawFeatures(featDisplay, mainRegion);
                    std::string t4Feat = "data/task4_features/feat_" + std::to_string(saveCount) + ".png";
                    cv::imwrite(t4Feat, featDisplay);

                    std::ofstream fout("data/task4_features/features_" + std::to_string(saveCount) + ".txt");
                    fout << "percentFilled: " << featureVec[0] << std::endl;
                    fout << "aspectRatio: " << featureVec[1] << std::endl;
                    for (int i = 0; i < 7; i++) {
                        fout << "huMoment[" << i << "]: " << featureVec[i + 2] << std::endl;
                    }
                    fout.close();
                }

                // Task 6: Classification result
                if (classifyOn && hasRegion && !classLabel.empty()) {
                    cv::Mat classDisplay = frame.clone();
                    drawFeatures(classDisplay, mainRegion, classLabel);
                    std::string t6Class = "data/task6_classification/class_" + std::to_string(saveCount) + ".png";
                    cv::imwrite(t6Class, classDisplay);
                }

                std::cout << "Saved images #" << saveCount << std::endl;
                saveCount++;
                break;
            }
        }
    }

    // === Step 4: Cleanup ===
    if (liveMode) cap.release();
    cv::destroyAllWindows();

    return 0;
}