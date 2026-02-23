/*
 * Name: Akash Shridhar Shetty , Skandhan Madhusudhana
 * Date: February 2025
 * File: src/main.cpp
 *
 * Usage:
 *   ./objrec                              # live camera
 *   ./objrec dev_images/chisel.jpg        # single image
 *   ./objrec dev_images/                  # directory of images
 *   ./objrec --batch dev_images/          # batch process for report
 *   ./objrec --evaluate eval.csv          # confusion matrix evaluation
 */

#include <sys/stat.h>
#include "vision.h"
#include "threshold.h"
#include "morphology.h"
#include "segment.h"
#include "features.h"
#include "classify.h"
#include "batch.h"
#include "embeddings.h"

// Professor's utility functions (defined in src/utilities.cpp)
extern int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
extern void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage,
                                int cx, int cy, float theta,
                                float minE1, float maxE1,
                                float minE2, float maxE2, int debug);

const std::string TRAINING_DB_PATH = "data/task5_training/training_db.csv";
const std::string MODEL_PATH       = "models/resnet18-v2-7.onnx";

bool isDirectory(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

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

// =============================================================
// runEvaluation - Task 7: Confusion Matrix
//
// Reads a CSV file with columns: image_path, true_label
// Runs the full pipeline on each image, classifies it,
// then builds and saves a confusion matrix.
//
// Usage:
//   ./objrec --evaluate data/task7_evaluation/eval_images.csv
//
// Output:
//   - Confusion matrix printed to console
//   - Saved to data/task7_evaluation/confusion_matrix.txt
//   - Visual confusion matrix image saved to
//     data/task7_evaluation/confusion_matrix.png
// =============================================================
int runEvaluation(const std::string &csvPath) {

    std::cout << "=== Task 7: Confusion Matrix Evaluation ===" << std::endl;
    std::cout << "Loading eval CSV: " << csvPath << std::endl;

    // --- Load training DB ---
    std::vector<TrainingEntry> trainingDB;
    if (loadTrainingDB(TRAINING_DB_PATH, trainingDB) != 0 || trainingDB.empty()) {
        std::cerr << "Error: Training DB is empty. Train objects first." << std::endl;
        return -1;
    }
    std::cout << "Training DB: " << trainingDB.size() << " entries" << std::endl;

    // --- Parse eval CSV ---
    std::ifstream fin(csvPath);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open " << csvPath << std::endl;
        return -1;
    }

    struct EvalEntry {
        std::string imagePath;
        std::string trueLabel;
    };

    std::vector<EvalEntry> evalSet;
    std::string line;
    bool firstLine = true;

    while (std::getline(fin, line)) {
        if (firstLine) { firstLine = false; continue; } // skip header
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        EvalEntry e;
        if (std::getline(ss, e.imagePath, ',') &&
            std::getline(ss, e.trueLabel, ',')) {
            // trim whitespace/carriage returns
            while (!e.trueLabel.empty() &&
                   (e.trueLabel.back() == '\r' || e.trueLabel.back() == ' '))
                e.trueLabel.pop_back();
            evalSet.push_back(e);
        }
    }
    fin.close();
    std::cout << "Eval set: " << evalSet.size() << " images" << std::endl;

    // --- Collect all unique labels (sorted) ---
    std::set<std::string> labelSet;
    for (const auto &e : evalSet) labelSet.insert(e.trueLabel);

    std::vector<std::string> labels(labelSet.begin(), labelSet.end());
    int N = static_cast<int>(labels.size());

    // Map label -> index
    std::map<std::string, int> labelIdx;
    for (int i = 0; i < N; i++) labelIdx[labels[i]] = i;

    std::cout << "Classes (" << N << "): ";
    for (const auto &l : labels) std::cout << l << "  ";
    std::cout << std::endl << std::endl;

    // --- Initialize confusion matrix (rows=true, cols=predicted) ---
    std::vector<std::vector<int>> confMatrix(N, std::vector<int>(N, 0));

    int correct = 0;
    int total = 0;
    int noDetection = 0;

    // --- Process each image ---
    for (const auto &e : evalSet) {
        std::cout << "Processing: " << e.imagePath
                  << " (true: " << e.trueLabel << ") -> ";

        cv::Mat frame = cv::imread(e.imagePath);
        if (frame.empty()) {
            std::cerr << "Cannot read image, skipping." << std::endl;
            noDetection++;
            continue;
        }

        // Full pipeline
        cv::Mat blurred, threshed, cleaned, regionMap, colorMap;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        dynamicThreshold(blurred, threshed);
        cleanBinary(threshed, cleaned);
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);

        if (nRegions == 0) {
            std::cout << "NO DETECTION" << std::endl;
            noDetection++;
            total++;
            // Count as wrong prediction - use first label as placeholder
            if (labelIdx.count(e.trueLabel)) {
                // no detection = no prediction, skip matrix entry
            }
            continue;
        }

        // Extract features from largest region
        RegionProps props;
        std::vector<float> featureVec;
        if (computeRegionProps(regionMap, 1, props) != 0) {
            std::cout << "FEATURE ERROR" << std::endl;
            noDetection++;
            total++;
            continue;
        }
        buildFeatureVector(props, featureVec);

        // Classify
        std::string predicted;
        float dist;
        classifyNN(featureVec, trainingDB, predicted, dist);

        std::cout << "predicted: " << predicted
                  << " (dist: " << dist << ")";

        // Update confusion matrix
        if (labelIdx.count(e.trueLabel) && labelIdx.count(predicted)) {
            int trueIdx = labelIdx[e.trueLabel];
            int predIdx = labelIdx[predicted];
            confMatrix[trueIdx][predIdx]++;

            if (e.trueLabel == predicted) {
                correct++;
                std::cout << " ✓";
            } else {
                std::cout << " ✗";
            }
        }
        total++;
        std::cout << std::endl;
    }

    // --- Print confusion matrix ---
    std::cout << std::endl;
    std::cout << "=== CONFUSION MATRIX ===" << std::endl;
    std::cout << "Rows = True Label, Cols = Predicted Label" << std::endl;
    std::cout << std::endl;

    // Column width for alignment
    int colW = 14;

    // Header row
    std::cout << std::setw(colW) << "TRUE\\PRED";
    for (const auto &l : labels)
        std::cout << std::setw(colW) << l;
    std::cout << std::endl;

    // Separator
    std::cout << std::string(colW * (N + 1), '-') << std::endl;

    // Data rows
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(colW) << labels[i];
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(colW) << confMatrix[i][j];
        }
        std::cout << std::endl;
    }

    // Accuracy
    std::cout << std::endl;
    float accuracy = (total > 0) ? (100.0f * correct / total) : 0.0f;
    std::cout << "Correct: " << correct << "/" << total
              << "  Accuracy: " << std::fixed << std::setprecision(1)
              << accuracy << "%" << std::endl;
    if (noDetection > 0)
        std::cout << "No detection: " << noDetection << " images" << std::endl;

    // --- Save to text file ---
    system("mkdir -p data/task7_evaluation");
    std::ofstream fout("data/task7_evaluation/confusion_matrix.txt");
    fout << "=== CONFUSION MATRIX ===" << std::endl;
    fout << "Rows = True Label, Cols = Predicted Label" << std::endl << std::endl;

    fout << std::setw(colW) << "TRUE\\PRED";
    for (const auto &l : labels) fout << std::setw(colW) << l;
    fout << std::endl;
    fout << std::string(colW * (N + 1), '-') << std::endl;

    for (int i = 0; i < N; i++) {
        fout << std::setw(colW) << labels[i];
        for (int j = 0; j < N; j++)
            fout << std::setw(colW) << confMatrix[i][j];
        fout << std::endl;
    }
    fout << std::endl;
    fout << "Correct: " << correct << "/" << total
         << "  Accuracy: " << std::fixed << std::setprecision(1)
         << accuracy << "%" << std::endl;
    fout.close();
    std::cout << std::endl << "Saved: data/task7_evaluation/confusion_matrix.txt" << std::endl;

    // --- Save visual confusion matrix image ---
    int cellSize = 120;
    int imgSize  = cellSize * (N + 1);
    cv::Mat cmImg(imgSize, imgSize, CV_8UC3, cv::Scalar(255, 255, 255));

    // Find max value for color scaling
    int maxVal = 1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            maxVal = std::max(maxVal, confMatrix[i][j]);

    // Draw cells
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int x = (j + 1) * cellSize;
            int y = (i + 1) * cellSize;
            int val = confMatrix[i][j];

            // Color: diagonal = green shades, off-diagonal = red shades
            cv::Scalar cellColor;
            if (i == j) {
                // Correct predictions: white->green
                int g = 255 - static_cast<int>(200.0f * val / maxVal);
                cellColor = cv::Scalar(g, 255, g);
            } else if (val > 0) {
                // Misclassifications: white->red
                int r = 255 - static_cast<int>(200.0f * val / maxVal);
                cellColor = cv::Scalar(r, r, 255);
            } else {
                cellColor = cv::Scalar(245, 245, 245);
            }

            cv::rectangle(cmImg,
                cv::Point(x, y),
                cv::Point(x + cellSize, y + cellSize),
                cellColor, cv::FILLED);
            cv::rectangle(cmImg,
                cv::Point(x, y),
                cv::Point(x + cellSize, y + cellSize),
                cv::Scalar(180, 180, 180), 1);

            // Draw count
            if (val > 0) {
                std::string txt = std::to_string(val);
                cv::putText(cmImg, txt,
                    cv::Point(x + cellSize/2 - 10, y + cellSize/2 + 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 0), 2);
            }
        }
    }

    // Draw labels (shortened to fit)
    for (int i = 0; i < N; i++) {
        std::string shortLabel = labels[i].substr(0, 8);

        // Row labels (true)
        cv::putText(cmImg, shortLabel,
            cv::Point(2, (i + 1) * cellSize + cellSize/2 + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 0), 1);

        // Column labels (predicted) - rotated via vertical text
        cv::putText(cmImg, shortLabel,
            cv::Point((i + 1) * cellSize + 5, cellSize - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 0), 1);
    }

    // Title
    cv::putText(cmImg, "Confusion Matrix",
        cv::Point(cellSize, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(0, 0, 150), 2);

    // Accuracy text
    std::string accStr = "Accuracy: " + std::to_string(static_cast<int>(accuracy)) + "%";
    cv::putText(cmImg, accStr,
        cv::Point(cellSize, imgSize - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(0, 100, 0), 2);

    cv::imwrite("data/task7_evaluation/confusion_matrix.png", cmImg);
    std::cout << "Saved: data/task7_evaluation/confusion_matrix.png" << std::endl;

    return 0;
}

// =============================================================
// main
// =============================================================
int main(int argc, char *argv[]) {

    // --- Batch mode ---
    if (argc > 2 && std::string(argv[1]) == "--batch") {
        return runBatchProcessing(argv[2]);
    }

    // --- Evaluate mode (Task 7) ---
    if (argc > 2 && std::string(argv[1]) == "--evaluate") {
        return runEvaluation(argv[2]);
    }

    // --- Embedding train mode (Task 8) ---
    if (argc > 2 && std::string(argv[1]) == "--embed-train") {
        return runEmbedTraining(argv[2], MODEL_PATH);
    }

    // --- Embedding evaluate mode (Task 8) ---
    if (argc > 2 && std::string(argv[1]) == "--embed-evaluate") {
        return runEmbedEvaluate(argv[2], MODEL_PATH);
    }

    // --- Determine input mode ---
    bool liveMode = true;
    bool dirMode  = false;
    cv::VideoCapture cap;
    cv::Mat staticImage;
    std::vector<std::string> imageFiles;
    int imageIndex = 0;

    if (argc > 1) {
        std::string inputPath = argv[1];

        if (isDirectory(inputPath)) {
            dirMode   = true;
            liveMode  = false;
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
            liveMode    = false;
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

    // --- Create output directories ---
    system("mkdir -p data/task1_threshold data/task2_morphology data/task3_segmentation");
    system("mkdir -p data/task4_features data/task5_training data/task6_classification");
    system("mkdir -p data/task7_evaluation data/task8_embeddings");

    // --- Load training database ---
    std::vector<TrainingEntry> trainingDB;
    loadTrainingDB(TRAINING_DB_PATH, trainingDB);

    // --- Load embedding database (Task 8) ---
    std::vector<EmbeddingEntry> embDB;
    loadEmbeddingDB("data/task8_embeddings/embedding_db.csv", embDB);

    // --- Load ResNet18 model (Task 8) ---
    cv::dnn::Net net;
    bool modelLoaded = false;
    if (!embDB.empty()) {
        try {
            net = cv::dnn::readNet(MODEL_PATH);
            modelLoaded = true;
            std::cout << "ResNet18 model loaded for embedding classification." << std::endl;
        } catch (...) {
            std::cout << "Warning: Could not load ResNet18 model." << std::endl;
        }
    }

    // --- Initialize state ---
    cv::Mat frame, blurred, threshed, cleaned, regionMap, colorMap;
    bool showThresh   = false;
    bool showMorph    = false;
    bool showSegment  = false;
    bool showFeatures = false;
    bool classifyOn   = false;
    bool useEmbedding = false; // false=hand-crafted, true=ResNet18
    int  saveCount    = 0;

    std::cout << "=== Real-time 2D Object Recognition ===" << std::endl;
    std::cout << "Keys: t=threshold, m=morphology, s=segment, f=features" << std::endl;
    std::cout << "      c=classify, e=toggle embedding mode, l=label/train, p=save, q=quit" << std::endl;
    if (dirMode)
        std::cout << "      n=next image, b=previous image" << std::endl;

    // --- Main loop ---
    while (true) {
        if (liveMode) {
            cap >> frame;
            if (frame.empty()) { std::cerr << "Error: Empty frame." << std::endl; break; }
        } else {
            frame = staticImage.clone();
        }

        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        int threshVal = dynamicThreshold(blurred, threshed);
        cleanBinary(threshed, cleaned);
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);

        RegionProps mainRegion;
        std::vector<float> featureVec;
        bool hasRegion = false;

        if (nRegions > 0) {
            if (computeRegionProps(regionMap, 1, mainRegion) == 0) {
                buildFeatureVector(mainRegion, featureVec);
                hasRegion = true;
            }
        }

        std::string classLabel = "";
        float classDist = 0.0f;

        if (classifyOn && hasRegion) {
            if (useEmbedding && modelLoaded && !embDB.empty()) {
                // === Task 8: ResNet18 embedding classification ===
                cv::Mat embimage;
                prepEmbeddingImage(frame, embimage,
                                   mainRegion.cx, mainRegion.cy,
                                   mainRegion.theta,
                                   mainRegion.minE1, mainRegion.maxE1,
                                   mainRegion.minE2, mainRegion.maxE2, 0);
                if (!embimage.empty()) {
                    cv::Mat embedding;
                    getEmbedding(embimage, embedding, net, 0);
                    std::vector<float> embVec;
                    if (embedding.isContinuous()) {
                        float *data = embedding.ptr<float>(0);
                        embVec.assign(data, data + embedding.total());
                    }
                    classifyEmbedding(embVec, embDB, classLabel, classDist);
                }
            } else if (!trainingDB.empty()) {
                // === Task 6: Hand-crafted feature classification ===
                classifyNN(featureVec, trainingDB, classLabel, classDist);
            }
        }

        cv::Mat display = frame.clone();

        if (showFeatures && hasRegion)
            drawFeatures(display, mainRegion);

        if (classifyOn && hasRegion && !classLabel.empty()) {
            drawFeatures(display, mainRegion, classLabel);
            cv::putText(display, "Class: " + classLabel,
                cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(0, 255, 255), 2);
            cv::putText(display, "Dist: " + std::to_string(classDist).substr(0, 5),
                cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 200, 200), 2);
        }

        // Mode indicator
        std::string modeStr = useEmbedding ? "Mode: ResNet18" : "Mode: Hand-crafted";
        cv::Scalar modeColor = useEmbedding ? cv::Scalar(255, 165, 0) : cv::Scalar(0, 255, 0);

        cv::putText(display, "Thresh: " + std::to_string(threshVal),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Regions: " + std::to_string(nRegions),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "DB: " + std::to_string(trainingDB.size()) + " entries",
            cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, modeStr,
            cv::Point(10, display.rows - 50), cv::FONT_HERSHEY_SIMPLEX,
            0.7, modeColor, 2);

        if (dirMode) {
            std::string fname = imageFiles[imageIndex];
            size_t slash = fname.find_last_of("/\\");
            if (slash != std::string::npos) fname = fname.substr(slash + 1);
            cv::putText(display, "Image " + std::to_string(imageIndex+1) + "/" +
                std::to_string(imageFiles.size()) + ": " + fname,
                cv::Point(10, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Video", display);
        if (showThresh)                    cv::imshow("Threshold",  threshed);
        if (showMorph)                     cv::imshow("Morphology", cleaned);
        if (showSegment && nRegions > 0)   cv::imshow("Segments",   colorMap);

        int key = cv::waitKey(liveMode ? 10 : 50) & 0xFF;
        if (key == 'q' || key == 27) { std::cout << "Exiting..." << std::endl; break; }

        switch (key) {
            case 't':
                showThresh = !showThresh;
                if (!showThresh) cv::destroyWindow("Threshold");
                break;
            case 'm':
                showMorph = !showMorph;
                if (!showMorph) cv::destroyWindow("Morphology");
                break;
            case 's':
                showSegment = !showSegment;
                if (!showSegment) cv::destroyWindow("Segments");
                break;
            case 'f':
                showFeatures = !showFeatures;
                break;
            case 'e':
                if (!modelLoaded || embDB.empty()) {
                    std::cout << "Embedding mode not available. Run --embed-train first." << std::endl;
                } else {
                    useEmbedding = !useEmbedding;
                    std::cout << "Classification mode: "
                              << (useEmbedding ? "ResNet18 Embeddings" : "Hand-crafted Features")
                              << std::endl;
                }
                break;
            case 'c':
                classifyOn = !classifyOn;
                std::cout << "Classification: " << (classifyOn ? "ON" : "OFF") << std::endl;
                if (classifyOn && trainingDB.empty())
                    std::cout << "  Warning: DB empty. Use 'l' to label first." << std::endl;
                break;
            case 'l': {
                if (!hasRegion) { std::cout << "No region detected." << std::endl; break; }
                std::cout << "Enter label: ";
                std::string label;
                std::getline(std::cin, label);
                if (!label.empty()) {
                    addTrainingEntry(trainingDB, label, featureVec);
                    saveTrainingDB(TRAINING_DB_PATH, trainingDB);
                    std::cout << "Saved '" << label << "' ("
                              << trainingDB.size() << " entries)" << std::endl;
                }
                break;
            }
            case 'n':
                if (dirMode && imageIndex < static_cast<int>(imageFiles.size()) - 1) {
                    imageIndex++;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                    std::cout << "Loaded: " << imageFiles[imageIndex] << std::endl;
                }
                break;
            case 'b':
                if (dirMode && imageIndex > 0) {
                    imageIndex--;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                    std::cout << "Loaded: " << imageFiles[imageIndex] << std::endl;
                }
                break;
            case 'p': {
                cv::imwrite("data/task1_threshold/orig_"   + std::to_string(saveCount) + ".png", frame);
                cv::imwrite("data/task1_threshold/thresh_" + std::to_string(saveCount) + ".png", threshed);
                cv::imwrite("data/task2_morphology/morph_" + std::to_string(saveCount) + ".png", cleaned);
                if (nRegions > 0)
                    cv::imwrite("data/task3_segmentation/seg_" + std::to_string(saveCount) + ".png", colorMap);
                if (hasRegion) {
                    cv::Mat fd = frame.clone();
                    drawFeatures(fd, mainRegion);
                    cv::imwrite("data/task4_features/feat_" + std::to_string(saveCount) + ".png", fd);
                    std::ofstream fout("data/task4_features/features_" + std::to_string(saveCount) + ".txt");
                    fout << "percentFilled: " << featureVec[0] << std::endl;
                    fout << "aspectRatio: "   << featureVec[1] << std::endl;
                    for (int i = 0; i < 7; i++)
                        fout << "huMoment[" << i << "]: " << featureVec[i+2] << std::endl;
                    fout.close();
                }
                if (classifyOn && hasRegion && !classLabel.empty()) {
                    cv::Mat cd = frame.clone();
                    drawFeatures(cd, mainRegion, classLabel);
                    cv::imwrite("data/task6_classification/class_" + std::to_string(saveCount) + ".png", cd);
                }
                std::cout << "Saved #" << saveCount << std::endl;
                saveCount++;
                break;
            }
        }
    }

    if (liveMode) cap.release();
    cv::destroyAllWindows();
    return 0;
}