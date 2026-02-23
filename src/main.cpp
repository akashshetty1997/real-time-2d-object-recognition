/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: src/main.cpp
 *
 * Usage:
 *   ./objrec                              # live camera (fullscreen dashboard)
 *   ./objrec dev_images/chisel.jpg        # single image
 *   ./objrec dev_images/                  # directory of images
 *   ./objrec --batch dev_images/          # batch process for report
 *   ./objrec --evaluate eval.csv          # confusion matrix evaluation
 *   ./objrec --train-dir training_images/ # bulk train from labeled folders
 *   ./objrec --embed-train train.csv      # embedding training
 *   ./objrec --embed-evaluate eval.csv    # embedding evaluation
 *
 * Key controls:
 *   c = classify, e = toggle embedding mode, l = label/train
 *   d = delete last DB entry, p = save, q = quit
 *   n = next image, b = previous image (directory mode only)
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
#include "gui.h"

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
// =============================================================
int runEvaluation(const std::string &csvPath) {

    std::cout << "=== Task 7: Confusion Matrix Evaluation ===" << std::endl;

    std::vector<TrainingEntry> trainingDB;
    if (loadTrainingDB(TRAINING_DB_PATH, trainingDB) != 0 || trainingDB.empty()) {
        std::cerr << "Error: Training DB is empty." << std::endl;
        return -1;
    }

    std::ifstream fin(csvPath);
    if (!fin.is_open()) { std::cerr << "Error: Cannot open " << csvPath << std::endl; return -1; }

    struct EvalEntry { std::string imagePath, trueLabel; };
    std::vector<EvalEntry> evalSet;
    std::string line;
    bool firstLine = true;

    while (std::getline(fin, line)) {
        if (firstLine) { firstLine = false; continue; }
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        EvalEntry e;
        if (std::getline(ss, e.imagePath, ',') && std::getline(ss, e.trueLabel, ',')) {
            while (!e.trueLabel.empty() && (e.trueLabel.back() == '\r' || e.trueLabel.back() == ' '))
                e.trueLabel.pop_back();
            evalSet.push_back(e);
        }
    }
    fin.close();

    std::set<std::string> labelSet;
    for (const auto &e : evalSet) labelSet.insert(e.trueLabel);
    std::vector<std::string> labels(labelSet.begin(), labelSet.end());
    int N = static_cast<int>(labels.size());
    std::map<std::string, int> labelIdx;
    for (int i = 0; i < N; i++) labelIdx[labels[i]] = i;

    std::vector<std::vector<int>> confMatrix(N, std::vector<int>(N, 0));
    int correct = 0, total = 0, noDetection = 0;

    for (const auto &e : evalSet) {
        std::cout << e.imagePath << " (true: " << e.trueLabel << ") -> ";
        cv::Mat frame = cv::imread(e.imagePath);
        if (frame.empty()) { std::cerr << "SKIP" << std::endl; noDetection++; continue; }

        cv::Mat blurred, threshed, cleaned, regionMap, colorMap;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        dynamicThreshold(blurred, threshed);
        cleanBinary(threshed, cleaned);
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);

        if (nRegions == 0) { std::cout << "NO DETECTION" << std::endl; noDetection++; total++; continue; }

        RegionProps props; std::vector<float> fv;
        if (computeRegionProps(regionMap, 1, props) != 0) { noDetection++; total++; continue; }
        buildFeatureVector(props, fv);

        std::string predicted; float dist;
        classifyNN(fv, trainingDB, predicted, dist);
        std::cout << predicted << " (dist: " << dist << ")";

        if (labelIdx.count(e.trueLabel) && labelIdx.count(predicted)) {
            confMatrix[labelIdx[e.trueLabel]][labelIdx[predicted]]++;
            if (e.trueLabel == predicted) { correct++; std::cout << " ✓"; } else { std::cout << " ✗"; }
        }
        total++;
        std::cout << std::endl;
    }

    int colW = 14;
    float accuracy = (total > 0) ? (100.0f * correct / total) : 0.0f;

    std::cout << std::endl << "=== CONFUSION MATRIX ===" << std::endl;
    std::cout << std::setw(colW) << "TRUE\\PRED";
    for (const auto &l : labels) std::cout << std::setw(colW) << l;
    std::cout << std::endl << std::string(colW * (N + 1), '-') << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(colW) << labels[i];
        for (int j = 0; j < N; j++) std::cout << std::setw(colW) << confMatrix[i][j];
        std::cout << std::endl;
    }
    std::cout << std::endl << "Accuracy: " << accuracy << "%" << std::endl;

    system("mkdir -p data/task7_evaluation");
    std::ofstream fout("data/task7_evaluation/confusion_matrix.txt");
    fout << "=== CONFUSION MATRIX ===" << std::endl;
    fout << std::setw(colW) << "TRUE\\PRED";
    for (const auto &l : labels) fout << std::setw(colW) << l;
    fout << std::endl;
    for (int i = 0; i < N; i++) {
        fout << std::setw(colW) << labels[i];
        for (int j = 0; j < N; j++) fout << std::setw(colW) << confMatrix[i][j];
        fout << std::endl;
    }
    fout << std::endl << "Accuracy: " << accuracy << "%" << std::endl;
    fout.close();

    // Visual confusion matrix image
    int cs = 120, is = cs * (N + 1);
    cv::Mat cmImg(is, is, CV_8UC3, cv::Scalar(255, 255, 255));
    int maxVal = 1;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) maxVal = std::max(maxVal, confMatrix[i][j]);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        int x = (j+1)*cs, y = (i+1)*cs; int val = confMatrix[i][j];
        cv::Scalar cc;
        if (i==j) { int g = 255 - (int)(200.0f*val/maxVal); cc = cv::Scalar(g,255,g); }
        else if (val>0) { int r = 255-(int)(200.0f*val/maxVal); cc = cv::Scalar(r,r,255); }
        else cc = cv::Scalar(245,245,245);
        cv::rectangle(cmImg, cv::Point(x,y), cv::Point(x+cs,y+cs), cc, cv::FILLED);
        cv::rectangle(cmImg, cv::Point(x,y), cv::Point(x+cs,y+cs), cv::Scalar(180,180,180), 1);
        if (val>0) cv::putText(cmImg, std::to_string(val), cv::Point(x+cs/2-10,y+cs/2+8), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,0), 2);
    }
    for (int i = 0; i < N; i++) {
        std::string s = labels[i].substr(0,8);
        cv::putText(cmImg, s, cv::Point(2,(i+1)*cs+cs/2+5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        cv::putText(cmImg, s, cv::Point((i+1)*cs+5,cs-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
    cv::imwrite("data/task7_evaluation/confusion_matrix.png", cmImg);
    std::cout << "Saved: data/task7_evaluation/" << std::endl;
    return 0;
}

// =============================================================
// runTrainDir - Train from labeled directory structure
//
// Directory structure:
//   training_images/
//     chisel/          <- folder name = label
//       img1.png
//     triangle/
//       img1.png
//
// Usage: ./objrec --train-dir training_images/
// =============================================================
int runTrainDir(const std::string &dirPath) {

    std::cout << "=== Training from Directory ===" << std::endl;
    system("mkdir -p data/task5_training");

    std::vector<TrainingEntry> trainingDB;
    loadTrainingDB(TRAINING_DB_PATH, trainingDB);
    int startSize = static_cast<int>(trainingDB.size());

    std::vector<cv::String> subdirs;
    cv::glob(dirPath + "/*", subdirs, false);

    for (const auto &subdir : subdirs) {
        struct stat info;
        if (stat(subdir.c_str(), &info) != 0) continue;
        if (!(info.st_mode & S_IFDIR)) continue;

        std::string label = subdir;
        size_t lastSlash = label.find_last_of("/\\");
        if (lastSlash != std::string::npos) label = label.substr(lastSlash + 1);
        if (label.empty() || label[0] == '.') continue;

        std::cout << std::endl << "Label: " << label << std::endl;

        std::vector<std::string> images;
        std::vector<cv::String> cvFiles;
        cv::glob(subdir + "/*.jpg", cvFiles, false);
        for (auto &f : cvFiles) images.push_back(f);
        cvFiles.clear();
        cv::glob(subdir + "/*.jpeg", cvFiles, false);
        for (auto &f : cvFiles) images.push_back(f);
        cvFiles.clear();
        cv::glob(subdir + "/*.png", cvFiles, false);
        for (auto &f : cvFiles) images.push_back(f);
        std::sort(images.begin(), images.end());

        for (const auto &imgPath : images) {
            std::cout << "  " << imgPath << " -> ";

            cv::Mat frame = cv::imread(imgPath);
            if (frame.empty()) { std::cout << "FAILED" << std::endl; continue; }

            cv::Mat blurred, threshed, cleaned, regionMap, colorMap;
            cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
            dynamicThreshold(blurred, threshed);
            cleanBinary(threshed, cleaned);
            int nRegions = segmentRegions(cleaned, regionMap, colorMap);

            if (nRegions == 0) { std::cout << "NO REGIONS" << std::endl; continue; }

            RegionProps props;
            std::vector<float> fv;
            if (computeRegionProps(regionMap, 1, props) != 0) { std::cout << "FAILED" << std::endl; continue; }
            buildFeatureVector(props, fv);

            addTrainingEntry(trainingDB, label, fv);
            std::cout << "OK (fill=" << fv[0] << " ar=" << fv[1] << ")" << std::endl;
        }
    }

    saveTrainingDB(TRAINING_DB_PATH, trainingDB);

    int added = static_cast<int>(trainingDB.size()) - startSize;
    std::cout << std::endl << "=== Training Complete ===" << std::endl;
    std::cout << "Added " << added << " new entries. Total: " << trainingDB.size() << std::endl;
    std::cout << "Saved to: " << TRAINING_DB_PATH << std::endl;
    return 0;
}

// =============================================================
// main
// =============================================================
int main(int argc, char *argv[]) {

    if (argc > 2 && std::string(argv[1]) == "--batch")
        return runBatchProcessing(argv[2]);
    if (argc > 2 && std::string(argv[1]) == "--evaluate")
        return runEvaluation(argv[2]);
    if (argc > 2 && std::string(argv[1]) == "--train-dir")
        return runTrainDir(argv[2]);
    if (argc > 2 && std::string(argv[1]) == "--embed-train")
        return runEmbedTraining(argv[2], MODEL_PATH);
    if (argc > 2 && std::string(argv[1]) == "--embed-evaluate")
        return runEmbedEvaluate(argv[2], MODEL_PATH);

    // --- Determine input mode ---
    bool liveMode = true, dirMode = false;
    cv::VideoCapture cap;
    cv::Mat staticImage;
    std::vector<std::string> imageFiles;
    int imageIndex = 0;

    if (argc > 1) {
        std::string inputPath = argv[1];
        if (isDirectory(inputPath)) {
            dirMode = true; liveMode = false;
            imageFiles = getImageFiles(inputPath);
            if (imageFiles.empty()) { std::cerr << "Error: No images in " << inputPath << std::endl; return -1; }
            std::cout << "Directory mode: " << imageFiles.size() << " images" << std::endl;
            staticImage = cv::imread(imageFiles[0]);
            if (staticImage.empty()) { std::cerr << "Error: Cannot read " << imageFiles[0] << std::endl; return -1; }
        } else {
            liveMode = false;
            staticImage = cv::imread(inputPath);
            if (staticImage.empty()) { std::cerr << "Error: Cannot read " << inputPath << std::endl; return -1; }
        }
    } else {
        cap.open(0);
        if (!cap.isOpened()) { std::cerr << "Error: Cannot open camera." << std::endl; return -1; }
        std::cout << "Live camera mode" << std::endl;
    }

    system("mkdir -p data/task1_threshold data/task2_morphology data/task3_segmentation");
    system("mkdir -p data/task4_features data/task5_training data/task6_classification");
    system("mkdir -p data/task7_evaluation data/task8_embeddings");

    // --- Load databases ---
    std::vector<TrainingEntry> trainingDB;
    loadTrainingDB(TRAINING_DB_PATH, trainingDB);

    std::vector<EmbeddingEntry> embDB;
    loadEmbeddingDB("data/task8_embeddings/embedding_db.csv", embDB);

    cv::dnn::Net net;
    bool modelLoaded = false;
    if (!embDB.empty()) {
        try { net = cv::dnn::readNet(MODEL_PATH); modelLoaded = true; }
        catch (...) { std::cout << "Warning: Could not load ResNet18." << std::endl; }
    }

    // --- State ---
    cv::Mat frame, blurred, threshed, cleaned, regionMap, colorMap;
    bool classifyOn = false, useEmbedding = false;
    int saveCount = 0;

    std::cout << "=== Real-time 2D Object Recognition ===" << std::endl;
    std::cout << "Keys: c=classify, e=embedding, l=label, d=delete, p=save, q=quit" << std::endl;
    if (dirMode) std::cout << "      n=next, b=previous" << std::endl;

    // --- Main loop ---
    while (true) {
        if (liveMode) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            frame = staticImage.clone();
        }

        // --- Pipeline ---
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        int threshVal = dynamicThreshold(blurred, threshed);
        cleanBinary(threshed, cleaned);
        int nRegions = segmentRegions(cleaned, regionMap, colorMap);

        RegionProps mainRegion;
        std::vector<float> featureVec;
        bool hasRegion = false;
        if (nRegions > 0 && computeRegionProps(regionMap, 1, mainRegion) == 0) {
            buildFeatureVector(mainRegion, featureVec);
            hasRegion = true;
        }

        // --- Classification ---
        std::string classLabel = "";
        float classDist = 0.0f;
        if (classifyOn && hasRegion) {
            if (useEmbedding && modelLoaded && !embDB.empty()) {
                cv::Mat embimage;
                prepEmbeddingImage(frame, embimage, mainRegion.cx, mainRegion.cy,
                                   mainRegion.theta, mainRegion.minE1, mainRegion.maxE1,
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
                classifyNN(featureVec, trainingDB, classLabel, classDist);
            }
        }

        // --- Build dashboard images ---
        cv::Mat featImg = frame.clone();
        cv::Mat classImg = frame.clone();
        if (hasRegion) drawFeatures(featImg, mainRegion);
        if (hasRegion && !classLabel.empty()) drawFeatures(classImg, mainRegion, classLabel);

        std::string imageInfo = "";
        if (dirMode) {
            std::string fname = imageFiles[imageIndex];
            size_t slash = fname.find_last_of("/\\");
            if (slash != std::string::npos) fname = fname.substr(slash + 1);
            imageInfo = "Image " + std::to_string(imageIndex + 1) + "/" +
                        std::to_string(imageFiles.size()) + ": " + fname;
        }

        // --- Unified fullscreen dashboard ---
        cv::Mat dashboard;
        createUnifiedDashboard(frame, threshed, cleaned,
                               nRegions > 0 ? colorMap : cv::Mat(),
                               featImg, classImg,
                               trainingDB, classLabel, classDist,
                               classifyOn, useEmbedding, imageInfo,
                               dashboard);
        cv::imshow("Dashboard", dashboard);

        // Make fullscreen on first frame
        static bool firstFrame = true;
        if (firstFrame) {
            cv::setWindowProperty("Dashboard", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            firstFrame = false;
        }

        // --- Key handling ---
        int key = cv::waitKey(liveMode ? 10 : 50) & 0xFF;
        if (key == 'q' || key == 27) { std::cout << "Exiting..." << std::endl; break; }

        switch (key) {
            case 'c':
                classifyOn = !classifyOn;
                std::cout << "Classification: " << (classifyOn ? "ON" : "OFF") << std::endl;
                if (classifyOn && trainingDB.empty())
                    std::cout << "  Warning: DB empty. Use 'l' or --train-dir first." << std::endl;
                break;
            case 'e':
                if (!modelLoaded || embDB.empty())
                    std::cout << "Embedding mode not available. Run --embed-train first." << std::endl;
                else {
                    useEmbedding = !useEmbedding;
                    std::cout << "Mode: " << (useEmbedding ? "ResNet18" : "Hand-crafted") << std::endl;
                }
                break;
            case 'd':
                if (!trainingDB.empty()) {
                    std::string removed = trainingDB.back().label;
                    trainingDB.pop_back();
                    saveTrainingDB(TRAINING_DB_PATH, trainingDB);
                    std::cout << "Deleted '" << removed << "'. " << trainingDB.size() << " left." << std::endl;
                } else std::cout << "DB empty." << std::endl;
                break;
            case 'l': {
                if (!hasRegion) { std::cout << "No region." << std::endl; break; }
                std::cout << "Enter label: ";
                std::string label;
                std::getline(std::cin, label);
                if (!label.empty()) {
                    addTrainingEntry(trainingDB, label, featureVec);
                    saveTrainingDB(TRAINING_DB_PATH, trainingDB);
                    std::cout << "Saved '" << label << "' (" << trainingDB.size() << " entries)" << std::endl;
                }
                break;
            }
            case 'n':
                if (dirMode && imageIndex < static_cast<int>(imageFiles.size()) - 1) {
                    imageIndex++;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                }
                break;
            case 'b':
                if (dirMode && imageIndex > 0) {
                    imageIndex--;
                    staticImage = cv::imread(imageFiles[imageIndex]);
                }
                break;
            case 'p': {
                cv::imwrite("data/task1_threshold/orig_" + std::to_string(saveCount) + ".png", frame);
                cv::imwrite("data/task1_threshold/thresh_" + std::to_string(saveCount) + ".png", threshed);
                cv::imwrite("data/task2_morphology/morph_" + std::to_string(saveCount) + ".png", cleaned);
                if (nRegions > 0)
                    cv::imwrite("data/task3_segmentation/seg_" + std::to_string(saveCount) + ".png", colorMap);
                if (hasRegion) {
                    cv::imwrite("data/task4_features/feat_" + std::to_string(saveCount) + ".png", featImg);
                    std::ofstream fo("data/task4_features/features_" + std::to_string(saveCount) + ".txt");
                    fo << "percentFilled: " << featureVec[0] << "\naspectRatio: " << featureVec[1] << std::endl;
                    for (int i = 0; i < 7; i++) fo << "huMoment[" << i << "]: " << featureVec[i+2] << std::endl;
                    fo.close();
                }
                if (classifyOn && hasRegion && !classLabel.empty())
                    cv::imwrite("data/task6_classification/class_" + std::to_string(saveCount) + ".png", classImg);
                cv::imwrite("data/task4_features/dashboard_" + std::to_string(saveCount) + ".png", dashboard);
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