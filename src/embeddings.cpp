/*
 * Name: Akash Shridhar Shetty , Skandhan Madhusudhana
 * Date: February 2025
 * File: src/embeddings.cpp
 *
 * Purpose:
 * Task 8 - One-shot classification using ResNet18 embeddings.
 *
 * Pipeline:
 *   1. Load pre-trained ResNet18 ONNX model
 *   2. For each training image:
 *      a. Run full vision pipeline (threshold, clean, segment, features)
 *      b. Use prepEmbeddingImage() to rotate+crop the object ROI
 *      c. Feed ROI through getEmbedding() to get 512-dim vector
 *      d. Store label + embedding in CSV database
 *   3. For classification: find nearest embedding by sum-squared difference
 *
 * Why embeddings vs hand-crafted features:
 *   Hand-crafted features (Hu moments, fill ratio) capture geometric shape.
 *   ResNet18 embeddings capture appearance, texture, and complex patterns
 *   learned from millions of images. One-shot means only ONE training
 *   example per class is needed.
 *
 * Attribution: prepEmbeddingImage() and getEmbedding() by Prof. Bruce Maxwell
 */

#include "embeddings.h"
#include "threshold.h"
#include "morphology.h"
#include "segment.h"
#include "features.h"

// Include professor's utility functions
// These are defined in src/utilities.cpp
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage,
                        int cx, int cy, float theta,
                        float minE1, float maxE1,
                        float minE2, float maxE2, int debug);

/**
 * loadEmbeddingDB - Load embedding database from CSV
 *
 * Format: label,e0,e1,e2,...,e511
 */
int loadEmbeddingDB(const std::string &filename, std::vector<EmbeddingEntry> &db) {
    db.clear();

    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cout << "Embedding DB not found at " << filename << " (starting fresh)" << std::endl;
        return 0;
    }

    std::string line;
    int lineNum = 0;

    while (std::getline(fin, line)) {
        lineNum++;
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        EmbeddingEntry entry;

        if (!std::getline(ss, entry.label, ',')) continue;

        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                entry.embedding.push_back(std::stof(val));
            } catch (...) {}
        }

        if (!entry.embedding.empty()) {
            db.push_back(entry);
        }
    }

    fin.close();
    std::cout << "Loaded " << db.size() << " embedding entries from " << filename << std::endl;
    return 0;
}

/**
 * saveEmbeddingDB - Save embedding database to CSV
 */
int saveEmbeddingDB(const std::string &filename, const std::vector<EmbeddingEntry> &db) {
    std::ofstream fout(filename);

    if (!fout.is_open()) {
        std::cerr << "Error: Cannot open " << filename << " for writing." << std::endl;
        return -1;
    }

    fout << "# Embedding DB: label,e0,e1,...,e511" << std::endl;

    for (const auto &entry : db) {
        fout << entry.label;
        for (float v : entry.embedding) {
            fout << "," << v;
        }
        fout << std::endl;
    }

    fout.close();
    std::cout << "Saved " << db.size() << " entries to " << filename << std::endl;
    return 0;
}

/**
 * classifyEmbedding - Find nearest neighbor using sum-squared difference
 *
 * SSD distance: sum( (a_i - b_i)^2 ) for all 512 dimensions
 * Lower = more similar
 */
int classifyEmbedding(const std::vector<float> &queryEmb,
                      const std::vector<EmbeddingEntry> &db,
                      std::string &bestLabel, float &bestDist) {

    if (db.empty()) {
        std::cerr << "Error: Embedding DB is empty." << std::endl;
        return -1;
    }

    bestDist  = std::numeric_limits<float>::max();
    bestLabel = "unknown";

    for (const auto &entry : db) {
        float ssd = 0.0f;
        int n = std::min(queryEmb.size(), entry.embedding.size());

        for (int i = 0; i < n; i++) {
            float diff = queryEmb[i] - entry.embedding[i];
            ssd += diff * diff;
        }

        if (ssd < bestDist) {
            bestDist  = ssd;
            bestLabel = entry.label;
        }
    }

    return 0;
}

/**
 * processImageForEmbedding - Run full pipeline and extract embedding
 *
 * Returns 0 on success, -1 if no object detected
 */
static int processImageForEmbedding(const std::string &imagePath,
                                     cv::dnn::Net &net,
                                     std::vector<float> &embVec) {
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Cannot read: " << imagePath << std::endl;
        return -1;
    }

    // Full vision pipeline
    cv::Mat blurred, threshed, cleaned, regionMap, colorMap;
    cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
    dynamicThreshold(blurred, threshed);
    cleanBinary(threshed, cleaned);
    int nRegions = segmentRegions(cleaned, regionMap, colorMap);

    if (nRegions == 0) {
        std::cerr << "No regions found in: " << imagePath << std::endl;
        return -1;
    }

    // Compute region properties for largest region
    RegionProps props;
    if (computeRegionProps(regionMap, 1, props) != 0) {
        std::cerr << "Feature computation failed for: " << imagePath << std::endl;
        return -1;
    }

    // Prepare embedding image using professor's utility
    // Rotates image by -theta and crops the oriented bounding box
    cv::Mat embimage;
    prepEmbeddingImage(frame, embimage,
                       props.cx, props.cy,
                       props.theta,
                       props.minE1, props.maxE1,
                       props.minE2, props.maxE2,
                       0); // debug=0

    if (embimage.empty()) {
        std::cerr << "Empty embedding image for: " << imagePath << std::endl;
        return -1;
    }

    // Get 512-dim embedding from ResNet18
    cv::Mat embedding;
    getEmbedding(embimage, embedding, net, 0);

    // Convert cv::Mat embedding to std::vector<float>
    embVec.clear();
    if (embedding.isContinuous()) {
        float *data = embedding.ptr<float>(0);
        int n = embedding.total();
        embVec.assign(data, data + n);
    }

    return 0;
}

/**
 * runEmbedTraining - Build embedding database from labeled training CSV
 *
 * CSV format (same as your existing training CSV or a new one):
 *   image_path,label
 *
 * Usage: ./objrec --embed-train data/task8_embeddings/train_images.csv
 */
int runEmbedTraining(const std::string &csvPath, const std::string &modelPath) {

    std::cout << "=== Task 8: Embedding Training ===" << std::endl;

    // Load ResNet18 model
    std::cout << "Loading model: " << modelPath << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNet(modelPath);
    } catch (const cv::Exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully." << std::endl;

    // Parse training CSV
    std::ifstream fin(csvPath);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open " << csvPath << std::endl;
        return -1;
    }

    struct TrainEntry { std::string path, label; };
    std::vector<TrainEntry> trainSet;
    std::string line;
    bool firstLine = true;

    while (std::getline(fin, line)) {
        if (firstLine) { firstLine = false; continue; }
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        TrainEntry e;
        if (std::getline(ss, e.path, ',') && std::getline(ss, e.label, ',')) {
            while (!e.label.empty() &&
                   (e.label.back() == '\r' || e.label.back() == ' '))
                e.label.pop_back();
            trainSet.push_back(e);
        }
    }
    fin.close();
    std::cout << "Training set: " << trainSet.size() << " images" << std::endl;

    // Compute embeddings
    std::vector<EmbeddingEntry> embDB;

    for (const auto &e : trainSet) {
        std::cout << "Processing: " << e.path << " (" << e.label << ") -> ";

        std::vector<float> embVec;
        if (processImageForEmbedding(e.path, net, embVec) != 0) {
            std::cout << "FAILED" << std::endl;
            continue;
        }

        EmbeddingEntry entry;
        entry.label     = e.label;
        entry.embedding = embVec;
        embDB.push_back(entry);

        std::cout << "OK (" << embVec.size() << " dims)" << std::endl;
    }

    // Save embedding DB
    system("mkdir -p data/task8_embeddings");
    std::string dbPath = "data/task8_embeddings/embedding_db.csv";
    saveEmbeddingDB(dbPath, embDB);

    std::cout << std::endl;
    std::cout << "Embedding DB saved to: " << dbPath << std::endl;
    std::cout << "Total entries: " << embDB.size() << std::endl;

    return 0;
}

/**
 * runEmbedEvaluate - Evaluate embedding classifier with confusion matrix
 *
 * Usage: ./objrec --embed-evaluate data/task7_evaluation/eval_images.csv
 */
int runEmbedEvaluate(const std::string &csvPath, const std::string &modelPath) {

    std::cout << "=== Task 8: Embedding Evaluation ===" << std::endl;

    // Load model
    std::cout << "Loading model: " << modelPath << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNet(modelPath);
    } catch (const cv::Exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Load embedding DB
    std::vector<EmbeddingEntry> embDB;
    if (loadEmbeddingDB("data/task8_embeddings/embedding_db.csv", embDB) != 0 ||
        embDB.empty()) {
        std::cerr << "Error: Embedding DB empty. Run --embed-train first." << std::endl;
        return -1;
    }

    // Parse eval CSV
    std::ifstream fin(csvPath);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open " << csvPath << std::endl;
        return -1;
    }

    struct EvalEntry { std::string path, trueLabel; };
    std::vector<EvalEntry> evalSet;
    std::string line;
    bool firstLine = true;

    while (std::getline(fin, line)) {
        if (firstLine) { firstLine = false; continue; }
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        EvalEntry e;
        if (std::getline(ss, e.path, ',') && std::getline(ss, e.trueLabel, ',')) {
            while (!e.trueLabel.empty() &&
                   (e.trueLabel.back() == '\r' || e.trueLabel.back() == ' '))
                e.trueLabel.pop_back();
            evalSet.push_back(e);
        }
    }
    fin.close();
    std::cout << "Eval set: " << evalSet.size() << " images" << std::endl;

    // Collect unique labels from eval set only
    std::set<std::string> labelSet;
    for (const auto &e : evalSet) labelSet.insert(e.trueLabel);
    std::vector<std::string> labels(labelSet.begin(), labelSet.end());
    int N = static_cast<int>(labels.size());

    std::map<std::string, int> labelIdx;
    for (int i = 0; i < N; i++) labelIdx[labels[i]] = i;

    // Initialize confusion matrix
    std::vector<std::vector<int>> confMatrix(N, std::vector<int>(N, 0));
    int correct = 0, total = 0, noDetect = 0;

    // Evaluate each image
    for (const auto &e : evalSet) {
        std::cout << "Processing: " << e.path
                  << " (true: " << e.trueLabel << ") -> ";

        std::vector<float> embVec;
        if (processImageForEmbedding(e.path, net, embVec) != 0) {
            std::cout << "NO DETECTION" << std::endl;
            noDetect++;
            total++;
            continue;
        }

        std::string predicted;
        float dist;
        classifyEmbedding(embVec, embDB, predicted, dist);

        std::cout << "predicted: " << predicted
                  << " (ssd: " << dist << ")";

        if (labelIdx.count(e.trueLabel) && labelIdx.count(predicted)) {
            confMatrix[labelIdx[e.trueLabel]][labelIdx[predicted]]++;
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

    // Print confusion matrix
    int colW = 14;
    std::cout << std::endl;
    std::cout << "=== EMBEDDING CONFUSION MATRIX ===" << std::endl;
    std::cout << "Rows = True Label, Cols = Predicted Label" << std::endl << std::endl;

    std::cout << std::setw(colW) << "TRUE\\PRED";
    for (const auto &l : labels) std::cout << std::setw(colW) << l;
    std::cout << std::endl;
    std::cout << std::string(colW * (N + 1), '-') << std::endl;

    for (int i = 0; i < N; i++) {
        std::cout << std::setw(colW) << labels[i];
        for (int j = 0; j < N; j++)
            std::cout << std::setw(colW) << confMatrix[i][j];
        std::cout << std::endl;
    }

    float accuracy = (total > 0) ? (100.0f * correct / total) : 0.0f;
    std::cout << std::endl;
    std::cout << "Correct: " << correct << "/" << total
              << "  Accuracy: " << std::fixed << std::setprecision(1)
              << accuracy << "%" << std::endl;
    if (noDetect > 0)
        std::cout << "No detection: " << noDetect << " images" << std::endl;

    // Save results
    system("mkdir -p data/task8_embeddings");

    std::ofstream fout("data/task8_embeddings/embedding_confusion_matrix.txt");
    fout << "=== EMBEDDING CONFUSION MATRIX ===" << std::endl;
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
    fout << std::endl;
    fout << "Comparison:" << std::endl;
    fout << "  Hand-crafted features: 58.3%" << std::endl;
    fout << "  ResNet18 embeddings:   " << std::fixed << std::setprecision(1)
         << accuracy << "%" << std::endl;
    fout.close();

    std::cout << std::endl;
    std::cout << "Saved: data/task8_embeddings/embedding_confusion_matrix.txt" << std::endl;
    std::cout << std::endl;
    std::cout << "=== COMPARISON ===" << std::endl;
    std::cout << "Hand-crafted features: 58.3%" << std::endl;
    std::cout << "ResNet18 embeddings:   " << std::fixed << std::setprecision(1)
              << accuracy << "%" << std::endl;

    return 0;
}
