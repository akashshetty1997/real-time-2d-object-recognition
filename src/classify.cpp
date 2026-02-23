/*
 * Name: Akash Shridhar Shetty , SKandhan Madhusudhana
 * Date: February 2025
 * File: src/classify.cpp
 *
 * Purpose:
 * Implementation of training database management and classification.
 *
 * Training database format (CSV):
 *   chisel,0.94,0.40,-1.8,-4.2,-9.5,-10.1,-20.3,-10.5,-20.8
 *   eraser,0.95,0.45,-1.7,-4.0,-9.2,-9.8,-19.5,-10.1,-20.2
 *
 * Classification uses scaled Euclidean distance:
 *   distance = sqrt( sum( ((x_i - y_i) / stdev_i)^2 ) )
 *
 * Why scaled Euclidean:
 *   - Without scaling, features with large values (like Hu moments ~20)
 *     would dominate over features with small values (like fill ~0.5)
 *   - Dividing by standard deviation normalizes each feature so they
 *     all contribute roughly equally to the distance
 *   - This is equivalent to assuming each feature has equal importance
 *
 * Example:
 *   Training DB has: chisel (fill=0.30, ar=0.08), eraser (fill=0.94, ar=0.40)
 *   Unknown object:  fill=0.92, ar=0.38
 *   stdev_fill = 0.32, stdev_ar = 0.16
 *
 *   Distance to chisel: sqrt(((0.92-0.30)/0.32)^2 + ((0.38-0.08)/0.16)^2)
 *                      = sqrt(3.78 + 3.52) = sqrt(7.30) = 2.70
 *   Distance to eraser: sqrt(((0.92-0.94)/0.32)^2 + ((0.38-0.40)/0.16)^2)
 *                      = sqrt(0.004 + 0.016) = sqrt(0.02) = 0.14
 *   -> Classified as "eraser" (nearest neighbor)
 */

#include "classify.h"

/**
 * loadTrainingDB - Load training database from CSV
 *
 * @param filename  Path to CSV file
 * @param db        Output vector of training entries
 * @return          0 on success, -1 on error
 *
 * CSV format: label,f0,f1,f2,...,fN
 * One entry per line. Empty lines and lines starting with # are skipped.
 *
 * If the file doesn't exist, returns an empty db (not an error,
 * since we might be starting fresh with no training data yet).
 */
int loadTrainingDB(const std::string &filename, std::vector<TrainingEntry> &db) {

    db.clear();

    std::ifstream fin(filename);

    // File not found is OK - just start with empty DB
    if (!fin.is_open()) {
        std::cout << "Training DB not found at " << filename << " (starting fresh)" << std::endl;
        return 0;
    }

    std::string line;
    int lineNum = 0;

    while (std::getline(fin, line)) {
        lineNum++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Parse: label,f0,f1,f2,...,fN
        std::stringstream ss(line);
        TrainingEntry entry;

        // Read the label (first field before comma)
        if (!std::getline(ss, entry.label, ',')) {
            std::cerr << "Warning: Cannot parse label on line " << lineNum << std::endl;
            continue;
        }

        // Read all feature values
        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                entry.features.push_back(std::stof(val));
            } catch (...) {
                std::cerr << "Warning: Cannot parse value '" << val
                          << "' on line " << lineNum << std::endl;
            }
        }

        if (!entry.features.empty()) {
            db.push_back(entry);
        }
    }

    fin.close();
    std::cout << "Loaded " << db.size() << " entries from " << filename << std::endl;

    return 0;
}

/**
 * saveTrainingDB - Save training database to CSV
 *
 * @param filename  Path to CSV file
 * @param db        Vector of training entries to save
 * @return          0 on success, -1 on error
 *
 * Writes one entry per line: label,f0,f1,f2,...,fN
 * Overwrites the file completely each time.
 */
int saveTrainingDB(const std::string &filename, const std::vector<TrainingEntry> &db) {

    std::ofstream fout(filename);

    if (!fout.is_open()) {
        std::cerr << "Error: Cannot open " << filename << " for writing." << std::endl;
        return -1;
    }

    // Write header comment
    fout << "# Training DB: label,feature0,feature1,...,featureN" << std::endl;

    for (const auto &entry : db) {
        fout << entry.label;

        for (size_t i = 0; i < entry.features.size(); i++) {
            fout << "," << entry.features[i];
        }

        fout << std::endl;
    }

    fout.close();
    return 0;
}

/**
 * addTrainingEntry - Add a new labeled entry to the training DB
 *
 * @param db        Training database (modified in place)
 * @param label     Object label
 * @param features  Feature vector
 *
 * Simply appends the new entry. Multiple entries per label are
 * allowed and encouraged - more examples per object improve
 * classification accuracy.
 */
void addTrainingEntry(std::vector<TrainingEntry> &db, const std::string &label,
                      const std::vector<float> &features) {

    TrainingEntry entry;
    entry.label = label;
    entry.features = features;
    db.push_back(entry);
}

/**
 * computeStdevs - Compute standard deviation for each feature
 *
 * @param db      Training database
 * @param stdevs  Output vector of standard deviations
 *
 * Algorithm:
 *   1. For each feature dimension i:
 *      a. Compute mean: mean_i = sum(x_i) / N
 *      b. Compute variance: var_i = sum((x_i - mean_i)^2) / (N-1)
 *      c. stdev_i = sqrt(var_i)
 *   2. If stdev is 0 (all values identical), set to 1.0 to avoid division by zero
 *
 * Example with 3 entries, 2 features each:
 *   Entry 0: [0.30, 0.08]
 *   Entry 1: [0.94, 0.40]
 *   Entry 2: [0.50, 0.75]
 *
 *   Feature 0: mean=0.58, stdev=0.33
 *   Feature 1: mean=0.41, stdev=0.34
 */
void computeStdevs(const std::vector<TrainingEntry> &db, std::vector<float> &stdevs) {

    stdevs.clear();

    if (db.empty()) return;

    int nFeatures = static_cast<int>(db[0].features.size());
    int n = static_cast<int>(db.size());

    stdevs.resize(nFeatures, 0.0f);

    // === Compute mean for each feature ===

    std::vector<float> means(nFeatures, 0.0f);

    for (const auto &entry : db) {
        for (int i = 0; i < nFeatures && i < static_cast<int>(entry.features.size()); i++) {
            means[i] += entry.features[i];
        }
    }

    for (int i = 0; i < nFeatures; i++) {
        means[i] /= n;
    }

    // === Compute standard deviation for each feature ===

    for (const auto &entry : db) {
        for (int i = 0; i < nFeatures && i < static_cast<int>(entry.features.size()); i++) {
            float diff = entry.features[i] - means[i];
            stdevs[i] += diff * diff;
        }
    }

    for (int i = 0; i < nFeatures; i++) {
        if (n > 1) {
            stdevs[i] = std::sqrt(stdevs[i] / (n - 1));
        }

        // Prevent division by zero: if stdev is 0, set to 1.0
        // This happens when all training entries have the same value for a feature
        if (stdevs[i] < 1e-6f) {
            stdevs[i] = 1.0f;
        }
    }
}

/**
 * classifyNN - Nearest-neighbor classification with scaled Euclidean distance
 *
 * @param features   Feature vector of unknown object
 * @param db         Training database
 * @param bestLabel  Output - label of closest match
 * @param bestDist   Output - distance to closest match
 * @return           0 on success, -1 if DB is empty
 *
 * Scaled Euclidean distance:
 *   d(x, y) = sqrt( sum_i ( (x_i - y_i) / stdev_i )^2 )
 *
 * This normalizes each feature by its standard deviation across the
 * training set, so features with large ranges don't dominate.
 *
 * Steps:
 *   1. Compute standard deviations across entire training DB
 *   2. For each training entry, compute scaled Euclidean distance
 *   3. Return the label of the entry with minimum distance
 *
 * Example:
 *   Unknown:  [0.92, 0.38, -1.7, ...]
 *   DB entry "eraser": [0.94, 0.40, -1.8, ...]
 *   stdevs: [0.32, 0.16, 0.5, ...]
 *
 *   distance = sqrt( ((0.92-0.94)/0.32)^2 + ((0.38-0.40)/0.16)^2 + ... )
 *            = sqrt( 0.004 + 0.016 + ... )
 *            = small value -> classified as "eraser"
 */
int classifyNN(const std::vector<float> &features, const std::vector<TrainingEntry> &db,
               std::string &bestLabel, float &bestDist) {

    // === Step 1: Validate ===

    if (db.empty()) {
        std::cerr << "Error: Training DB is empty, cannot classify." << std::endl;
        return -1;
    }

    if (features.empty()) {
        std::cerr << "Error: Feature vector is empty, cannot classify." << std::endl;
        return -1;
    }

    // === Step 2: Compute standard deviations ===

    std::vector<float> stdevs;
    computeStdevs(db, stdevs);

    // === Step 3: Find nearest neighbor ===

    bestDist = std::numeric_limits<float>::max();
    bestLabel = "unknown";

    for (const auto &entry : db) {
        float dist = 0.0f;

        // Compute scaled Euclidean distance
        int nFeatures = std::min(static_cast<int>(features.size()),
                                  static_cast<int>(entry.features.size()));

        for (int i = 0; i < nFeatures; i++) {
            float diff = features[i] - entry.features[i];
            float scaled = diff / stdevs[i];
            dist += scaled * scaled;
        }

        dist = std::sqrt(dist);

        // Update best match if this is closer
        if (dist < bestDist) {
            bestDist = dist;
            bestLabel = entry.label;
        }
    }

    return 0;
}