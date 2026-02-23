/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: include/classify.h
 *
 * Purpose:
 * Header for training database management and classification functions.
 *
 * Training database:
 *   - Stored as a CSV file with label and feature vector per row
 *   - Each row: label,feature0,feature1,...,featureN
 *   - Loaded at startup, saved when new entries are added
 *
 * Classification:
 *   - Scaled Euclidean distance (nearest neighbor)
 *   - Each feature dimension is divided by its standard deviation
 *     across the training set, so all features contribute equally
 *   - Unknown object is assigned the label of the closest match
 */

#ifndef CLASSIFY_H
#define CLASSIFY_H

#include "vision.h"

/*
 * loadTrainingDB
 *
 * Load training database from a CSV file.
 * Each line: label,f0,f1,f2,...,fN
 *
 * @param filename  Path to CSV file
 * @param db        Output vector of training entries
 * @return          0 on success, -1 on error (file not found is OK, returns empty db)
 */
int loadTrainingDB(const std::string &filename, std::vector<TrainingEntry> &db);

/*
 * saveTrainingDB
 *
 * Save training database to a CSV file.
 *
 * @param filename  Path to CSV file
 * @param db        Vector of training entries to save
 * @return          0 on success, -1 on error
 */
int saveTrainingDB(const std::string &filename, const std::vector<TrainingEntry> &db);

/*
 * addTrainingEntry
 *
 * Add a new labeled feature vector to the training database.
 *
 * @param db        Training database (modified in place)
 * @param label     Object label (e.g., "chisel", "eraser")
 * @param features  Feature vector from buildFeatureVector()
 */
void addTrainingEntry(std::vector<TrainingEntry> &db, const std::string &label,
                      const std::vector<float> &features);

/*
 * computeStdevs
 *
 * Compute standard deviation for each feature across the entire DB.
 * Used by classifyNN for scaled Euclidean distance.
 *
 * @param db      Training database
 * @param stdevs  Output vector of standard deviations (one per feature)
 */
void computeStdevs(const std::vector<TrainingEntry> &db, std::vector<float> &stdevs);

/*
 * classifyNN
 *
 * Classify an unknown feature vector using nearest-neighbor with
 * scaled Euclidean distance: sum((x_i - y_i)^2 / stdev_i^2)
 *
 * @param features   Feature vector of unknown object
 * @param db         Training database
 * @param bestLabel  Output - label of closest match
 * @param bestDist   Output - distance to closest match
 * @return           0 on success, -1 if DB is empty
 */
int classifyNN(const std::vector<float> &features, const std::vector<TrainingEntry> &db,
               std::string &bestLabel, float &bestDist);

#endif // CLASSIFY_H