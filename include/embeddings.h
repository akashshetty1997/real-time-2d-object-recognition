/*
 * Name: Akash Shridhar Shetty , Skandhan Madhusudhana
 * Date: February 2025
 * File: include/embeddings.h
 *
 * Purpose:
 * Header for ResNet18 embedding-based one-shot classification (Task 8).
 * Uses Professor Maxwell's utility functions to extract ROI and compute
 * 512-dimensional embeddings from a pre-trained ResNet18 network.
 */

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "vision.h"

// Entry in the embedding training database
struct EmbeddingEntry {
    std::string label;
    std::vector<float> embedding; // 512-dimensional ResNet18 embedding
};

/*
 * loadEmbeddingDB - Load embedding database from CSV
 */
int loadEmbeddingDB(const std::string &filename, std::vector<EmbeddingEntry> &db);

/*
 * saveEmbeddingDB - Save embedding database to CSV
 */
int saveEmbeddingDB(const std::string &filename, const std::vector<EmbeddingEntry> &db);

/*
 * classifyEmbedding - Classify using sum-squared difference on embeddings
 */
int classifyEmbedding(const std::vector<float> &queryEmb,
                      const std::vector<EmbeddingEntry> &db,
                      std::string &bestLabel, float &bestDist);

/*
 * runEmbedTraining - Train embedding DB from labeled images CSV
 * Usage: ./objrec --embed-train data/task8_embeddings/train_images.csv
 */
int runEmbedTraining(const std::string &csvPath, const std::string &modelPath);

/*
 * runEmbedEvaluate - Evaluate embedding classifier, produce confusion matrix
 * Usage: ./objrec --embed-evaluate data/task7_evaluation/eval_images.csv
 */
int runEmbedEvaluate(const std::string &csvPath, const std::string &modelPath);

#endif // EMBEDDINGS_H
