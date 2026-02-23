/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: include/batch.h
 *
 * Purpose:
 * Header for batch processing tool.
 * Processes all images in a directory and saves report images.
 */

#ifndef BATCH_H
#define BATCH_H

#include "vision.h"

// Extract filename without path or extension
std::string extractFilename(const std::string &path);

// Process all images in a directory through the pipeline
int runBatchProcessing(const std::string &dirPath);

#endif // BATCH_H