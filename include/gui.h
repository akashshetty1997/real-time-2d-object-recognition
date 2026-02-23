/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: include/gui.h
 *
 * Purpose:
 * Header for the unified fullscreen GUI dashboard extension.
 * Single window showing entire pipeline + DB + stats + controls.
 */

#ifndef GUI_H
#define GUI_H

#include "vision.h"
#include "features.h"

/*
 * createUnifiedDashboard
 *
 * Build a single fullscreen dashboard showing everything:
 *   Left: 2x3 pipeline grid
 *   Right: classification result, confidence, DB summary, controls
 */
void createUnifiedDashboard(const cv::Mat &original,
                            const cv::Mat &threshed,
                            const cv::Mat &cleaned,
                            const cv::Mat &colorMap,
                            const cv::Mat &featImg,
                            const cv::Mat &classImg,
                            const std::vector<TrainingEntry> &db,
                            const std::string &classLabel,
                            float classDist,
                            bool classifyOn,
                            bool useEmbedding,
                            const std::string &imageInfo,
                            cv::Mat &dashboard,
                            int cellW = 280, int cellH = 210);

#endif // GUI_H