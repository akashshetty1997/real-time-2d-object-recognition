/*
 * Name: Akash Shridhar Shetty, Skandhan Madhusudhana
 * Date: February 2025
 * File: src/gui.cpp
 *
 * Purpose:
 * Extension: Unified fullscreen GUI dashboard.
 * Shows EVERYTHING in ONE window — no toggle keys needed.
 * Left side: 2x3 grid of pipeline stages (original, threshold, morph, segments, features, classification)
 */

#include "gui.h"

/**
 * resizeToCell - Resize any image to fit a dashboard cell
 */
static void resizeToCell(const cv::Mat &src, cv::Mat &dst, int w, int h) {
    cv::Mat temp;
    if (src.empty()) {
        dst = cv::Mat::zeros(h, w, CV_8UC3);
        cv::putText(dst, "No Data", cv::Point(w / 2 - 40, h / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);
        return;
    }
    if (src.channels() == 1) cv::cvtColor(src, temp, cv::COLOR_GRAY2BGR);
    else temp = src.clone();
    cv::resize(temp, dst, cv::Size(w, h));
}

/**
 * drawSidePanel - Draw the right-side info panel
 */
static void drawSidePanel(cv::Mat &panel,
                           const std::vector<TrainingEntry> &db,
                           const std::string &classLabel,
                           float classDist,
                           bool classifyOn,
                           bool useEmbedding,
                           const std::string &imageInfo) {

    int w = panel.cols;
    int y = 10;

    // === Section 1: Classification Result ===

    cv::putText(panel, "CLASSIFICATION",
                cv::Point(10, y + 15), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0, 200, 255), 1);
    y += 28;

    if (classifyOn && !classLabel.empty()) {
        cv::putText(panel, classLabel,
                    cv::Point(10, y + 28), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);
        y += 40;

        cv::putText(panel, "Dist: " + std::to_string(classDist).substr(0, 5),
                    cv::Point(10, y + 14), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, cv::Scalar(180, 180, 180), 1);
        y += 22;

        float confidence = std::max(0.0f, std::min(1.0f, 1.0f - classDist / 10.0f));
        int barW = static_cast<int>((w - 30) * confidence);

        cv::rectangle(panel, cv::Point(10, y), cv::Point(w - 10, y + 22),
                      cv::Scalar(50, 50, 50), cv::FILLED);

        cv::Scalar barColor;
        if (confidence > 0.6f)      barColor = cv::Scalar(0, 200, 0);
        else if (confidence > 0.3f) barColor = cv::Scalar(0, 200, 200);
        else                        barColor = cv::Scalar(0, 0, 200);

        cv::rectangle(panel, cv::Point(10, y), cv::Point(10 + barW, y + 22),
                      barColor, cv::FILLED);

        std::string confStr = std::to_string(static_cast<int>(confidence * 100)) + "% confidence";
        cv::putText(panel, confStr, cv::Point(15, y + 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        y += 32;

    } else if (classifyOn) {
        cv::putText(panel, "No object", cv::Point(10, y + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(100, 100, 100), 1);
        y += 30;
    } else {
        cv::putText(panel, "OFF (press c)", cv::Point(10, y + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(100, 100, 100), 1);
        y += 30;
    }

    // === Separator ===
    y += 5;
    cv::line(panel, cv::Point(10, y), cv::Point(w - 10, y), cv::Scalar(60, 60, 60), 1);
    y += 12;

    // === Section 2: Mode indicator ===

    std::string modeStr = useEmbedding ? "ResNet18 Embeddings" : "Hand-crafted Features";
    cv::Scalar modeCol = useEmbedding ? cv::Scalar(255, 165, 0) : cv::Scalar(0, 200, 0);
    cv::putText(panel, "Mode: " + modeStr, cv::Point(10, y + 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, modeCol, 1);
    y += 24;

    // === Separator ===
    cv::line(panel, cv::Point(10, y), cv::Point(w - 10, y), cv::Scalar(60, 60, 60), 1);
    y += 12;

    // === Section 3: Database Summary ===

    cv::putText(panel, "DATABASE", cv::Point(10, y + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 200, 255), 1);
    y += 28;

    std::map<std::string, int> classCounts;
    for (const auto &e : db) classCounts[e.label]++;

    cv::putText(panel, std::to_string(classCounts.size()) + " classes, " +
                std::to_string(db.size()) + " total entries",
                cv::Point(10, y + 14), cv::FONT_HERSHEY_SIMPLEX,
                0.42, cv::Scalar(180, 180, 180), 1);
    y += 24;

    if (db.empty()) {
        cv::putText(panel, "No data yet", cv::Point(10, y + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 150), 1);
        y += 18;
        cv::putText(panel, "Press 'l' to label objects", cv::Point(10, y + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 150), 1);
        y += 18;
        cv::putText(panel, "or use --train-dir", cv::Point(10, y + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 150), 1);
    } else {
        int maxCount = 1;
        for (const auto &p : classCounts)
            if (p.second > maxCount) maxCount = p.second;

        std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 0, 220),   cv::Scalar(0, 200, 0),
            cv::Scalar(220, 0, 0),   cv::Scalar(0, 200, 200),
            cv::Scalar(200, 0, 200), cv::Scalar(200, 200, 0),
            cv::Scalar(0, 128, 255), cv::Scalar(128, 255, 0),
            cv::Scalar(255, 128, 0), cv::Scalar(128, 0, 255)
        };

        int ci = 0;
        int barMaxW = w - 130;

        for (const auto &p : classCounts) {
            if (y + 24 > panel.rows - 80) break;

            cv::Scalar col = colors[ci % colors.size()];

            if (p.first == classLabel && classifyOn) {
                cv::rectangle(panel, cv::Point(5, y - 2),
                              cv::Point(w - 5, y + 20),
                              cv::Scalar(40, 60, 40), cv::FILLED);
            }

            std::string shortName = p.first.substr(0, 12);
            cv::putText(panel, shortName, cv::Point(10, y + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(200, 200, 200), 1);

            int barW = static_cast<int>(barMaxW * p.second / maxCount);
            cv::rectangle(panel, cv::Point(110, y + 3),
                          cv::Point(110 + barW, y + 18),
                          col, cv::FILLED);

            cv::putText(panel, std::to_string(p.second),
                        cv::Point(115 + barW, y + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(150, 150, 150), 1);

            y += 24;
            ci++;
        }
    }

    // === Section 4: Key controls ===

    int bottomY = panel.rows - 70;
    cv::line(panel, cv::Point(10, bottomY), cv::Point(w - 10, bottomY),
             cv::Scalar(60, 60, 60), 1);
    bottomY += 14;

    cv::putText(panel, "CONTROLS", cv::Point(10, bottomY),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 200, 255), 1);
    bottomY += 18;
    cv::putText(panel, "c=classify  l=label  d=delete",
                cv::Point(10, bottomY), cv::FONT_HERSHEY_SIMPLEX,
                0.35, cv::Scalar(130, 130, 130), 1);
    bottomY += 15;
    cv::putText(panel, "e=embed  n/b=nav  p=save  q=quit",
                cv::Point(10, bottomY), cv::FONT_HERSHEY_SIMPLEX,
                0.35, cv::Scalar(130, 130, 130), 1);

    // === Image info ===
    if (!imageInfo.empty()) {
        cv::putText(panel, imageInfo, cv::Point(10, panel.rows - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(100, 100, 100), 1);
    }
}

/**
 * createUnifiedDashboard - Build the complete fullscreen dashboard
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
                            int cellW, int cellH) {

    // === Fullscreen sizing (MacBook Air M4: 1440x900) ===
    int screenW = 1440;
    int screenH = 900;
    int panelW = 300;

    int gridW = screenW - panelW;
    cellW = gridW / 3;
    cellH = screenH / 2;

    int totalW = cellW * 3 + panelW;
    int totalH = cellH * 2;

    dashboard = cv::Mat(totalH, totalW, CV_8UC3, cv::Scalar(25, 25, 25));

    // === Left side: 2x3 pipeline grid ===

    cv::Mat cells[6];
    resizeToCell(original, cells[0], cellW, cellH);
    resizeToCell(threshed, cells[1], cellW, cellH);
    resizeToCell(cleaned,  cells[2], cellW, cellH);
    resizeToCell(colorMap, cells[3], cellW, cellH);
    resizeToCell(featImg,  cells[4], cellW, cellH);
    resizeToCell(classImg, cells[5], cellW, cellH);

    std::string titles[6] = {
        "1. Original",   "2. Threshold",  "3. Morphology",
        "4. Segments",   "5. Features",   "6. Classification"
    };

    cv::Scalar titleColors[6] = {
        cv::Scalar(255, 255, 255), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255),   cv::Scalar(0, 200, 255),
        cv::Scalar(255, 100, 0),   cv::Scalar(0, 255, 255)
    };

    for (int i = 0; i < 6; i++) {
        cv::rectangle(cells[i], cv::Point(0, 0), cv::Point(cellW, 25),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(cells[i], titles[i], cv::Point(8, 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, titleColors[i], 1);
    }

    // Row 0
    cells[0].copyTo(dashboard(cv::Rect(0,        0,     cellW, cellH)));
    cells[1].copyTo(dashboard(cv::Rect(cellW,    0,     cellW, cellH)));
    cells[2].copyTo(dashboard(cv::Rect(cellW*2,  0,     cellW, cellH)));
    // Row 1
    cells[3].copyTo(dashboard(cv::Rect(0,        cellH, cellW, cellH)));
    cells[4].copyTo(dashboard(cv::Rect(cellW,    cellH, cellW, cellH)));
    cells[5].copyTo(dashboard(cv::Rect(cellW*2,  cellH, cellW, cellH)));

    // Grid lines
    cv::Scalar gridCol(60, 60, 60);
    cv::line(dashboard, cv::Point(cellW, 0), cv::Point(cellW, totalH), gridCol, 1);
    cv::line(dashboard, cv::Point(cellW*2, 0), cv::Point(cellW*2, totalH), gridCol, 1);
    cv::line(dashboard, cv::Point(0, cellH), cv::Point(cellW*3, cellH), gridCol, 1);

    // === Right side panel ===

    int panelX = cellW * 3;
    cv::Mat sidePanel = dashboard(cv::Rect(panelX, 0, panelW, totalH));
    sidePanel.setTo(cv::Scalar(30, 30, 30));

    cv::line(dashboard, cv::Point(panelX, 0), cv::Point(panelX, totalH),
             cv::Scalar(80, 80, 80), 2);

    drawSidePanel(sidePanel, db, classLabel, classDist,
                  classifyOn, useEmbedding, imageInfo);
}