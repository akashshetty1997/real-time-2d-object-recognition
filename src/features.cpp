/*
 * Name: Akash Shridhar Shetty
 * Date: February 2025
 * File: src/features.cpp
 *
 * Purpose:
 * Implementation of feature extraction and region analysis.
 *
 * Uses OpenCV's moments() function which computes:
 *   - Spatial moments: m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
 *   - Central moments: mu20, mu11, mu02, mu30, mu21, mu12, mu03
 *   - Normalized central moments: nu20, nu11, nu02, nu30, nu21, nu12, nu03
 *
 * From these we derive:
 *   - Centroid: cx = m10/m00, cy = m01/m00
 *   - Orientation (theta): 0.5 * atan2(2*mu11, mu20 - mu02)
 *     This is the angle of the axis of LEAST central moment
 *   - Oriented bounding box: project all region pixels onto the
 *     primary and secondary axes to find extents
 *   - Percent filled: region_area / (oriented_bbox_width * oriented_bbox_height)
 *   - Aspect ratio: oriented_bbox_height / oriented_bbox_width
 *   - Hu moments: 7 invariant moments computed from normalized central moments
 *
 * Why these features:
 *   - Percent filled: distinguishes solid objects (eraser ~0.95) from
 *     hollow ones (triangle ruler ~0.5) or thin ones (pen ~0.3)
 *   - Aspect ratio: distinguishes long thin objects (pen ~0.1) from
 *     square-ish ones (eraser ~0.5)
 *   - Hu moments: capture shape characteristics that are invariant to
 *     translation, scale, and rotation. Different shapes produce
 *     different Hu moment signatures.
 */

#include "features.h"

/**
 * computeRegionProps - Compute all properties for a region
 *
 * @param regionMap  Labeled region map (32SC1)
 * @param regionID   Label of the region to analyze
 * @param props      Output RegionProps struct
 * @return           0 on success, -1 on error
 *
 * Implementation details:
 *
 * Step-by-step:
 *   1. Create a binary mask for the target region
 *   2. Compute moments using OpenCV moments()
 *   3. Calculate centroid from spatial moments
 *   4. Calculate orientation angle from central moments
 *   5. Project region pixels onto primary/secondary axes for extents
 *   6. Compute oriented bounding box, percent filled, aspect ratio
 *   7. Compute Hu moments
 *
 * Example with eraser (rectangular object):
 *   m00 = 15000 (area in pixels)
 *   centroid = (320, 240) (center of image)
 *   theta = 0.3 rad (~17 degrees tilted)
 *   oriented bbox = 200 x 80 pixels
 *   percentFilled = 15000 / (200*80) = 0.94 (very solid)
 *   aspectRatio = 80/200 = 0.4 (wider than tall)
 *
 * Example with allen wrench (L-shaped):
 *   percentFilled ~ 0.3 (lots of empty space in bbox)
 *   aspectRatio ~ 0.7 (more square-ish bbox)
 */
int computeRegionProps(const cv::Mat &regionMap, int regionID, RegionProps &props) {

    // === Step 1: Validate input ===

    if (regionMap.empty()) {
        std::cerr << "Error: Region map is empty in computeRegionProps()" << std::endl;
        return -1;
    }

    // === Step 2: Create binary mask for this region ===
    //
    // Convert the labeled region map into a binary image where
    // only pixels belonging to regionID are white (255).
    // This is needed because OpenCV moments() expects a binary image.

    cv::Mat mask = cv::Mat::zeros(regionMap.size(), CV_8UC1);

    for (int r = 0; r < regionMap.rows; r++) {
        const int *regionRow = regionMap.ptr<int>(r);
        uchar *maskRow = mask.ptr<uchar>(r);

        for (int c = 0; c < regionMap.cols; c++) {
            if (regionRow[c] == regionID) {
                maskRow[c] = 255;
            }
        }
    }

    // === Step 3: Compute moments ===
    //
    // OpenCV moments() computes all spatial, central, and normalized
    // central moments from the binary mask.
    //
    // Spatial moments (m_pq):
    //   m00 = sum of all pixel values = area (for binary image)
    //   m10 = sum(x * pixel), m01 = sum(y * pixel)
    //
    // Central moments (mu_pq): moments about the centroid
    //   mu20, mu11, mu02 are the 2nd order central moments
    //   These describe the spread/shape of the region
    //
    // The second parameter (true) computes binary moments
    // (treats all non-zero pixels as 1)

    cv::Moments m = cv::moments(mask, true);

    // Check if region has enough pixels
    if (m.m00 < 1) {
        std::cerr << "Error: Region " << regionID << " has no pixels." << std::endl;
        return -1;
    }

    // === Step 4: Calculate centroid ===
    //
    // centroid_x = m10 / m00
    // centroid_y = m01 / m00
    //
    // This is the "center of mass" of the region.

    props.label = regionID;
    props.area = static_cast<int>(m.m00);
    props.cx = static_cast<int>(m.m10 / m.m00);
    props.cy = static_cast<int>(m.m01 / m.m00);

    // === Step 5: Calculate orientation angle (theta) ===
    //
    // The axis of least central moment is the direction along which
    // the region has the least spread (variance).
    //
    // theta = 0.5 * atan2(2 * mu11, mu20 - mu02)
    //
    // This gives the angle of the primary axis in radians.
    // Range: [-pi/2, pi/2]
    //
    // For the eraser tilted at ~17 degrees: theta ~ 0.3 rad
    // For a horizontal pen: theta ~ 0 rad
    // For a vertical pen: theta ~ pi/2 rad

    props.theta = 0.5f * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02);

    // === Step 6: Project region pixels onto primary/secondary axes ===
    //
    // To find the oriented bounding box, we project every region pixel
    // onto the primary axis (angle theta) and secondary axis (theta + 90).
    //
    // For each pixel (x, y):
    //   dx = x - cx, dy = y - cy  (translate to centroid)
    //   e1 = dx * cos(theta) + dy * sin(theta)   (projection on primary axis)
    //   e2 = -dx * sin(theta) + dy * cos(theta)  (projection on secondary axis)
    //
    // Track min/max of e1 and e2 to get the bounding box extents.

    float cosT = std::cos(props.theta);
    float sinT = std::sin(props.theta);

    props.minE1 = 1e9f;
    props.maxE1 = -1e9f;
    props.minE2 = 1e9f;
    props.maxE2 = -1e9f;

    for (int r = 0; r < regionMap.rows; r++) {
        const int *regionRow = regionMap.ptr<int>(r);

        for (int c = 0; c < regionMap.cols; c++) {
            if (regionRow[c] == regionID) {
                float dx = static_cast<float>(c - props.cx);
                float dy = static_cast<float>(r - props.cy);

                // Project onto primary axis (axis of least moment)
                float e1 = dx * cosT + dy * sinT;
                // Project onto secondary axis (perpendicular)
                float e2 = -dx * sinT + dy * cosT;

                // Track extents
                if (e1 < props.minE1) props.minE1 = e1;
                if (e1 > props.maxE1) props.maxE1 = e1;
                if (e2 < props.minE2) props.minE2 = e2;
                if (e2 > props.maxE2) props.maxE2 = e2;
            }
        }
    }

    // === Step 7: Compute oriented bounding box properties ===
    //
    // Width  = extent along primary axis = maxE1 - minE1
    // Height = extent along secondary axis = maxE2 - minE2
    //
    // Percent filled = area / (width * height)
    //   Solid rectangle: ~0.95
    //   Triangle: ~0.50
    //   L-shape: ~0.30
    //
    // Aspect ratio = min(width, height) / max(width, height)
    //   Always in [0, 1] so it's scale invariant
    //   Square: ~1.0
    //   Long thin: ~0.1

    float width = props.maxE1 - props.minE1;
    float height = props.maxE2 - props.minE2;

    // Prevent division by zero
    if (width < 1) width = 1;
    if (height < 1) height = 1;

    float bboxArea = width * height;
    props.percentFilled = static_cast<float>(props.area) / bboxArea;

    // Aspect ratio: always <= 1 (shorter / longer)
    if (width > height) {
        props.bboxAspectRatio = height / width;
    } else {
        props.bboxAspectRatio = width / height;
    }

    // === Step 8: Create OpenCV RotatedRect for drawing ===
    //
    // RotatedRect takes center, size, and angle (in degrees).
    // OpenCV uses degrees and measures angle differently,
    // so we convert our theta.

    props.orientedBBox = cv::RotatedRect(
        cv::Point2f(static_cast<float>(props.cx), static_cast<float>(props.cy)),
        cv::Size2f(width, height),
        props.theta * 180.0f / M_PI
    );

    // === Step 9: Compute Hu moments ===
    //
    // Hu moments are 7 values derived from normalized central moments.
    // They are invariant to translation, scale, and rotation.
    //
    // hu[0]: Related to the spread of the object
    // hu[1]: Related to the symmetry
    // hu[2]-hu[6]: Higher-order shape descriptors
    //
    // We store the raw values; in buildFeatureVector() we'll
    // log-transform them for better numerical behavior.

    cv::HuMoments(m, props.huMoments);

    return 0;
}

/**
 * buildFeatureVector - Extract classification features from RegionProps
 *
 * @param props     Input region properties (from computeRegionProps)
 * @param features  Output feature vector
 * @return          0 on success, -1 on error
 *
 * Feature vector layout (9 elements):
 *   [0] percentFilled       range ~[0.1, 1.0]
 *   [1] bboxAspectRatio     range ~[0.1, 1.0]
 *   [2] log|huMoments[0]|   range ~[-1, -10]
 *   [3] log|huMoments[1]|   range ~[-2, -20]
 *   [4] log|huMoments[2]|   range ~[-3, -30]
 *   [5] log|huMoments[3]|   range ~[-3, -30]
 *   [6] log|huMoments[4]|   range ~[-6, -60]
 *   [7] log|huMoments[5]|   range ~[-4, -40]
 *   [8] log|huMoments[6]|   range ~[-6, -60]
 *
 * Why log-transform Hu moments:
 *   Raw Hu moments vary over many orders of magnitude (e.g., 0.2 to 0.0000001).
 *   Log-transform compresses this range, making them more comparable
 *   when computing distances. We use -sign * log10(|value|) so the
 *   sign is preserved.
 *
 * Example feature vectors:
 *   Eraser:    [0.94, 0.40, -1.8, -4.2, -9.5, -10.1, -20.3, -10.5, -20.8]
 *   Pen:       [0.30, 0.08, -2.1, -5.8, -12.3, -13.1, -25.6, -12.9, -26.1]
 *   Triangle:  [0.50, 0.75, -1.5, -3.5, -7.8,  -8.2, -16.1,  -8.5, -16.5]
 */
int buildFeatureVector(const RegionProps &props, std::vector<float> &features) {

    features.clear();

    // === Feature 0: Percent filled ===
    features.push_back(props.percentFilled);

    // === Feature 1: Bounding box aspect ratio ===
    features.push_back(props.bboxAspectRatio);

    // === Features 2-8: Log-transformed Hu moments ===
    //
    // Transform: -sign(hu[i]) * log10(|hu[i]|)
    //
    // This gives values in a reasonable range (typically -1 to -30)
    // while preserving the sign information.
    //
    // If hu[i] is exactly 0 (very rare), we use a large negative value.

    for (int i = 0; i < 7; i++) {
        double h = props.huMoments[i];

        if (std::abs(h) > 0) {
            // -sign * log10(|value|)
            double sign = (h > 0) ? 1.0 : -1.0;
            features.push_back(static_cast<float>(-sign * std::log10(std::abs(h))));
        } else {
            features.push_back(0.0f);
        }
    }

    return 0;
}

/**
 * drawFeatures - Draw feature overlay on image
 *
 * @param frame  Image to draw on (modified in place)
 * @param props  Region properties to visualize
 * @param label  Optional text label to display
 *
 * Draws:
 *   1. Oriented bounding box (green rectangle)
 *   2. Primary axis line through centroid (red line)
 *   3. Centroid dot (blue circle)
 *   4. Label text near centroid (if provided)
 *   5. Feature values text (percent filled, aspect ratio)
 *
 * The axis and bounding box rotate with the object, visually
 * confirming that orientation computation is correct.
 */
void drawFeatures(cv::Mat &frame, const RegionProps &props, const std::string &label) {

    // === Draw oriented bounding box (green) ===
    //
    // Get the 4 corners of the RotatedRect and draw lines between them.

    cv::Point2f vertices[4];
    props.orientedBBox.points(vertices);

    for (int i = 0; i < 4; i++) {
        cv::line(frame, vertices[i], vertices[(i + 1) % 4],
                 cv::Scalar(0, 255, 0), 2);
    }

    // === Draw primary axis (red line) ===
    //
    // Draw a line through the centroid along the primary axis direction.
    // Length = half the extent along the primary axis.

    float axisLen = (props.maxE1 - props.minE1) / 2.0f;
    float cosT = std::cos(props.theta);
    float sinT = std::sin(props.theta);

    cv::Point2f axisStart(
        props.cx - axisLen * cosT,
        props.cy - axisLen * sinT
    );
    cv::Point2f axisEnd(
        props.cx + axisLen * cosT,
        props.cy + axisLen * sinT
    );

    cv::line(frame, axisStart, axisEnd, cv::Scalar(0, 0, 255), 2);

    // === Draw centroid (blue dot) ===

    cv::circle(frame, cv::Point(props.cx, props.cy), 5,
               cv::Scalar(255, 0, 0), -1);

    // === Draw label text ===

    if (!label.empty()) {
        cv::putText(frame, label,
                    cv::Point(props.cx - 30, props.cy - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);
    }

    // === Draw feature values ===

    std::string fillStr = "Fill: " + std::to_string(props.percentFilled).substr(0, 4);
    std::string arStr = "AR: " + std::to_string(props.bboxAspectRatio).substr(0, 4);

    cv::putText(frame, fillStr,
                cv::Point(props.cx - 30, props.cy + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 0), 1);
    cv::putText(frame, arStr,
                cv::Point(props.cx - 30, props.cy + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 0), 1);
}