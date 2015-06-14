#ifndef HAND_SEGMENTATION_RESULT_H
#define HAND_SEGMENTATION_RESULT_H

#include <opencv2/core/core.hpp>

struct HandSegmentationResult {
    cv::Mat_<uchar> mask_;
    cv::Rect roi_;
};

#endif