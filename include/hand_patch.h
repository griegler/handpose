#ifndef HAND_PATCH_H
#define HAND_PATCH_H

#include <opencv2/core/core.hpp>

template <typename Dtype>
class HandPatch {
public:
    HandPatch() {}
    virtual ~HandPatch() {}
    
    std::vector<cv::Vec3f> annoInHandPatch(const std::vector<cv::Vec3f>& anno, const cv::Rect& roi) const;
    std::vector<cv::Vec3f> annoFromHandPatch(const std::vector<cv::Vec3f>& anno, const cv::Rect& roi) const;
    
    std::vector<cv::Vec3f> centerAnno(const std::vector<cv::Vec3f>& anno) const;
    std::vector<cv::Vec3f> uncenterAnno(const std::vector<cv::Vec3f>& anno) const;
    
    bool isReasonableCenteredAnno(const std::vector<cv::Vec3f>& anno, bool normalized) const;
    
public:
    cv::Mat_<Dtype> patch_;
    
    float scale_;
    int row_offset_;
    int col_offset_;
    
    Dtype mean_depth_;
    Dtype std_depth_;
};

#endif