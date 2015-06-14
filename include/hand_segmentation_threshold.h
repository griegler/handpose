#ifndef HAND_SEGMENTATION_THRESHOLD_H
#define HAND_SEGMENTATION_THRESHOLD_H

#include "hand_segmentation.h"
#include "projection.h"

template <typename Dtype>
class HandSegmentationThreshold : public HandSegmentation<Dtype> {
public: 
    HandSegmentationThreshold(const pose::SegmentationParameter& param, const Projection& projection, Dtype bg_value) : 
        HandSegmentation<Dtype>(param), projection_(projection), bg_value_(bg_value) {}
    virtual ~HandSegmentationThreshold() {}
    
    virtual std::vector<HandSegmentationResult> operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const;
    
protected:
    const Projection& projection_;
    const Dtype bg_value_;
};

#endif