#ifndef HAND_SEGMENTATION_MEANSHIFT_H
#define HAND_SEGMENTATION_MEANSHIFT_H

#include "hand_segmentation.h"
#include "projection.h"

template <typename Dtype>
class HandSegmentationMeanshift : public HandSegmentation<Dtype> {
public: 
    HandSegmentationMeanshift(const pose::SegmentationParameter& param, const Projection& projection, Dtype bg_value) : 
        HandSegmentation<Dtype>(param), projection_(projection), bg_value_(bg_value) {}
    virtual ~HandSegmentationMeanshift() {}
    
    virtual std::vector<HandSegmentationResult> operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const;
    
protected:
    const Projection& projection_;
    const Dtype bg_value_;
};

#endif