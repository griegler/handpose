#ifndef HAND_SEGMENTATION_H
#define HAND_SEGMENTATION_H

#include "hand_segmentation_result.h"

#include <opencv2/core/core.hpp>

#include "pose.pb.h"

template <typename Dtype>
class HandSegmentation {
public: 
  HandSegmentation(const pose::SegmentationParameter& param) : param_(param) { }
  virtual ~HandSegmentation() {}
  
  virtual std::vector<HandSegmentationResult> operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const = 0;
      
protected:
  const pose::SegmentationParameter& param_;
};

#endif