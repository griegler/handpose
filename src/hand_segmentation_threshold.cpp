#include "hand_segmentation_threshold.h"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>


template <typename Dtype>
std::vector< HandSegmentationResult > HandSegmentationThreshold<Dtype>::operator()(const cv::Mat_< Dtype >& depth_in, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const {
  const pose::SegmentationThresholdParameter& threshold_param = this->param_.threshold_param();
  float min_ir = threshold_param.min_ir();
  float max_ir = threshold_param.max_ir();
  float min_d = threshold_param.min_d();
  float max_d = threshold_param.max_d();
  float t = threshold_param.t();
  
  
  cv::Mat_<Dtype> depth;
  if(threshold_param.type() == pose::SegmentationThresholdParameter_Type_NEAREST) {
    cv::medianBlur(depth_in, depth, 5);
  }
  else {
    depth = depth_in;
  }
  
  
  float min_depth = 1e9;
  if(threshold_param.type() == pose::SegmentationThresholdParameter_Type_NEAREST) {
    for(int row = 0; row < depth.rows; ++row) {
      for(int col = 0; col < depth.cols; ++col) {
        float d = depth(row, col);
        if(d < min_depth) min_depth = d;
      }
    }
  }
  
  
  cv::Mat_<uchar> mask = cv::Mat_<uchar>::zeros(depth.rows, depth.cols);
  int min_y = mask.rows;
  int max_y = 0;
  int min_x = mask.cols;
  int max_x = 0;
  for(int row = 0; row < mask.rows; ++row) {
    for(int col = 0; col < mask.cols; ++col) {
      Dtype d = depth(row, col);
      Dtype i = ir(row, col);
      
      if(threshold_param.type() == pose::SegmentationThresholdParameter_Type_RANGE) {
        if(d >= min_d && d <= max_d && i >= min_ir && i <= max_ir) {
          mask(row, col ) = 1;
          
          if(row < min_y) min_y = row;
          if(row > max_y) max_y = row;
          if(col < min_x) min_x = col;
          if(col > max_x) max_x = col;
        }
      }
      else if(threshold_param.type() == pose::SegmentationThresholdParameter_Type_NEAREST) {
        if(d >= min_depth && d <= (min_depth + t)) {
          mask(row, col ) = 1;
          
          if(row < min_y) min_y = row;
          if(row > max_y) max_y = row;
          if(col < min_x) min_x = col;
          if(col > max_x) max_x = col;
        }
      }
      
    }
  }
  
  cv::Rect roi(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  
  std::vector<HandSegmentationResult> result(1);
  result[0].roi_ = roi;
  result[0].mask_ = mask;
  
  return result;
}


template class HandSegmentationThreshold<float>;
template class HandSegmentationThreshold<double>;