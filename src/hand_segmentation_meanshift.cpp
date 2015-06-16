// Copyright (C) 2015 Institute for Computer Graphics and Vision (ICG),
//   Graz University of Technology (TU GRAZ)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the ICG, TU GRAZ.
// 4. Neither the name of the ICG, TU GRAZ nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY ICG, TU GRAZ ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL ICG, TU GRAZ BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "hand_segmentation_meanshift.h"
#include "common.h"

#include <rv/ocv/meanshift.h>
#include <rv/ocv/io/csv.h>
#include <rv/stat/core.h>

#include <opencv2/imgproc/imgproc.hpp>



template <typename Dtype>
std::vector<HandSegmentationResult> HandSegmentationMeanshift<Dtype>::operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d_c) const {    
  
  const pose::SegmentationMeanshiftParameter param = this->param_.meanshift_param();
  bool use_hint = param.use_hint();
  float bandwidth = param.bandwidth();
  float neighbourhood = param.neighbourhood();
  float max_iters = param.max_iters();
  float eps = param.eps();
  
  //  smooth
  cv::Mat_<Dtype> smoothed;
  cv::medianBlur(depth, smoothed, 5);
  
  //get starting point
  cv::Vec3f hint_2d = hint_2d_c;
  if(!use_hint || smoothed(hint_2d(1), hint_2d(0)) >= bg_value_) {
    hint_2d(2) = bg_value_;
    for(int row = 0; row < smoothed.rows; ++row) {
      for(int col = 0; col < smoothed.cols; ++col) {
        Dtype d = smoothed(row, col);
        if(d < hint_2d(2)) {
          hint_2d(0) = col;
          hint_2d(1) = row;
          hint_2d(2) = d;
        }
      }
    }
  }
  
  
  //-----------------------------------
  //meanshift clustering

  //  project points 
  std::vector<cv::Vec3f> pts3;
  for(int y = 0; y < smoothed.rows; ++y) {
    for(int x = 0; x < smoothed.cols; ++x) {
      Dtype d = smoothed(y, x);
      if(d < bg_value_) {
        cv::Vec3f pt3 = projection_.to3D(cv::Vec3f(x, y, d));
        pts3.push_back(pt3);
      }
    }
  }

  //cluster hand with mean shift
  std::vector<int> indices;
  cv::Vec3f hint_3d =  projection_.to3D(hint_2d);
//   std::vector<cv::Vec3f> centers = rv::ocv::meanshift(pts3, hint_3d, 0.075, 1e-5, 100, 0.1, indices);
  std::vector<cv::Vec3f> centers = rv::ocv::meanshift(pts3, hint_3d, bandwidth, neighbourhood, max_iters, eps, indices);


  //find bb 
  int min_x = depth.cols;
  int min_y = depth.rows;
  int max_x = 0;
  int max_y = 0;
  Dtype max_d = 0;
  for(size_t idx = 0; idx < indices.size(); ++idx) {
    cv::Vec3f pt2 = projection_.to2D(pts3[indices[idx]]);
    
    float x = pt2(0);
    float y = pt2(1);
    float d = pt2(2);
    
    if(min_x > x) min_x = x;
    if(max_x < x) max_x = x;
    if(min_y > y) min_y = y;
    if(max_y < y) max_y = y;
    if(max_d < d) max_d = d;
  }  

  if(min_x == depth.cols || min_y == depth.rows || max_x == 0 || max_y == 0) {
    return std::vector<HandSegmentationResult>();
  }

  cv::Rect roi = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y); //TODO: add +1 to width/height

  //create hand mask
  cv::Mat_<uchar> mask = cv::Mat_<uchar>::zeros(smoothed.rows, smoothed.cols);
  size_t idx = 0;
  size_t pt_idx = 0;
  for(int y = 0; y < smoothed.rows; ++y) {
    for(int x = 0; x < smoothed.cols; ++x) {
      Dtype d = smoothed(y, x);
      if(d < bg_value_) {
        bool contains = false;                
        if(pt_idx < indices.size() && idx == indices[pt_idx]) {
          contains = true;
          pt_idx++;
        }
        
        if(d < max_d) {
          contains = true;
        }
        
        if(contains) {
          mask(y, x) = 1;
        }
        
        idx++;
      }
    }
  } 

  std::vector<HandSegmentationResult> result(1);
  result[0].roi_ = roi;
  result[0].mask_ = mask;

  return result;
}



template class HandSegmentationMeanshift<float>;
template class HandSegmentationMeanshift<double>; 
