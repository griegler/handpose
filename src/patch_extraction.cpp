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

#include <iostream>

#include "patch_extraction.h"

#include <rv/stat/core.h>

#include <opencv2/imgproc/imgproc.hpp>

template <typename Dtype>
std::vector< HandPatch<Dtype> > PatchExtraction<Dtype>::operator()(const cv::Mat_< Dtype >& depth, const HandSegmentationResult& segmentation) const {

  cv::Rect roi = segmentation.roi_;
  cv::Mat_<uchar> mask = segmentation.mask_;
  
  //extract hand pixels in roi where mask(row, col) == 1
  cv::Mat_<Dtype> hand_depth(roi.height, roi.width);
  for(int y = 0; y < roi.height; ++y) {
    for(int x = 0; x < roi.width; ++x) {
      int col = x + roi.x;
      int row = y + roi.y;
      hand_depth(y, x) = mask(row, col) == 0 ? bg_value_ : depth(row, col);
    }
  }
  
  //cut out patches
  std::vector<HandPatch<Dtype> > hand_patches(patch_widths_.size());
  for(size_t hp_idx = 0; hp_idx < hand_patches.size(); ++hp_idx) {
    int patch_width = patch_widths_[hp_idx];
    
    int max_width = roi.height > roi.width ? roi.height : roi.width;
    hand_patches[hp_idx].scale_ = float(patch_width) / float(max_width);
    
    cv::Mat_<Dtype> hand_depth_scaled;
    cv::resize(hand_depth, hand_depth_scaled, cv::Size(), hand_patches[hp_idx].scale_, hand_patches[hp_idx].scale_, cv::INTER_NEAREST);
    
    hand_patches[hp_idx].patch_ = cv::Mat_<Dtype>(patch_width, patch_width, bg_value_);
    hand_patches[hp_idx].row_offset_ = (patch_width - hand_depth_scaled.rows) / 2;
    hand_patches[hp_idx].col_offset_ = (patch_width - hand_depth_scaled.cols) / 2;
    
    for(int row = 0; row < hand_depth_scaled.rows; ++row) {
      for(int col = 0; col < hand_depth_scaled.cols; ++col) {
        hand_patches[hp_idx].patch_(hand_patches[hp_idx].row_offset_ + row, hand_patches[hp_idx].col_offset_ + col) = hand_depth_scaled(row, col);
      }
    }
    
    //normalize
    if(normalize_depth_) {
      std::vector<Dtype> values;
      for(int row = 0; row < hand_patches[hp_idx].patch_.rows; ++row) {
        for(int col = 0; col < hand_patches[hp_idx].patch_.cols; ++col) {
          Dtype val = hand_patches[hp_idx].patch_(row, col);
          if(val < bg_value_) {
            values.push_back(val);
          }
        }
      }
      
      hand_patches[hp_idx].mean_depth_ = rv::stat::Mean(values);
      hand_patches[hp_idx].std_depth_ = rv::stat::StandardDeviation(values, hand_patches[hp_idx].mean_depth_);
    }
    else {
      hand_patches[hp_idx].mean_depth_ = 0;
      hand_patches[hp_idx].std_depth_ = 1;
    }
    
    
    for(int row = 0; row < hand_patches[hp_idx].patch_.rows; ++row) {
      for(int col = 0; col < hand_patches[hp_idx].patch_.cols; ++col) {
        Dtype val = (hand_patches[hp_idx].patch_(row, col) - hand_patches[hp_idx].mean_depth_) 
          / hand_patches[hp_idx].std_depth_;
        hand_patches[hp_idx].patch_(row, col) = val < 3.0 ? val : 3.0;
      }
    }
  }
  
  return hand_patches;
}


template <typename Dtype>
bool PatchExtraction<Dtype>::isReasonableHandPatch(const std::vector< HandPatch< Dtype > >& patches) const {
  for(size_t idx = 0; idx < patches.size(); ++idx) {
    for(int row = 0; row < patches[idx].patch_.rows; ++row) {
      for(int col = 0; col < patches[idx].patch_.cols; ++col) {
        Dtype d = patches[idx].patch_(row, col);
        if(d < -6 || d > 3.2) {
          std::cout << "mu = " << patches[idx].mean_depth_ << " sigma = " << patches[idx].std_depth_ << std::endl;
          std::cout << "d = " << d << std::endl;
          return false;
        }
      }
    }
  }
  
  return true;
}



template class PatchExtraction<float>;
template class PatchExtraction<double>;
