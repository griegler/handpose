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

#include "hand_patch.h"

template <typename Dtype>
std::vector< cv::Vec3f > HandPatch<Dtype>::annoInHandPatch(const std::vector< cv::Vec3f >& anno, const cv::Rect& roi) const {
  std::vector<cv::Vec3f> adjusted_anno(anno.size());
  for(int row = 0; row < adjusted_anno.size(); ++row) {
    Dtype x = (anno[row](0) - roi.x) * scale_ + col_offset_;
    Dtype y = (anno[row](1) - roi.y) * scale_ + row_offset_;
    Dtype z = (anno[row](2) - mean_depth_) / std_depth_;
    
    adjusted_anno[row](0) = x;
    adjusted_anno[row](1) = y;
    adjusted_anno[row](2) = z;
  }
  
  return centerAnno(adjusted_anno);
}

template <typename Dtype>
std::vector< cv::Vec3f > HandPatch<Dtype>::annoFromHandPatch(const std::vector< cv::Vec3f >& anno, const cv::Rect& roi) const {
  std::vector<cv::Vec3f> adjusted_anno = uncenterAnno(anno);
  for(int row = 0; row < anno.size(); ++row) {
    Dtype x = adjusted_anno[row](0);
    Dtype y = adjusted_anno[row](1);
    Dtype z = adjusted_anno[row](2);
    
    x = ((x - col_offset_) / scale_) + roi.x;
    y = ((y - row_offset_) / scale_) + roi.y;
    z = z * std_depth_ + mean_depth_;
    
    adjusted_anno[row](0) = x;
    adjusted_anno[row](1) = y;
    adjusted_anno[row](2) = z;
  }
  
  return adjusted_anno;
}

template <typename Dtype>
std::vector< cv::Vec3f > HandPatch<Dtype>::centerAnno(const std::vector< cv::Vec3f >& anno) const {
std::vector<cv::Vec3f> adjusted_anno(anno.size());
  for(int row = 0; row < anno.size(); ++row) {
    Dtype x = anno[row](0);
    Dtype y = anno[row](1);
    Dtype z = anno[row](2);
    
    x = (x - patch_.cols / 2.0) / (patch_.cols / 2.0);
    y = (y - patch_.rows / 2.0) / (patch_.rows / 2.0);
    
    adjusted_anno[row](0) = x;
    adjusted_anno[row](1) = y;
    adjusted_anno[row](2) = z;
  }
  return adjusted_anno;
}

template <typename Dtype>
std::vector< cv::Vec3f > HandPatch<Dtype>::uncenterAnno(const std::vector< cv::Vec3f >& anno) const {
  std::vector<cv::Vec3f> adjusted_anno(anno.size());
  for(int row = 0; row < anno.size(); ++row) {
    Dtype x = anno[row](0);
    Dtype y = anno[row](1);
    Dtype z = anno[row](2);
    
    x = x * (patch_.cols / 2.0) + patch_.cols / 2.0;
    y = y * (patch_.rows / 2.0) + patch_.rows / 2.0;
    
    adjusted_anno[row](0) = x;
    adjusted_anno[row](1) = y;
    adjusted_anno[row](2) = z;
  }
  return adjusted_anno;
}


template <typename Dtype>
bool HandPatch<Dtype>::isReasonableCenteredAnno(const std::vector< cv::Vec3f >& anno, bool normalized) const {
  for(size_t anno_idx = 0; anno_idx < anno.size(); ++anno_idx) {
    cv::Vec3f a = anno[anno_idx];
    Dtype x = a(0);
    Dtype y = a(1);
    Dtype z = a(2);
    
    if(x < -1.2 || x > 1.2 || y < -1.2 || y > 1.2 || (normalized && z < -5) || (normalized && z > 3.2)) {
      return false;
    }
  }
  
  return true;
}


template class HandPatch<float>;
template class HandPatch<double>;
