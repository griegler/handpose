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

#include "inference_rf_simple.h"

template <typename Dtype> 
std::vector< cv::Vec3f > InferenceRfSimple<Dtype>::operator()(const std::vector< HandPatch< Dtype > >& patches) const {
   
  cv::Mat_<float> data = patches[0].patch_;
  int data_dim = data.rows * data.cols;
  
  rv::rf::RfMatPtr patch = rv::rf::CreateRfMat(data_dim, 1);
  int idx = 0;
  for(int w = 0; w < data.cols; ++w) {
    for(int h = 0; h < data.rows; ++h) {
      (*patch)(idx, 0) = data(h, w);
      idx++;
    }
  }

  if(T_ != 0) {
    *patch = (*T_) * (*patch);
  }

  rv::rf::SamplePtr sample = boost::make_shared<rv::rf::MatSample>(patch);
  rv::rf::VecPtrTargetPtr estimated_targets = forest_->inferencemt(sample);
  rv::rf::RfMatPtr result = (*estimated_targets)[0]->data();

  int result_blob_data_idx = 0;
  std::vector<cv::Vec3f> es(result->rows() / 3);
  for(int row = 0; row < result->rows() / 3; ++row, result_blob_data_idx += 3) {
    Dtype x = (*result)(result_blob_data_idx, 0);
    Dtype y = (*result)(result_blob_data_idx + 1, 0);
    Dtype z = (*result)(result_blob_data_idx + 2, 0);

    es[row](0) = x;
    es[row](1) = y;
    es[row](2) = z;

    // std::cout << es[row] << std::endl;
  }

  return es;
}


template class InferenceRfSimple<float>;
template class InferenceRfSimple<double>;
