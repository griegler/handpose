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

#include "hand_segmentation_rf.h"

#include <rv/ocv/convert.h>
#include <rv/ocv/linalg.h>
#include <rv/ocv/colormap/colormap.h>
#include <rv/ocv/colormap/colormap_cool_warm.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

template <typename Dtype>
std::vector<HandSegmentationResult> HandSegmentationRf<Dtype>::operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const {    
  cv::Mat_<Dtype> fg_prob = cv::Mat_<Dtype>::zeros(depth.rows, depth.cols);
  cv::Mat_<uchar> mask = cv::Mat_<uchar>::zeros(depth.rows, depth.cols);

  int n_found = 0;
  
  for(int row = 0; row < depth.rows; ++row) {
    for(int col = 0; col < depth.cols; ++col) {
      Dtype d = depth(row, col);
      if(d < bg_value_ && d > 0) {
        rv::rf::SamplePtr sample = boost::make_shared<rv::rf::ImgReplicateBorderSample>(depth, col, row);
        rv::rf::VecPtrTargetPtr estimated_targets = forest_->inferencest(sample);
        rv::rf::RfMatPtr es_mat = (*estimated_targets)[0]->data();
        float p = (*es_mat)(0, 1);
        
        fg_prob(row, col) = p;
        
        if(p > this->param_.rf_param().min_prob()) {
          mask(row, col) = 1;
          n_found++;
        }
      }
    }
  }
  
  if(n_found < 10) {
    std::cout << "[INFO] only 10 pixels classified as hand" << std::endl;
    return std::vector<HandSegmentationResult>();
  }
  
  static rv::ocv::ColorMap<Dtype>& cmap = rv::ocv::ColorMapCoolWarm<Dtype>::i();
  // cv::imshow("depth", rv::ocv::clamp(depth));
  cv::imshow("bg_prob", cmap.Map(1 - fg_prob));
  cv::imshow("fg_prob", cmap.Map(fg_prob));
  cv::imshow("mask", mask * 255);  
  
  //clean up mask
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, struc_elem_);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, struc_elem_);
  
  cv::imshow("mask2", mask * 255);  
  // cv::waitKey(0);
  
  
  //find roi
  int min_y = mask.rows;
  int max_y = 0;
  int min_x = mask.cols;
  int max_x = 0;
  
  for(int row = 0; row < mask.rows; ++row) {
    for(int col = 0; col < mask.cols; ++col) {
      if(mask(row, col) > 0) {
        if(row < min_y) min_y = row;
        if(row > max_y) max_y = row;
        if(col < min_x) min_x = col;
        if(col > max_x) max_x = col;
      }
    }
  }
  
  cv::Rect roi(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  
  
  //return result
  std::vector<HandSegmentationResult> result(1);
  result[0].roi_ = roi;
  result[0].mask_ = mask;
  
  return result;
}




template <typename Dtype>
rv::rf::ForestPtr HandSegmentationRf<Dtype>::train(    
  boost::shared_ptr<DataProvider<Dtype> > data_provider, 
  boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation,
  const rv::rf::ForestParameter& params) {
  
  data_provider->shuffle();
  
  const int pos_samples_p_image = 120;
  const int neg_samples_p_image = 60;
  
  const float pos_samples_weight = float(neg_samples_p_image) / float(pos_samples_p_image + neg_samples_p_image);
  const float neg_samples_weight = float(pos_samples_p_image) / float(pos_samples_p_image + neg_samples_p_image);
  
  //create samples
  std::vector<rv::rf::SamplePtr> samples;
  rv::rf::VecPtrTargetPtr targets = boost::make_shared<rv::rf::VecTargetPtr>();
  
  for(; data_provider->hasNext(); data_provider->next()) {
    std::cout << "extract template from: " << data_provider->depthPath() << std::endl;
    
    cv::Mat_<Dtype> depth = data_provider->depth();
    cv::Mat_<Dtype> ir = data_provider->ir();
    cv::Vec3f hint_2d = data_provider->hint2d();        
    
    std::vector<HandSegmentationResult> segmentations = (*hand_segmentation)(depth, ir, hint_2d);
    
    for(int segmentation_idx = 0; segmentation_idx < segmentations.size(); ++segmentation_idx) {
      cv::Mat_<uchar> mask = segmentations[segmentation_idx].mask_;
      
      int n_neg_samples = 0;
      int n_pos_samples = 0;
      while(n_neg_samples < neg_samples_p_image) {
        int x = rv::rand::Rand<Dtype>::Uniform(0, depth.cols);
        int y = rv::rand::Rand<Dtype>::Uniform(0, depth.rows);
        
        if(mask(y, x) == 0 && depth(y, x) < data_provider->bgValue()) {
          samples.push_back(boost::make_shared<rv::rf::ImgReplicateBorderSample>(depth, x, y));
          targets->push_back(boost::make_shared<rv::rf::Target>(0, 2, neg_samples_weight));
          n_neg_samples++;
        }
      }
      
      while(n_pos_samples < pos_samples_p_image) {
        int x = rv::rand::Rand<Dtype>::Uniform(0, depth.cols);
        int y = rv::rand::Rand<Dtype>::Uniform(0, depth.rows);
        
        if(mask(y, x) > 0) {
          samples.push_back(boost::make_shared<rv::rf::ImgReplicateBorderSample>(depth, x, y));
          targets->push_back(boost::make_shared<rv::rf::Target>(1, 2, pos_samples_weight));
          n_pos_samples++;
        }
      }
    }
  }

  std::vector<rv::rf::VecPtrTargetPtr> vec_targets;
  vec_targets.push_back(targets);
  
  std::cout << "n samples: " << samples.size() << std::endl;
  std::cout << "n targets: " << targets->size() << std::endl;
  
  rv::rf::TrainForest rf_train(params, true);
  rv::rf::ForestPtr forest;
  forest = rf_train.Train(samples, vec_targets, vec_targets, rv::rf::TRAIN, forest);
  
  return forest;
}


template class HandSegmentationRf<float>;
