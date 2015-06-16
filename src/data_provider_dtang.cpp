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

#include "data_provider_dtang.h"
#include "common.h"

#include <rv/io/ls.h>
#include <rv/ocv/io/csv.h>


template <typename Dtype>
DataProviderDTang<Dtype>::DataProviderDTang(const pose::DataProviderParameter& param) : DataProvider<Dtype>(param) {
  std::cout << "[INFO] read label file " << param.dtang_param().label_path() << std::endl;
  
  boost::filesystem::path label_path = param.dtang_param().label_path();
  
  std::ifstream label_file(label_path.c_str());
  std::string line;
  
  int idx = 0;
  
  while(std::getline(label_file, line)) {
    std::istringstream iss(line);
    
    std::string relative_depth_path;
    iss >> relative_depth_path;

    boost::filesystem::path depth_path = label_path.parent_path() / "Depth" / 
      boost::filesystem::path(relative_depth_path);
    
    std::vector<cv::Vec3f> anno(param.n_pts());
    for(int pt_idx = 0; pt_idx < anno.size(); ++pt_idx) {
      iss >> anno[pt_idx](0);
      iss >> anno[pt_idx](1);
      iss >> anno[pt_idx](2);
    }
    
    //only load original data - no rotated
    if(depth_path.parent_path().parent_path().filename().string() == "Depth" && (idx % this->param_.inc() == 0)) {
      depth_paths_.push_back(depth_path);
      annos_.push_back(anno);
    }
    
    idx++;
  }
  
  this->max_idx_ = depth_paths_.size();
  
  label_file.close();
  
  std::cout << "[INFO] label file " << label_path << " contained " << depth_paths_.size() << " annotated depth images" << std::endl;       
}

template <typename Dtype>
boost::filesystem::path DataProviderDTang<Dtype>::depthPath(int idx) const {
    return depth_paths_[idx];
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderDTang<Dtype>::depth_internal(int idx) {
    std::string depth_path_str = depth_paths_[idx].string();
    cv::Mat depthIn = cv::imread(depth_path_str, CV_LOAD_IMAGE_ANYDEPTH);
    cv::Mat_<Dtype> depth;
    
    depthIn.convertTo(depth, depth.type());
    
    Dtype bg_value = this->param_.bg_value();
    
    for(int row = 0; row < depth.rows; ++row) {
        for(int col = 0; col < depthIn.cols; ++col) {            
            if(depth(row, col) > bg_value) {
                depth(row, col) = bg_value;
            }
        }
    }

    return depth;
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderDTang<Dtype>::ir_internal(int idx) {
    return depth_internal(idx);
}

template <typename Dtype>
std::vector< cv::Vec3f > DataProviderDTang<Dtype>::gt_internal(int idx) const {
    return annos_[idx];
}

template <typename Dtype>
cv::Vec3f DataProviderDTang<Dtype>::hint2d_internal(int idx) const {
    return annos_[idx][0];
}


template <typename Dtype>
void DataProviderDTang<Dtype>::shuffle() {
  static cv::RNG rng(time(0));
                                  
  std::vector<boost::filesystem::path> shuffled_depth_paths(depth_paths_.size());
  std::vector<std::vector<cv::Vec3f> > shuffled_annos(annos_.size());
  
  for(size_t idx = 0; idx < depth_paths_.size(); ++idx) {
    int rand_idx = rng.uniform(0, depth_paths_.size());
    shuffled_depth_paths[idx] = depth_paths_[rand_idx];
    shuffled_annos[idx] = annos_[rand_idx];
  }
  
  depth_paths_ = shuffled_depth_paths;
  annos_ = shuffled_annos;
}


template class DataProviderDTang<float>;
template class DataProviderDTang<double>;
