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

#ifdef BUILD_DATA_PROVIDER_CAM

#include "data_provider_cam.h"
#include "common.h"

#include <rv/io/ls.h>
#include <rv/ocv/io/csv.h>
#include <rv/camera/creative.h>
#include <rv/camera/pmd.h>

#include <boost/make_shared.hpp>

#include <limits>


template <typename Dtype>
DataProviderCam<Dtype>::DataProviderCam(const pose::DataProviderParameter& param) : DataProvider<Dtype>(param) {

  if(param.cam_param().type() == pose::CamParameter::NONE) {
    this->max_idx_ = 0;
  }
  else if(param.cam_param().type() == pose::CamParameter::CREATIVE) {
    rv::camera::Creative325Camera& cam = rv::camera::Creative325Camera::i();
    cam_ = &cam;
    delete_cam_ = false;
    this->max_idx_ = std::numeric_limits<int>::max();
    im_type_depth_ = rv::camera::Creative325Camera::typeDepth();
    im_type_ir_ = rv::camera::Creative325Camera::typeIr();
  }
  else if(param.cam_param().type() == pose::CamParameter::PMD_PICO) {
    cam_ = new rv::camera::PmdPicoCamera();
    delete_cam_ = true;
    this->max_idx_ = std::numeric_limits<int>::max();
    im_type_depth_ = rv::camera::PmdPicoCamera::typeDepth();
    im_type_ir_ = rv::camera::PmdPicoCamera::typeIr();
  }
  
  if(cam_ != 0) { 
    cam_->registerType(im_type_depth_);
    cam_->registerType(im_type_ir_);
     
    cam_->start();
  }
}

template <typename Dtype>
DataProviderCam<Dtype>::~DataProviderCam() {
  if(cam_ != 0) {
    cam_->stop();
    if(delete_cam_) {
      delete cam_;
    }
  }
}

template <typename Dtype>
boost::filesystem::path DataProviderCam<Dtype>::depthPath(int idx) const {
  return "";
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderCam<Dtype>::depth_internal(int idx) {
  rv::camera::CameraImage img_depth;
  cam_->read(im_type_depth_, img_depth);
  depth_ = img_depth.im_;
  
  rv::camera::CameraImage img_ir;
  cam_->read(im_type_ir_, img_ir);
  ir_ = img_ir.im_;

  
  Dtype bg_value = this->param_.bg_value();
  int median_ksize = 5;
  float min_ir = 150;
  
  for(int row = 0; row < depth_.rows; ++row) {
    for(int col = 0; col < depth_.cols; ++col) {
      Dtype d = depth_(row, col);
      Dtype i = ir_(row, col);
      
      if(d < 0 || d > bg_value || i < min_ir) {
        depth_(row, col) = bg_value;
      }
    }
  }
  
  if(median_ksize > 0) {
    cv::medianBlur(depth_, depth_, median_ksize);
  }

  return depth_;
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderCam<Dtype>::ir_internal(int idx) {
  return ir_;  
}

template <typename Dtype>
std::vector< cv::Vec3f > DataProviderCam<Dtype>::gt_internal(int idx) const {
  return std::vector<cv::Vec3f>(this->param_.n_pts(), cv::Vec3f(0, 0, 0));
}

template <typename Dtype>
cv::Vec3f DataProviderCam<Dtype>::hint2d_internal(int idx) const {
  return cv::Vec3f(0, 0, 0);
}

template class DataProviderCam<float>;
template class DataProviderCam<double>;


#endif //BUILD_DATA_PROVIDER_CAM
