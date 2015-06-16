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

#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rv/ocv/colormap/colormap_cool_warm.h>

#include <boost/filesystem.hpp>
#include <pose.pb.h>
#include "make_projection.h"

template <typename Dtype>
class DataProvider {
public:
  DataProvider(const pose::DataProviderParameter& param) : 
      idx_(0), max_idx_(0), param_(param) {
  }

  virtual ~DataProvider() {}
  
  virtual bool hasNext() { return idx_ < max_idx_; }
  virtual void next() { idx_++; }
  virtual void reset() { idx_ = 0; }
  
  virtual boost::filesystem::path depthPath(int idx) const = 0;

  virtual boost::filesystem::path depthPath() const { return depthPath(idx_); }
  virtual cv::Mat_<Dtype> depth() { return depth(idx_); }
  virtual cv::Mat_<Dtype> ir() { return ir(idx_); }
  virtual std::vector<cv::Vec3f> gt() { return gt(idx_); }
  virtual cv::Vec3f hint2d() { return hint2d(idx_); }
  
  virtual cv::Mat_<Dtype> depth(int idx) {
    cv::Mat_<Dtype> img = depth_internal(idx);
    cv::Mat_<Dtype> img_und;
    if(param_.undistort()) {
      createProjection();
      cv::remap(img, img_und, undist_map1_, undist_map2_, cv::INTER_NEAREST, cv::BORDER_CONSTANT, bgValue());

      // static rv::ocv::ColorMap<Dtype>& cmap = rv::ocv::ColorMapCoolWarm<Dtype>::i();
      // cv::imshow("input", cmap.Map(img));
      // cv::imshow("undistort", cmap.Map(img_und));
      // cv::waitKey(0);
    }
    else {
      img_und = img;
    }

    if(this->param_.flip()) {
      cv::flip(img_und, img_und, 1);
    }

    return img_und;
  }

  virtual cv::Mat_<Dtype> ir(int idx) {
    cv::Mat_<Dtype> img = ir_internal(idx);
    cv::Mat_<Dtype> img_und;
    if(param_.undistort()) {
      createProjection();
      cv::remap(img, img_und, undist_map1_, undist_map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    }
    else {
      img_und = img;
    }

    if(this->param_.flip()) {
      cv::flip(img_und, img_und, 1);
    }

    return img_und;
  }

  virtual std::vector<cv::Vec3f> gt(int idx) {
    std::vector<cv::Vec3f> pts = gt_internal(idx);
    if(param_.undistort()) {
      createProjection();
      cv::Mat_<cv::Vec2f> pts2(pts.size(), 1);
      for(int pts_idx = 0; pts_idx < pts.size(); ++pts_idx) {
        pts2(pts_idx, 0)(0) = pts[pts_idx][0];
        pts2(pts_idx, 0)(1) = pts[pts_idx][1];
      }

      cv::Mat_<cv::Vec2f> pts2_und(pts.size(), 1);
      cv::Matx33d K = projection_->K();
      cv::undistortPoints(pts2, pts2_und, K, projection_->d(), cv::noArray(), K);

      for(int pts_idx = 0; pts_idx < pts.size(); ++pts_idx) {
        pts[pts_idx](0) = pts2_und(pts_idx, 0)(0);
        pts[pts_idx](1) = pts2_und(pts_idx, 0)(1);
      }
    }

    return pts;
  }

  virtual cv::Vec3f hint2d(int idx) {
    cv::Vec3f pt = hint2d_internal(idx);
    if(param_.undistort()) {
      createProjection();

      cv::Mat_<cv::Vec2f> pts2(1, 1);
      pts2(0, 0)(0) = pt[0];
      pts2(0, 0)(1) = pt[1];

      cv::Mat_<cv::Vec2f> pts2_und(1, 1);
      cv::Matx33d K = projection_->K();
      cv::undistortPoints(pts2, pts2_und, K, projection_->d(), cv::noArray(), K);

      pt(0) = pts2_und(0, 0)(0);
      pt(1) = pts2_und(0, 0)(1);
    }

    return pt;
  }
  
  virtual int nPts() const { return param_.n_pts(); }
  virtual Dtype bgValue() const { return param_.bg_value(); }
  
  virtual void shuffle() = 0;
  
  virtual int maxIdx() const { return max_idx_; }

  virtual boost::shared_ptr<Projection> projection() { 
    createProjection();
    return projection_; 
  }
    
protected:
  virtual cv::Mat_<Dtype> depth_internal(int idx) = 0;
  virtual cv::Mat_<Dtype> ir_internal(int idx) = 0;
  virtual std::vector<cv::Vec3f> gt_internal(int idx) const = 0;
  virtual cv::Vec3f hint2d_internal(int idx) const = 0;


  virtual void createProjection() {
    if(projection_ == 0) {
      cv::Mat_<float> img = depth_internal(0);
      projection_ = makeProjection(param_.projection_param(), depthPath());

      cv::Matx33d K = projection_->K();
      cv::Matx18d d = projection_->d();

      std::cout << "K = " << std::endl << K << std::endl;
      std::cout << "d = " << std::endl << d << std::endl;

      cv::initUndistortRectifyMap(K, d, cv::Mat_<float>::eye(3, 3), K, img.size(), CV_16SC2, undist_map1_, undist_map2_);
    }
  }

  virtual std::vector<boost::filesystem::path> sortReducePaths(std::vector<boost::filesystem::path>& paths) {
    std::sort(paths.begin(), paths.end());
    
    //mind inc parameter
    int inc = param_.inc();
    std::vector<boost::filesystem::path> reduced_paths;
    for(int idx = 0; idx < paths.size(); idx = idx + inc) {
      reduced_paths.push_back(paths[idx]);
    }
    
    return reduced_paths;
  }
  
  virtual void stat(const std::vector<std::vector<cv::Vec3f> >& anno, 
                    cv::Vec3f& mean, cv::Vec3f& std, 
                    cv::Vec3f& mean_0, cv::Vec3f& std_0) {
    mean = 0;
    std = 0;
    size_t n = 0;
    
    mean_0 = 0;
    std_0 = 0;
    
    //mean
    for(size_t anno_idx = 0; anno_idx < anno.size(); ++anno_idx) {
      mean_0 += anno[anno_idx][0];
      
      for(size_t joint_idx = 0; joint_idx < anno[anno_idx].size(); ++joint_idx) {
        mean += anno[anno_idx][joint_idx];
        n++;
      }
    }
    mean_0 = mean_0 / double(anno.size());
    mean = mean / double(n);
    
    //std
    for(size_t anno_idx = 0; anno_idx < anno.size(); ++anno_idx) {
      cv::Vec3f d_0 = anno[anno_idx][0] - mean_0;
      std_0 += cv::Vec3f(d_0(0) * d_0(0), d_0(1) * d_0(1), d_0(2) * d_0(2));
      
      for(size_t joint_idx = 0; joint_idx < anno[anno_idx].size(); ++joint_idx) {
        cv::Vec3f d = anno[anno_idx][0] - mean_0;
        std += cv::Vec3f(d(0) * d(0), d(1) * d(1), d(2) * d(2));
      }
    }
    std_0 = std_0 / double(anno.size() - 1);
    std = std / double(n - 1);
    
    for(int i = 0; i < 3; ++i) {
      std_0(i) = std::sqrt(std_0(i));
      std(i) = std::sqrt(std(i));
    }
  }
    
protected:
  boost::shared_ptr<Projection> projection_;
  cv::Mat undist_map1_;
  cv::Mat undist_map2_;

  int idx_;
  int max_idx_;
  
  const pose::DataProviderParameter& param_;
};

#endif // DATAPROVIDER_H
