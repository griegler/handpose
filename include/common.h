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

#ifndef COMMON_H
#define COMMON_H

#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <rv/ocv/color.h>
#include <rv/ocv/convert.h>
#include <rv/ocv/visualize.h>
#include <rv/ocv/colormap/colormap.h>
#include <rv/ocv/colormap/colormap_cool_warm.h>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

#include "hand_patch.h"

#include "pose.pb.h"


void depthmap2pts(const cv::Mat_<float>& dm, const cv::Mat_<float>& ir,
    const cv::Mat_<cv::Vec3b>& img, const float& max_d, const float& min_ir,
    std::vector<cv::Vec3f>& pts, std::vector<cv::Vec3f>& colors);

std::vector<cv::Vec3f> readAnno(boost::filesystem::path anno_path, 
    int n_joints);

void writePoints(std::ostream& out, const std::vector<cv::Vec3f>& pts);

cv::Mat_<float> readExrDepth(const boost::filesystem::path& p);

bool readProtoFromTextFile(const boost::filesystem::path& path, 
    google::protobuf::Message* proto);

//-----------------------------------------------------------------------------
void getIdTs(boost::filesystem::path depth_map_path, std::string& id, 
    std::string& ts);
std::string getAnnotator(boost::filesystem::path anno_path);
std::vector<boost::filesystem::path> lsAnnoPaths(
    boost::filesystem::path depth_map_path);
boost::filesystem::path infraredPath(boost::filesystem::path depth_map_path);
boost::filesystem::path blenderRgbPath(boost::filesystem::path depth_map_path);


cv::Rect enclosingRect(const std::vector<cv::Vec3f>& pts);


template <typename Dtype>
cv::Mat_<cv::Vec3b> annoShow(const cv::Mat_<Dtype>& im, 
    const std::vector<cv::Vec3f>& anno, Dtype bg_value) {
  std::vector<cv::Scalar> colors = rv::ocv::hsv(anno.size());
  static rv::ocv::ColorMap<Dtype>& cmap = rv::ocv::ColorMapCoolWarm<Dtype>::i();
  
  std::vector<cv::Point> pts(anno.size());
  for(int idx = 0; idx < pts.size(); ++idx) {
    pts[idx].x = anno[idx](0);
    pts[idx].y = anno[idx](1);
  }
  cv::Mat_<cv::Vec3b> im3 = cmap.Map(im, 0, bg_value);
  if(pts.size() > 0) {
    rv::ocv::drawPoints(im3, pts, colors);
  }

  return im3;
}

template <typename Dtype>
cv::Mat_<cv::Vec3b> handPatchShow(const HandPatch<Dtype>& hp, 
    const std::vector<cv::Vec3f>& anno) {
  std::vector<cv::Scalar> colors = rv::ocv::hsv(anno.size());
  static rv::ocv::ColorMap<Dtype>& cmap = rv::ocv::ColorMapCoolWarm<Dtype>::i();

  std::vector<cv::Vec3f> adjusted_anno;
  if(anno.size() > 0) {
    adjusted_anno = hp.uncenterAnno(anno);
  }

  std::vector<cv::Point> pts(anno.size());
  for(int idx = 0; idx < pts.size(); ++idx) {
    pts[idx].x = adjusted_anno[idx](0);
    pts[idx].y = adjusted_anno[idx](1);
  }

  cv::Mat_<cv::Vec3b> im3 = cmap.Map(hp.patch_, -3, 3);
  if(pts.size() > 0) {
    rv::ocv::drawPoints(im3, pts, colors, 2);
  }

  return im3;
}

#endif
