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

#ifndef DATA_PROVIDER_BLENDER_H
#define DATA_PROVIDER_BLENDER_H

#include <data_provider.h>

#include "pose.pb.h"

template <typename Dtype>
class DataProviderBlender : public DataProvider<Dtype> {
public:
  DataProviderBlender(const pose::DataProviderParameter& param);
  ~DataProviderBlender() {}
  
  virtual boost::filesystem::path depthPath(int idx) const;
  
  virtual void shuffle();

protected:
  virtual std::vector< cv::Vec3f > gt_internal(int idx) const;
  virtual cv::Vec3f hint2d_internal(int idx) const;
  virtual cv::Mat_< Dtype > depth_internal(int idx);
  virtual cv::Mat_< Dtype > ir_internal(int idx);

  boost::filesystem::path annoPath(const boost::filesystem::path& depth_path) const;
  
private:  
  std::vector<boost::filesystem::path> depth_paths_;
  std::vector<std::vector<cv::Vec3f> > annos_;
};

#endif
