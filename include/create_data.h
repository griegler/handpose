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

#ifndef CREATE_DATA_H
#define CREATE_DATA_H

#include "data_provider.h"
#include "hand_segmentation.h"
#include "hand_patch.h"
#include "patch_extraction.h"
#include "pose.pb.h"

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rv/io/h5_block_writer.h>

template <typename Dtype>
class CreateData {
public:
  CreateData(const pose::CreateDataParameter& param,
    DataProvider<Dtype>& data_provider, 
    const HandSegmentation<Dtype>& hand_segmentation, 
    const PatchExtraction<Dtype>& patch_extraction) :
      param_(param), data_provider_(data_provider), 
      hand_segmentation_(hand_segmentation), 
      patch_extraction_(patch_extraction) {
        
    createRotations();
  }
    
  virtual ~CreateData() {};

  virtual void data2Hdf5() const;    
  
  
  virtual void mirror(const cv::Mat_<Dtype>& depth, 
      const cv::Mat_<Dtype>& ir, const std::vector<cv::Vec3f>& anno,
      cv::Mat_<Dtype>& mirrored_depth, cv::Mat_<Dtype>& mirrored_ir,
      std::vector<cv::Vec3f>& mirrored_anno) const;
  virtual void rotate(const cv::Mat_<Dtype>& depth, 
      const cv::Mat_<Dtype>& ir, const std::vector<cv::Vec3f>& anno,
      cv::Mat_<Dtype>& rotated_depth, cv::Mat_<Dtype>& rotated_ir,
      std::vector<cv::Vec3f>& rotated_anno, float rot_deg) const;
    
protected:
  virtual void createRotations();
                        
protected:
  const pose::CreateDataParameter& param_; 
  DataProvider<Dtype>& data_provider_;
  const HandSegmentation<Dtype>& hand_segmentation_;
  const PatchExtraction<Dtype>& patch_extraction_;
  
  std::vector<float> rotations_;
};




#endif
