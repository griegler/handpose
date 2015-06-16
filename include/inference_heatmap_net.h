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

#ifndef INFERENCE_HEATMAP_NET_H
#define INFERENCE_HEATMAP_NET_H

#include "inference.h"

#include <caffe/common.hpp>
#include <caffe/net.hpp>

#include "pose.pb.h"

template <typename Dtype>
class InferenceHeatmapNet : public Inference<Dtype> {
public:
    InferenceHeatmapNet(boost::shared_ptr<caffe::Net<Dtype> > net, const pose::NNHeatmapParameter& param, int n_pts) : 
        net_(net), param_(param), n_pts_(n_pts) {}
    virtual ~InferenceHeatmapNet() {}
    
    virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const;
    
protected:
  
  virtual std::vector<cv::Vec3f> inference2d() const;
  
  virtual void inferenceD(std::vector<cv::Vec3f>& es) const;
  virtual void patchFitD(const std::vector<HandPatch<Dtype> >& patches, std::vector<cv::Vec3f>& es) const;
  
    
private:    
    boost::shared_ptr<caffe::Net<Dtype> > net_;
    const pose::NNHeatmapParameter& param_;
    int n_pts_;
    
};

#endif
