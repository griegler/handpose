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

#ifndef INFERENCE_RF_SIMPLE_H
#define INFERENCE_RF_SIMPLE_H

#include "inference.h"

#include "pose.pb.h"

#include <rv/ml/rf/forest.h>
#include <rv/ml/rf/data/matsample.h>
#include <rv/io/serialization/compressed_serialization.h>
#include <rv/eigen/io/csv.h>

template <typename Dtype>
class InferenceRfSimple : public Inference<Dtype> {
public:
  InferenceRfSimple(const pose::RFParameter& param) {
    rv::io::CompressedSerializationIn serialization(param.forest_path());
    forest_ = boost::make_shared<rv::rf::Forest>();
    forest_->Load(serialization);

    if(param.transform_csv() != "") {
      T_ = rv::rf::CreateRfMat(0, 0);
      rv::eigen::ReadCsv(param.transform_csv(), *T_);
    }
  }

  virtual ~InferenceRfSimple() {}
  
  virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const;
private:
  rv::rf::ForestPtr forest_;
  rv::rf::RfMatPtr T_;
};

#endif
