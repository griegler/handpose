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

#ifndef CREATE_TEST_PATCH_H
#define CREATE_TEST_PATCH_H

#include "inference_heatmap_net.h"
#include "inference_regression_net.h"
#include "inference_rf_simple.h"
#include "inference_dummy.h"

#include "pose.pb.h"

#include <boost/make_shared.hpp>

template <typename Dtype>
boost::shared_ptr< Inference< Dtype > > makeInference(const pose::InferenceParameter& param, int n_pts) {
    boost::shared_ptr<Inference<Dtype> > inference;
    
    if(param.type() == pose::InferenceParameter_InferenceType_RF_SIMPLE_REGRESSION) {
      std::cout << "[INFO] Creating InferenceRfSimple" << std::endl;
      inference = boost::make_shared<InferenceRfSimple<Dtype> >(param.rf_param());
    }
    else if(param.type() == pose::InferenceParameter_InferenceType_NN_REGRESSION) {
      std::cout << "[INFO] Creating InferenceRegressionNet" << std::endl;
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      // caffe::Caffe::set_mode(caffe::Caffe::CPU);
      
      std::string net_path = param.nn_regression_param().net_path();
      std::string weights_path = param.nn_regression_param().weights_path();
      boost::shared_ptr<caffe::Net<Dtype> > net = boost::make_shared<caffe::Net<Dtype> >(net_path, caffe::TEST);
      net->CopyTrainedLayersFrom(weights_path);
      std::cout << "caffe::mode: " << caffe::Caffe::mode() << " CPU: " << caffe::Caffe::CPU << "/GPU: " << caffe::Caffe::GPU << std::endl;

      inference = boost::make_shared<InferenceRegressionNet<Dtype> >(net);
    }
    else if(param.type() == pose::InferenceParameter_InferenceType_NN_HEATMAP) {
      std::cout << "[INFO] Creating InferenceHeatmapNet" << std::endl;
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
      // caffe::Caffe::set_mode(caffe::Caffe::CPU);

      std::string net_path = param.nn_heatmap_param().net_path();
      std::string weights_path = param.nn_heatmap_param().weights_path();
      boost::shared_ptr<caffe::Net<Dtype> > net = boost::make_shared<caffe::Net<Dtype> >(net_path, caffe::TEST);
      net->CopyTrainedLayersFrom(weights_path);
      std::cout << "caffe::mode: " << caffe::Caffe::mode() << " CPU: " << caffe::Caffe::CPU << "/GPU: " << caffe::Caffe::GPU << std::endl;

      inference = boost::make_shared<InferenceHeatmapNet<Dtype> >(net, param.nn_heatmap_param(), n_pts);
    }
    else {
      std::cout << "[INFO] Make InferenceDummy" << std::endl;
      inference = boost::make_shared<InferenceDummy<Dtype> >(n_pts);
    }
    
    return inference;
}

#endif
