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
