#ifndef INFERENCE_REGRESSION_NET_H
#define INFERENCE_REGRESSION_NET_H

#include "inference.h"

#include <caffe/common.hpp>
#include <caffe/net.hpp>

template <typename Dtype>
class InferenceRegressionNet : public Inference<Dtype> {
public:
    InferenceRegressionNet(boost::shared_ptr<caffe::Net<Dtype> > net) : 
        net_(net) {}
    virtual ~InferenceRegressionNet() {}
    
    virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const;
    
private:
    boost::shared_ptr<caffe::Net<Dtype> > net_;
};

#endif