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
