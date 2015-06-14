#ifndef INFERENCE_DUMMY_H
#define INFERENCE_DUMMY_H

#include "inference.h"

template <typename Dtype>
class InferenceDummy : public Inference<Dtype> {
public:
    InferenceDummy(int n_pts) : n_pts_(n_pts) {}
    virtual ~InferenceDummy() {}
    
    virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const {
      std::vector<cv::Vec3f> result(n_pts_, cv::Vec3f(0.0f, 0.0f, 0.0f));
      return result;
    }
    
private:
  int n_pts_;
};

#endif