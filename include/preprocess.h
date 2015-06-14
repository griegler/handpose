#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "hand_patch.h"

#include <opencv2/core/core.hpp>

template <typename Dtype>
class Preprocess {
public:
    Preprocess() : subtract_mean_(false) {}
    virtual ~Preprocess() {}
    
    virtual void operator()(std::vector<HandPatch<Dtype> >& hand_patch) const;
    
    virtual void addSubtractMean(cv::Mat_<Dtype>& mean) {
        subtract_mean_ = true;
        mean_ = mean;
    }
    
private:
    bool subtract_mean_;
    cv::Mat_<Dtype> mean_;
};

#endif