#ifndef INFERENCE_H
#define INFERENCE_H

#include "hand_segmentation.h"
#include "hand_patch.h"

template <typename Dtype>
class Inference {
public:
    Inference() {}
    virtual ~Inference() {}
    
    virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const = 0;
    
protected:
};

#endif