#ifndef MAKE_HAND_SEGMENTATION_H
#define MAKE_HAND_SEGMENTATION_H

#include "hand_segmentation.h"

#include "hand_segmentation_meanshift.h"
#include "hand_segmentation_rf.h"
#include "hand_segmentation_threshold.h"

#include "pose.pb.h"

#include <boost/make_shared.hpp>


template <typename Dtype>
boost::shared_ptr<HandSegmentation<Dtype> > makeHandSegmentation(const pose::SegmentationParameter& param, 
                                                                 const Projection& projection,
                                                                 Dtype bg_value) {
    boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation;
    
    if(param.type() == pose::SegmentationParameter_SegmentationType_MEANSHIFT) {
        std::cout << "[INFO] Creating HandSegmentationMeanshift" << std::endl;
        hand_segmentation = boost::make_shared<HandSegmentationMeanshift<Dtype> >(param, projection, bg_value);
    }
    else if(param.type() == pose::SegmentationParameter_SegmentationType_THRESHOLD) {
        std::cout << "[INFO] Creating HandSegmentationThreshold" << std::endl;
        hand_segmentation = boost::make_shared<HandSegmentationThreshold<Dtype> >(param, projection, bg_value);
    }
    else if(param.type() == pose::SegmentationParameter_SegmentationType_RF) {
        std::cout << "[INFO] Creating HandSegmentationRf" << std::endl;
        hand_segmentation = boost::make_shared<HandSegmentationRf<Dtype> >(param, projection, bg_value);
    }
    else {
      throw std::runtime_error("[ERROR] unknown hand segmentation");
    }
    
    return hand_segmentation;
}

#endif