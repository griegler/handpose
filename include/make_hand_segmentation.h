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
