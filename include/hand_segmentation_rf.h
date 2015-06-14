#ifndef HAND_SEGMENTATION_RF_H
#define HAND_SEGMENTATION_RF_H

#include "hand_segmentation.h"
#include "data_provider.h"
#include "projection.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <rv/ml/rf/train/traintreerecursive.h>
#include <rv/ml/rf/train/trainforest.h>
#include <rv/ml/rf/splitevaluator/classificationsplitevaluator.h>
#include <rv/ml/rf/splitfunction/splitfunctionpixelvalue.h>
#include <rv/ml/rf/splitfunction/splitfunctionpixeldifference.h>
#include <rv/ml/rf/splitfunction/splitfunctionpixeldifferencedependent.h>
#include <rv/ml/rf/leafnodefcn/classificationleafnodefcn.h>
#include <rv/ml/rf/data/imgsample.h>
#include <rv/io/serialization/compressed_serialization.h>

template <typename Dtype>
class HandSegmentationRf : public HandSegmentation<Dtype> {
public:
    HandSegmentationRf(const pose::SegmentationParameter& param, const Projection& projection, Dtype bg_value) : 
        HandSegmentation<Dtype>(param), bg_value_(bg_value) {
      //create and load forest
      forest_ = boost::make_shared<rv::rf::Forest>();
      
      std::string forest_path = param.rf_param().forest_path();
      if(boost::filesystem::exists(forest_path)) {
        rv::io::CompressedSerializationIn serialization(param.rf_param().forest_path());
        forest_->Load(serialization);
      }
      else {
        throw std::runtime_error("[ERROR] forest path does not exist in HandSegmentationRf");
      }
      
      //create morph structure element
      int struc_elem_size = param.rf_param().struc_elem_size();
      struc_elem_ = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(struc_elem_size, struc_elem_size));
      std::cout << "struc_elem_: " << std::endl << struc_elem_ << std::endl;
    }
        
    virtual ~HandSegmentationRf() {}
    
    virtual std::vector<HandSegmentationResult> operator()(const cv::Mat_<Dtype>& depth, const cv::Mat_<Dtype>& ir, const cv::Vec3f& hint_2d) const;
                                                         
    static rv::rf::ForestPtr train(
        boost::shared_ptr<DataProvider<Dtype> > data_provider, 
        boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation,
        const rv::rf::ForestParameter& params);
    
protected:
    rv::rf::ForestPtr forest_;
    cv::Mat struc_elem_;
    
    Dtype bg_value_;
};

#endif
