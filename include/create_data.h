#ifndef CREATE_DATA_H
#define CREATE_DATA_H

#include "data_provider.h"
#include "hand_segmentation.h"
#include "hand_patch.h"
#include "patch_extraction.h"
#include "pose.pb.h"

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rv/io/h5_block_writer.h>

template <typename Dtype>
class CreateData {
public:
  CreateData(const pose::CreateDataParameter& param,
    DataProvider<Dtype>& data_provider, 
    const HandSegmentation<Dtype>& hand_segmentation, 
    const PatchExtraction<Dtype>& patch_extraction) :
      param_(param), data_provider_(data_provider), 
      hand_segmentation_(hand_segmentation), 
      patch_extraction_(patch_extraction) {
        
    createRotations();
  }
    
  virtual ~CreateData() {};

  virtual void data2Hdf5() const;    
  
  
  virtual void mirror(const cv::Mat_<Dtype>& depth, 
      const cv::Mat_<Dtype>& ir, const std::vector<cv::Vec3f>& anno,
      cv::Mat_<Dtype>& mirrored_depth, cv::Mat_<Dtype>& mirrored_ir,
      std::vector<cv::Vec3f>& mirrored_anno) const;
  virtual void rotate(const cv::Mat_<Dtype>& depth, 
      const cv::Mat_<Dtype>& ir, const std::vector<cv::Vec3f>& anno,
      cv::Mat_<Dtype>& rotated_depth, cv::Mat_<Dtype>& rotated_ir,
      std::vector<cv::Vec3f>& rotated_anno, float rot_deg) const;
    
protected:
  virtual void createRotations();
                        
protected:
  const pose::CreateDataParameter& param_; 
  DataProvider<Dtype>& data_provider_;
  const HandSegmentation<Dtype>& hand_segmentation_;
  const PatchExtraction<Dtype>& patch_extraction_;
  
  std::vector<float> rotations_;
};




#endif
