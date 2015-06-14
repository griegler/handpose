#ifdef BUILD_DATA_PROVIDER_CAM

#ifndef DATA_PROVIDER_CAM_H
#define DATA_PROVIDER_CAM_H

#include "data_provider.h"
#include "pose.pb.h"

#include <rv/camera/camera.h>

#include <boost/shared_ptr.hpp>

template <typename Dtype>
class DataProviderCam : public DataProvider<Dtype> {
public:
    DataProviderCam(const pose::DataProviderParameter& param);
    virtual ~DataProviderCam(); 
    
    virtual boost::filesystem::path depthPath(int idx) const;    
    virtual void shuffle() {}

protected:
  virtual std::vector< cv::Vec3f > gt_internal(int idx) const;
  virtual cv::Vec3f hint2d_internal(int idx) const;
  virtual cv::Mat_< Dtype > depth_internal(int idx);
  virtual cv::Mat_< Dtype > ir_internal(int idx);

private:
  rv::camera::Camera* cam_;
  bool delete_cam_;

  std::string im_type_depth_;
  std::string im_type_ir_;

  cv::Mat_<Dtype> ir_;
  cv::Mat_<Dtype> depth_;
};

#endif

#endif //BUILD_DATA_PROVIDER_CAM
