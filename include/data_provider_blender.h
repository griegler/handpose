#ifndef DATA_PROVIDER_BLENDER_H
#define DATA_PROVIDER_BLENDER_H

#include <data_provider.h>

#include "pose.pb.h"

template <typename Dtype>
class DataProviderBlender : public DataProvider<Dtype> {
public:
  DataProviderBlender(const pose::DataProviderParameter& param);
  ~DataProviderBlender() {}
  
  virtual boost::filesystem::path depthPath(int idx) const;
  
  virtual void shuffle();

protected:
  virtual std::vector< cv::Vec3f > gt_internal(int idx) const;
  virtual cv::Vec3f hint2d_internal(int idx) const;
  virtual cv::Mat_< Dtype > depth_internal(int idx);
  virtual cv::Mat_< Dtype > ir_internal(int idx);

  boost::filesystem::path annoPath(const boost::filesystem::path& depth_path) const;
  
private:  
  std::vector<boost::filesystem::path> depth_paths_;
  std::vector<std::vector<cv::Vec3f> > annos_;
};

#endif
