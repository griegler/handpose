#ifndef INFERENCE_RF_SIMPLE_H
#define INFERENCE_RF_SIMPLE_H

#include "inference.h"

#include "pose.pb.h"

#include <rv/ml/rf/forest.h>
#include <rv/ml/rf/data/matsample.h>
#include <rv/io/serialization/compressed_serialization.h>
#include <rv/eigen/io/csv.h>

template <typename Dtype>
class InferenceRfSimple : public Inference<Dtype> {
public:
  InferenceRfSimple(const pose::RFParameter& param) {
    rv::io::CompressedSerializationIn serialization(param.forest_path());
    forest_ = boost::make_shared<rv::rf::Forest>();
    forest_->Load(serialization);

    if(param.transform_csv() != "") {
      T_ = rv::rf::CreateRfMat(0, 0);
      rv::eigen::ReadCsv(param.transform_csv(), *T_);
    }
  }

  virtual ~InferenceRfSimple() {}
  
  virtual std::vector<cv::Vec3f> operator()(const std::vector<HandPatch<Dtype> >& patches) const;
private:
  rv::rf::ForestPtr forest_;
  rv::rf::RfMatPtr T_;
};

#endif
