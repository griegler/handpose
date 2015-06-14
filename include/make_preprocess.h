#ifndef MAKE_PREPROCESS_H
#define MAKE_PREPROCESS_H

#include "preprocess.h"

#include "pose.pb.h"

#include <rv/ocv/io/csv.h>

#include <boost/make_shared.hpp>


template <typename Dtype>
boost::shared_ptr< Preprocess<Dtype> > makePreprocess(const pose::PreprocessParameter& param) {
  
  std::cout << "[INFO] Creating Preprocess" << std::endl;
  
  boost::shared_ptr<Preprocess<Dtype> > preprocess = boost::make_shared<Preprocess<Dtype> >();
  
  if(param.mean_path() != "") {
    std::cout << "[INFO] add mean to preprocess" << std::endl;
    cv::Mat_<Dtype> mean = rv::ocv::ReadCsv<Dtype>(param.mean_path());
    preprocess->addSubtractMean(mean);
  }
  
  return preprocess;
}

#endif
