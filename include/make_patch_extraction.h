#ifndef MAKE_PATCH_EXTRACTION_H
#define MAKE_PATCH_EXTRACTION_H

#include "patch_extraction.h"

#include "pose.pb.h"

#include <boost/make_shared.hpp>


template <typename Dtype>
boost::shared_ptr< PatchExtraction<Dtype> > makePatchExtraction(const pose::PatchExtractionParameter& param, Dtype bg_value) {
    
  std::cout << "[INFO] Creating PatchExtraction" << std::endl;
  
  std::vector<int> patch_widths(param.patch_width_size());
  for(int idx = 0; idx < param.patch_width_size(); ++idx) {
    patch_widths[idx] = param.patch_width(idx);
  }
  
  return boost::make_shared<PatchExtraction<Dtype> >(patch_widths, bg_value, param.normalize_depth());
}

#endif