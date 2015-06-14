#ifndef MAKE_PROJECTION_H
#define MAKE_PROJECTION_H

#include <fstream>
#include <iostream>

#include "projection.h"

#include "pose.pb.h"

#include <boost/make_shared.hpp>
#include <boost/filesystem.hpp>


inline boost::shared_ptr<Projection> makeProjection(const pose::ProjectionParameter& param, 
                                             const boost::filesystem::path& hint) {
  
  boost::shared_ptr<Projection> projection;
  
  //automatic find calib if possible from data_provider depth path
  std::string calib_path = param.calib_path();
  if(calib_path == "") {
    boost::filesystem::path parent = hint;

    while(parent.parent_path() != parent && calib_path == "") {
      if(boost::filesystem::exists(parent / "calibration_depth.txt")) {
        calib_path = (parent / "calibration_depth.txt").string();
      }

      if(boost::filesystem::exists(parent / "calib.txt")) {
        calib_path = (parent / "calib.txt").string();
      }

      parent = parent.parent_path();
    }
  }
  
  if(calib_path != "") {
    cv::Matx33d K = 0;
    cv::Matx18d d = 0;
    
    std::ifstream fin(calib_path.c_str());
    
    fin >> K(0, 0);
    fin >> K(1, 1);
    fin >> K(0, 2);
    fin >> K(1, 2);
    K(2, 2) = 1;
    
    for(int d_idx = 0; d_idx < d.cols; ++d_idx) {
      if(fin.good()) {
        fin >> d(0, d_idx);
      }
    }

    fin.close();
    
    if(param.type() == pose::ProjectionParameter_ProjectionType_ORTHOGRAPHIC) {
      std::cout << "[INFO] Creating OrthographicProjection" << std::endl;
      projection = boost::make_shared<OrthographicProjection>(K, d);
    }
    else if(param.type() == pose::ProjectionParameter_ProjectionType_PROJECTIVE) {
      std::cout << "[INFO] Creating ProjectiveProjection" << std::endl;
      projection = boost::make_shared<ProjectiveProjection>(K, d);
    }
  }
  else {
    std::cout << "[ERROR] no calibration file found" << std::endl;
  }
  
  if(projection == 0) {
    std::cout << "[ERROR] No Projection created" << std::endl;
  }
  
  return projection;
}

#endif
