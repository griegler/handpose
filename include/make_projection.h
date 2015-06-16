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
