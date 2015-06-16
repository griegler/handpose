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

#include <iostream>

#include "make_hand_segmentation.h"
#include "make_data_provider.h"
#include "make_projection.h"
#include "common.h"
#include "hand_segmentation_rf.h"

#include <rv/ocv/color.h>
#include <rv/ocv/convert.h>
#include <rv/ocv/visualize.h>
#include <rv/ocv/linalg.h>
#include <rv/timer/cpu_timer.h>
#include <rv/io/serialization/compressed_serialization.h>


#include <opencv2/core/core.hpp>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;


typedef float Dtype;


int main(int argc, char** argv) {
  std::string forest_config_path;
  boost::filesystem::path config_path;
  boost::filesystem::path forest_path;
  
  po::options_description desc;
  desc.add_options()
    ("config_path", po::value<std::string>()->required(), "")
    ("forest_config_path", po::value<std::string>()->required(), "")
    ("forest_out_path", po::value<std::string>()->required(), "")
  ;
  
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    config_path = vm["config_path"].as<std::string>();
    forest_config_path = vm["forest_config_path"].as<std::string>();
    forest_path = vm["forest_out_path"].as<std::string>();
  } catch(std::exception& e) { 
    std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cout << desc << std::endl; 
    return -1; 
  }
  
  //read forest config
  rv::rf::ForestParameter forest_param;
  if(!readProtoFromTextFile(forest_config_path, &forest_param)) {
    std::cout << "ERROR: parsing config_path: " << forest_config_path << std::endl;
    return -1;
  }
    
  //read config
  pose::CreateDataParameter param;
  if(!readProtoFromTextFile(config_path, &param)) {
    std::cout << "ERROR: parsing config_path: " << config_path << std::endl;
    return -1;
  }
  
  //Init DataProvider
  boost::shared_ptr<DataProvider<Dtype> > data_provider = makeDataProvider<Dtype>(param.data_provider_param());
  boost::shared_ptr<Projection> projection = data_provider->projection();

  //Init HandSegmentation
  boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation = makeHandSegmentation<Dtype>(param.segmentation_param(), *projection, data_provider->bgValue());
  
  //train forest
  rv::rf::ForestPtr forest = HandSegmentationRf<Dtype>::train(data_provider, hand_segmentation, forest_param);
  
  //save forest
  if(!boost::filesystem::exists(forest_path.parent_path())) {
    boost::filesystem::create_directories(forest_path.parent_path());
  }
  rv::io::CompressedSerializationOut serialization(forest_path.string());
  forest->Save(serialization);  
  std::cout << "[INFO] stored forest to " << forest_path << std::endl;
}
