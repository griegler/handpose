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

#include "common.h"
#include "make_data_provider.h"
#include "make_hand_segmentation.h"
#include "make_inference.h"

#include <rv/ocv/color.h>
#include <rv/ocv/convert.h>
#include <rv/ocv/linalg.h>
#include <rv/io/h5_block_reader.h>
#include <rv/ml/rf/data/matsample.h>
#include <rv/io/serialization/compressed_serialization.h>
#include <rv/io/serialization/text_serialization.h>

#include <opencv2/core/core.hpp>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;


typedef float Dtype;

void hdf2samples(const boost::filesystem::path& hdf_path,
                 const std::string& data_name,
                 int inc,
                 std::vector<rv::rf::SamplePtr>& samples,
                 rv::rf::VecPtrTargetPtr& targets) {
  rv::io::H5BlockReader data_reader(hdf_path, data_name);
  rv::io::H5BlockReader regression_reader(hdf_path, "regression");
  
  int data_channels = data_reader.channels();
  int data_height = data_reader.height();
  int data_width = data_reader.width();
  int data_dim = data_channels * data_height * data_width;
  
  int regression_channels = regression_reader.channels();
  int regression_height = regression_reader.height();
  int regression_width = regression_reader.width();
  int regression_dim = regression_channels * regression_height * regression_width;
  
  while(data_reader.hasNext() && regression_reader.hasNext()) {
      rv::rf::RfMatPtr patch = rv::rf::CreateRfMat(data_dim, 1);
      int patch_idx = 0;
      for(int c = 0; c < data_channels; ++c) {
        for(int w = 0; w < data_width; ++w) {
          for(int h = 0; h < data_height; ++h) {
            (*patch)(patch_idx, 0) = data_reader.data(c, h, w);
            patch_idx++;
          }
        }
      }

      rv::rf::RfMatPtr regression = rv::rf::CreateRfMat(regression_dim, 1);
      int regression_idx = 0;
      for(int c = 0; c < regression_channels; ++c) {
        for(int h = 0; h < regression_height; ++h) {
          for(int w = 0; w < regression_width; ++w) {
            (*regression)(regression_idx, 0) = regression_reader.data(c, h, w);
            regression_idx++;
          }
        }
      }
      
//     cv::imshow("sample", rv::ocv::clamp(patches[0]));
//     std::cout << "regression_vec: " << regression << std::endl;
//     cv::waitKey(0);
      
    samples.push_back(boost::make_shared<rv::rf::MatSample>(patch));
    targets->push_back(boost::make_shared<rv::rf::Target>(regression));
    
    for(int i = 0; i < inc && data_reader.hasNext() && regression_reader.hasNext(); ++i) {
      data_reader.next(); 
      regression_reader.next();
    }
  }
}


int main(int argc, char** argv) {
  std::string forest_config_path;
  boost::filesystem::path out_forest_path;
  boost::filesystem::path train_hdf_path;
  boost::filesystem::path test_hdf_path;
  std::string data_name;
  int inc;
  
  po::options_description desc;
  desc.add_options()
    ("forest_config_path", po::value<std::string>()->required(), "")
    ("out_forest_path", po::value<std::string>()->required(), "")
    ("train_hdf_path", po::value<std::string>()->required(), "")
    ("test_hdf_path", po::value<std::string>()->default_value(""), "")
    ("data_name", po::value<std::string>()->required(), "")
    ("inc", po::value<int>()->required(), "")
  ;
  
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    
    forest_config_path = vm["forest_config_path"].as<std::string>();
    out_forest_path = vm["out_forest_path"].as<std::string>();
    train_hdf_path = vm["train_hdf_path"].as<std::string>();
    test_hdf_path = vm["test_hdf_path"].as<std::string>();
    data_name = vm["data_name"].as<std::string>();
    inc = vm["inc"].as<int>();
  } catch(std::exception& e) { 
    std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cout << desc << std::endl; 
    return -1; 
  }
  
  rv::rf::ForestParameter forest_param;
  if(!readProtoFromTextFile(forest_config_path, &forest_param)) {
      std::cout << "ERROR: parsing config_path: " << forest_config_path << std::endl;
      return -1;
  }
  
        
  //read hdf data to samples and targets    
  std::vector<rv::rf::SamplePtr> train_samples;
  rv::rf::VecPtrTargetPtr train_targets = boost::make_shared<rv::rf::VecTargetPtr>();
  hdf2samples(train_hdf_path, data_name, inc, train_samples, train_targets);

  std::vector<rv::rf::VecPtrTargetPtr> vec_train_targets;
  vec_train_targets.push_back(train_targets);
  
  //sample stat
  int data_width = train_samples[0]->width();
  int data_height = train_samples[0]->height();
  std::cout << "[INFO] loaded " << train_samples.size() << " samples and targets for training" << std::endl;
  
  //train forest
  rv::rf::TrainForest rf_train(forest_param, true);
  rv::rf::ForestPtr forest;
  forest = rf_train.Train(train_samples, vec_train_targets, vec_train_targets, rv::rf::TRAIN, forest);
  if(!boost::filesystem::exists(out_forest_path.parent_path())) {
    boost::filesystem::create_directories(out_forest_path.parent_path());
  }

  rv::io::CompressedSerializationOut serialization_bin(out_forest_path.string());
  forest->Save(serialization_bin);  

  return 0;
}
