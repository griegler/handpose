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
#include <fstream>

#include <caffe/caffe.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;


typedef float Dtype;


int main(int argc, char** argv) {
    
  boost::filesystem::path solver_path;
  boost::filesystem::path weights_path;
  int cuda_device;
  
  po::options_description desc;
  desc.add_options()
    ("solver_path", po::value<std::string>()->required(), "")
    ("weights_path", po::value<std::string>()->default_value(""), "")
    ("cuda_device", po::value<int>()->default_value(0), "")
  ;
  
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    
    solver_path = vm["solver_path"].as<std::string>();
    weights_path = vm["weights_path"].as<std::string>();
    
    cuda_device = vm["cuda_device"].as<int>();
    if(cuda_device >= 0) {
        caffe::Caffe::SetDevice(cuda_device);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    } 
    else {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }
  } catch(std::exception& e) { 
      std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cout << desc << std::endl; 
      return -1; 
  }
  
  //read solver config
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_path.string(), &solver_param);
  
  //get solver
  boost::shared_ptr<caffe::Solver<Dtype> > solver(caffe::GetSolver<Dtype>(solver_param));

  //copy weights if given
  if(weights_path != "") {
    if(!boost::filesystem::exists(weights_path)) {
      std::cout << "[ERROR] weights path does not exist" << std::endl;
      return -1;
    }
    solver->net()->CopyTrainedLayersFrom(weights_path.string());
  }
  
  //solve network
  solver->Solve();
  
  return 0;
}
