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
