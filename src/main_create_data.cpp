#include <iostream>

#include <omp.h>

#include <common.h>
#include <make_hand_segmentation.h>
#include <make_data_provider.h>
#include <make_patch_extraction.h>
#include <make_projection.h>
#include <create_data.h>

#include <pose.pb.h>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

typedef float Dtype;


int main(int argc, char** argv) {    
    
    boost::filesystem::path config_path;
    int n_threads;
    
    po::options_description desc;
    desc.add_options()
        ("config_path", po::value<std::string>()->required(), "")
        ("n_threads", po::value<int>()->default_value(0), "<=0 as many as possible")
    ;
    
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        po::notify(vm);
        config_path = vm["config_path"].as<std::string>();
        n_threads = vm["n_threads"].as<int>();
        
        if(n_threads > 0) {
          omp_set_num_threads(n_threads);
        }
    } catch(std::exception& e) { 
        std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
        std::cout << desc << std::endl; 
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
    
    //Init HandSegmentation
    boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation = makeHandSegmentation<Dtype>(param.segmentation_param(), *(data_provider->projection()), data_provider->bgValue());
    
    //Init ExtractPatch
    boost::shared_ptr<PatchExtraction<Dtype> > patch_extraction = makePatchExtraction<Dtype>(param.patch_extraction_param(), data_provider->bgValue());
    
    //Init CreateData 
    CreateData<Dtype> create_data(param, *data_provider, *hand_segmentation, *patch_extraction);
//     create_data.addHeatmaps(18, 3);
    
    create_data.data2Hdf5();
    
    return 0;
}
