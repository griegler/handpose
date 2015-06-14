#ifndef MAKE_DATA_PROVIDER_H
#define MAKE_DATA_PROVIDER_H

#include <iostream>

#include "data_provider.h"
#include "data_provider_blender.h"
#include "data_provider_csv.h"
#include "data_provider_dtang.h"
#include "data_provider_cam.h"

#include "pose.pb.h"

#include <boost/make_shared.hpp>


template <typename Dtype>
boost::shared_ptr< DataProvider<Dtype> > makeDataProvider(const pose::DataProviderParameter& param) {
    boost::shared_ptr<DataProvider<Dtype> > data_provider;
    
    if(param.type() == pose::DataProviderParameter_DataProviderType_DTANG) {
      std::cout << "[INFO] Creating DataProviderDTang" << std::endl;
      data_provider = boost::make_shared<DataProviderDTang<Dtype> >(param);
    }
    else if(param.type() == pose::DataProviderParameter_DataProviderType_CSV) {
      std::cout << "[INFO] Creating DataProviderCsv" << std::endl;
      data_provider = boost::make_shared<DataProviderCsv<Dtype> >(param);
    }
    else if(param.type() == pose::DataProviderParameter_DataProviderType_BLENDER) {
      std::cout << "[INFO] Creating DataProviderBlender" << std::endl;
      data_provider = boost::make_shared<DataProviderBlender<Dtype> >(param);
    }
#ifdef BUILD_DATA_PROVIDER_CAM
    else if(param.type() == pose::DataProviderParameter_DataProviderType_CAM) {
      std::cout << "[INFO] Creating DataProviderCam" << std::endl;
      data_provider = boost::make_shared<DataProviderCam<Dtype> >(param);
    }
#endif
    
    if(param.shuffle()) {
      data_provider->shuffle();
    }
    
    return data_provider;
}

#endif
