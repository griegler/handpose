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
