#include "inference_regression_net.h"
#include "create_data.h"

#include <rv/ocv/convert.h>
#include <rv/ocv/caffe.h>

#include <opencv2/highgui/highgui.hpp>

template <typename Dtype>
std::vector< cv::Vec3f > InferenceRegressionNet<Dtype>::operator()(const std::vector<HandPatch<Dtype> >& patches) const {
    std::vector<boost::shared_ptr<caffe::Blob<Dtype> > > blobs(patches.size()); //have to store shared_ptr, otherwise => bad time
    std::vector<caffe::Blob<Dtype>* > bottom(patches.size());
    for(size_t hp_idx = 0; hp_idx < patches.size(); ++hp_idx) {
        cv::Mat_<Dtype> patch = patches[hp_idx].patch_;
        blobs[hp_idx] = rv::ocv::mat2Blob(patch);
        
        bottom[hp_idx] = blobs[hp_idx].get();
    }
    
    Dtype loss;
    net_->Forward(bottom, &loss);
    
    boost::shared_ptr<caffe::Blob<Dtype> > blob_ptr;
    if(net_->has_blob("ip_regression")) {
        blob_ptr = net_->blob_by_name("ip_regression");
    }
    else if(net_->has_blob("ip_regression_16")) {
        blob_ptr = net_->blob_by_name("ip_regression_16");
    }
    else if(net_->has_blob("ip_regression_20")) {
        blob_ptr = net_->blob_by_name("ip_regression_20");
    }
    else {   
        LOG(ERROR) << "no blob for regression found";
    }
    const Dtype* result_blob_data = blob_ptr->cpu_data();
        
    int result_blob_data_idx = 0;
    std::vector<cv::Vec3f> es(blob_ptr->channels() / 3);
    for(int row = 0; row < blob_ptr->channels() / 3; ++row, result_blob_data_idx += 3) {
        Dtype x = result_blob_data[result_blob_data_idx];
        Dtype y = result_blob_data[result_blob_data_idx + 1];
        Dtype z = result_blob_data[result_blob_data_idx + 2];
        
        es[row](0) = x;
        es[row](1) = y;
        es[row](2) = z;
    }
    
//     for(size_t idx = 0; idx < es.size(); ++idx) {
//         std::cout << es[idx] << std::endl;
//     }
    
    return es;
}

template class InferenceRegressionNet<float>;
template class InferenceRegressionNet<double>; 
