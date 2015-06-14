#include "inference_rf_simple.h"

template <typename Dtype> 
std::vector< cv::Vec3f > InferenceRfSimple<Dtype>::operator()(const std::vector< HandPatch< Dtype > >& patches) const {
   
  cv::Mat_<float> data = patches[0].patch_;
  int data_dim = data.rows * data.cols;
  
  rv::rf::RfMatPtr patch = rv::rf::CreateRfMat(data_dim, 1);
  int idx = 0;
  for(int w = 0; w < data.cols; ++w) {
    for(int h = 0; h < data.rows; ++h) {
      (*patch)(idx, 0) = data(h, w);
      idx++;
    }
  }

  if(T_ != 0) {
    *patch = (*T_) * (*patch);
  }

  rv::rf::SamplePtr sample = boost::make_shared<rv::rf::MatSample>(patch);
  rv::rf::VecPtrTargetPtr estimated_targets = forest_->inferencemt(sample);
  rv::rf::RfMatPtr result = (*estimated_targets)[0]->data();

  int result_blob_data_idx = 0;
  std::vector<cv::Vec3f> es(result->rows() / 3);
  for(int row = 0; row < result->rows() / 3; ++row, result_blob_data_idx += 3) {
    Dtype x = (*result)(result_blob_data_idx, 0);
    Dtype y = (*result)(result_blob_data_idx + 1, 0);
    Dtype z = (*result)(result_blob_data_idx + 2, 0);

    es[row](0) = x;
    es[row](1) = y;
    es[row](2) = z;

    // std::cout << es[row] << std::endl;
  }

  return es;
}


template class InferenceRfSimple<float>;
template class InferenceRfSimple<double>;
