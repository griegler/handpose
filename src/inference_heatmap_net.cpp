#include "inference_heatmap_net.h"
#include "create_data.h"

#include <rv/ocv/convert.h>
#include <rv/ocv/caffe.h>
#include <rv/ocv/colormap/colormap_cool_warm.h>
#include <rv/stat/core.h>

#include <opencv2/highgui/highgui.hpp>

template <typename Dtype>
std::vector< cv::Vec3f > InferenceHeatmapNet<Dtype>::operator()(const std::vector<HandPatch<Dtype> >& patches) const {
  std::vector<boost::shared_ptr<caffe::Blob<Dtype> > > blobs(patches.size()); //have to store shared_ptr, otherwise => bad time
  std::vector<caffe::Blob<Dtype>* > bottom(patches.size());
  for(size_t hp_idx = 0; hp_idx < patches.size(); ++hp_idx) {
    cv::Mat_<Dtype> patch = patches[hp_idx].patch_;
    blobs[hp_idx] = rv::ocv::mat2Blob(patch);
    
    bottom[hp_idx] = blobs[hp_idx].get();
  }
  
  Dtype loss;
  net_->Forward(bottom, &loss);
  
  if(param_.type() == pose::NNHeatmapParameter_HeatmapType_TWO_D) {
    std::vector<cv::Vec3f> es = inference2d();
    patchFitD(patches, es);
    return es;
  }
  else if(param_.type() == pose::NNHeatmapParameter_HeatmapType_TWO_POINT_FIVE_D) {
    std::vector<cv::Vec3f> es = inference2d();
    inferenceD(es);
    return es;
  }
  else {
    return std::vector<cv::Vec3f>(n_pts_, cv::Vec3f(0.0, 0.0, 0.0));
  }
}


template <typename Dtype>
std::vector< cv::Vec3f > InferenceHeatmapNet<Dtype>::inference2d() const {
  int heatmap_width = param_.dim_x();
  int heatmap_height = param_.dim_y();
  float min_heatmap_val = param_.min_heatmap_val();

  boost::shared_ptr<caffe::Blob<Dtype> > blob_ptr;
  if(net_->has_blob(param_.layer_name_2d())) {
    blob_ptr = net_->blob_by_name(param_.layer_name_2d());
  }
  else {
    LOG(ERROR) << "no blob for 2d heatmap found";
  }
  CHECK_EQ(blob_ptr->count(), n_pts_ * heatmap_height * heatmap_width);
  
  const Dtype* result_blob_data = blob_ptr->cpu_data();
  
  int result_blob_data_idx = 0;
  std::vector<cv::Vec3f> estimates;
  for(int heatmap_idx = 0; heatmap_idx < n_pts_; ++heatmap_idx) {
//     cv::Mat_<float> heatmap = cv::Mat_<float>::zeros(heatmap_height, heatmap_width);
    
    //collect points
    std::vector<cv::Vec2f> pts;
    std::vector<float> w;
    cv::Vec2f pt_0;
    float w_0 = 0;
    for(int x = 0; x < heatmap_width; ++x) {
      for(int y = 0; y < heatmap_height; ++y) {
        Dtype val = result_blob_data[result_blob_data_idx];
        result_blob_data_idx++;
        if(val > min_heatmap_val) {
          pts.push_back(cv::Vec2f(x, y));
          w.push_back(val);
        }
        
        if(val > w_0) {
          w_0 = val;
          pt_0 = cv::Vec2f(x, y);
        }
      }
    }
    
    //mean shift
    float bandwidth = 0.75;
    int max_iters = 10;
    for(int iter = 0; iter < max_iters; ++iter) {
      cv::Vec2f pt_1(0.0, 0.0);
      float sum_k = 0;
      for(size_t pts_idx = 0; pts_idx < pts.size(); ++pts_idx) {
        float diff_x = (pt_0(0) - pts[pts_idx](0)) / bandwidth;
        float diff_y = (pt_0(1) - pts[pts_idx](1)) / bandwidth;
        float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
        float k = w[pts_idx] * std::exp(-dist);
        
        sum_k += k;
        pt_1 = pt_1 + k * pts[pts_idx];
      }
      pt_1 /= sum_k;
      
//       std::cout << "from: " << pt_0 << " to " << pt_1 << std::endl;
      pt_0 = pt_1;
    }
    
    
//     static rv::ocv::ColorMap& cmap = rv::ocv::CoolWarmColorMap::i();
//     cv::imshow((boost::format("hm_%d") % heatmap_idx).str(), cmap.map(heatmap));
    
    cv::Vec3f es(pt_0(0), pt_0(1), 0);
    //to [-1, 1]
    es(0) = 2 * es(0) / heatmap_width - 1;
    es(1) = 2 * es(1) / heatmap_height - 1;
    estimates.push_back(es);
    
//     std::cout << "es_" << heatmap_idx << "  : " << es << std::endl;
  }
  
  return estimates;
}


template <typename Dtype>
void InferenceHeatmapNet<Dtype>::patchFitD(const std::vector< HandPatch< Dtype > >& patches, std::vector< cv::Vec3f >& es) const {
  int heatmap_width = param_.dim_x();
  int heatmap_height = param_.dim_y();
  
  //estimate depth
  std::vector<Dtype> valid_depth_values;
  std::vector<int> invalid_depth_indices;
  for(int idx = 0; idx < es.size(); ++idx) {
    //  to patch coord
    int col = (es[idx](0) + 1) / 2.0 * patches[0].patch_.cols;
    int row = (es[idx](1) + 1) / 2.0 * patches[0].patch_.rows;
    Dtype depth = patches[0].patch_(row, col);
    if(depth >= 3.0 || depth <= -3.0) { //TODO: not always true
      invalid_depth_indices.push_back(idx);
    }
    else {
      valid_depth_values.push_back(depth);
      es[idx](2) = depth;
    }
  }
  
  Dtype median_depth = 0;
  if(valid_depth_values.size() > 0) {
    median_depth = rv::stat::Median(valid_depth_values);
  }
  
  for(int idx = 0; idx < invalid_depth_indices.size(); ++idx) {
    es[invalid_depth_indices[idx]](2) = median_depth;
  }
}



template <typename Dtype>
void InferenceHeatmapNet<Dtype>::inferenceD(std::vector< cv::Vec3f >& es) const {

  boost::shared_ptr<caffe::Blob<Dtype> > blob_ptr;
  if(net_->has_blob(param_.layer_name_depth())) {
    blob_ptr = net_->blob_by_name(param_.layer_name_depth());
    LOG(INFO) << "infer d";
  }
  else {
    LOG(ERROR) << "no blob for depth heatmap found";
  }
  CHECK_EQ(blob_ptr->count(), n_pts_ * param_.dim_z());
  
  const Dtype* result_blob_data = blob_ptr->cpu_data();
  
  int result_blob_data_idx = 0;
  for(int pt_idx = 0; pt_idx < n_pts_; ++pt_idx) {
    Dtype mean = 0;
    Dtype normalizer = 0;
    for(int z = 0; z < param_.dim_z(); ++z) {
      Dtype val = result_blob_data[result_blob_data_idx];
      result_blob_data_idx++;
      if(val > param_.min_heatmap_val()) {
        mean = mean + val * z;
        normalizer += val;
      }
    }
    
    mean /= normalizer;
    
    //to [-1, 1]
    es[pt_idx](2) = 2 * mean / param_.dim_z() - 1;
  }
}



template class InferenceHeatmapNet<float>;
template class InferenceHeatmapNet<double>; 
