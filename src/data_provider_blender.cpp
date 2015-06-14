#include "data_provider_blender.h"
#include "common.h"

#include <rv/io/ls.h>
#include <rv/rand/rand.h>
#include <rv/timer/cpu_timer.h>

#include <boost/format.hpp>

#include <omp.h>

template <typename Dtype>
DataProviderBlender<Dtype>::DataProviderBlender(const pose::DataProviderParameter& param) : 
    DataProvider<Dtype>(param) {
  
  std::vector<boost::filesystem::path> all_depth_paths;
  if(param.blender_param().recursive()) {
    rv::io::ListFilesRecursive(param.blender_param().data_path(), ".*_depth_0002.exr", all_depth_paths);
  }
  else {
    rv::io::ListFiles(param.blender_param().data_path(), ".*_depth_0002.exr", all_depth_paths);
  }
  std::cout << "[INFO] found " << all_depth_paths.size() << " depth images" << std::endl;
  all_depth_paths = this->sortReducePaths(all_depth_paths);
  std::cout << "[INFO] sorted and reduced to " << all_depth_paths.size() << " depth images" << std::endl;
  
  rv::timer::CpuBatchedTimer& timer = rv::timer::CpuBatchedTimer::i();
  
  unsigned int valid_anno_idx = 0;
  depth_paths_.resize(all_depth_paths.size());
  annos_.resize(all_depth_paths.size());
  
  timer.start("overall");
  // #pragma omp parallel for
  for(unsigned int depth_idx = 0; depth_idx < all_depth_paths.size(); ++depth_idx) {
    timer.start("all_depth_paths");
    boost::filesystem::path depth_path = all_depth_paths[depth_idx];
    timer.stop("all_depth_paths");
    
    timer.start("annoPath");
    boost::filesystem::path anno_path = annoPath(depth_path);
    timer.stop("annoPath");
    
    if(boost::filesystem::exists(anno_path)) {
      timer.start("readAnno");
      std::vector<cv::Vec3f> anno = readAnno(anno_path, param.n_pts());
      timer.stop("readAnno");
      
      #pragma omp critical (DataProviderBlender_constructor_append)
      {
        timer.start("push_back");
        depth_paths_[valid_anno_idx] = depth_path;
        annos_[valid_anno_idx] = anno;
        valid_anno_idx++;
        if(valid_anno_idx % 10000 == 0) {
          std::cout << "read " << valid_anno_idx << " annotations" << std::endl;
        }
        timer.stop("push_back");
      }
    }
  }
  timer.stop("overall");
  
  timer.print(std::cout);
  
  depth_paths_.resize(valid_anno_idx);
  annos_.resize(valid_anno_idx);
  this->max_idx_ = depth_paths_.size();
  
  std::cout << "[INFO] label file " << param.blender_param().data_path() << " contained " << depth_paths_.size() << " annotated depth images" << std::endl; 
  
  cv::Vec3f mean, mean_0, std, std_0;
  this->stat(annos_, mean, std, mean_0, std_0);
  std::cout << "[INFO] anno[0] = " << mean_0 << " +- " << std_0 << " | anno[all] = " << mean << " +- " << std << std::endl;
}


template <typename Dtype>
boost::filesystem::path DataProviderBlender<Dtype>::annoPath(const boost::filesystem::path& depth_path) const {
  std::string id, ts;
  getIdTs(depth_path, id, ts);
  
  boost::format fmt("%s_%s_anno_blender.txt");
  return depth_path.parent_path() / (fmt % id % ts).str();
}


template <typename Dtype>
boost::filesystem::path DataProviderBlender<Dtype>::depthPath(int idx) const {
    return depth_paths_[idx];
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderBlender<Dtype>::depth_internal(int idx) {
    cv::Mat_<Dtype> dm = readExrDepth(depth_paths_[idx]) * 1000; //to mm
    
    Dtype bg_value = this->param_.bg_value();
    float noise_gaussian_sigma = this->param_.blender_param().noise_gaussian_sigma();
    
    for(int row = 0; row < dm.rows; ++row) {
      for(int col = 0; col < dm.cols; ++col) {
        Dtype d = dm(row, col);
        if(d < 0 || d > bg_value) {
          d = bg_value;
        }
        else {
          if(noise_gaussian_sigma > 0) {
            d = d + rv::rand::Rand<Dtype>::i().Gaussian(0, noise_gaussian_sigma);
          }
        }
        
        dm(row, col) = d;
      }
    }
    
    return dm;
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderBlender<Dtype>::ir_internal(int idx) {
    std::string ir_path = depth_paths_[idx].string();
    boost::replace_last(ir_path, "depth", "rgb");
    boost::replace_last(ir_path, "exr", "png");
    
    cv::Mat_<Dtype> ir;
    if(boost::filesystem::exists(ir_path)) {
      cv::Mat_<cv::Vec3b> bgr = cv::imread(ir_path);
      
      ir = cv::Mat_<Dtype>::zeros(bgr.rows, bgr.cols);
      for(int row = 0; row < ir.rows; ++row) {
        for(int col = 0; col < ir.cols; ++col) {
          cv::Vec3b v = bgr(row, col);
          if(v(2) >= 128 && v(1) <= 32 && v(0) <= 32) {
            ir(row, col) = 750;
          }
        }
      }
    }
    else {
      ir = 1.0 / readExrDepth(depth_paths_[idx]);
    }
    
    return ir;
}

template <typename Dtype>
std::vector< cv::Vec3f > DataProviderBlender<Dtype>::gt_internal(int idx) const {
    return annos_[idx];
}

template <typename Dtype>
cv::Vec3f DataProviderBlender<Dtype>::hint2d_internal(int idx) const {
    return annos_[idx][0];
}



template <typename Dtype>
void DataProviderBlender<Dtype>::shuffle() {
    static cv::RNG rng(time(0));
                                    
    std::vector<boost::filesystem::path> shuffled_depth_paths(depth_paths_.size());
    std::vector<std::vector<cv::Vec3f> > shuffled_annos(annos_.size());
    
    for(size_t idx = 0; idx < depth_paths_.size(); ++idx) {
        int rand_idx = rng.uniform(0, depth_paths_.size());
        shuffled_depth_paths[idx] = depth_paths_[rand_idx];
        shuffled_annos[idx] = annos_[rand_idx];
    }
    
    depth_paths_ = shuffled_depth_paths;
    annos_ = shuffled_annos;
}



template class DataProviderBlender<float>;
template class DataProviderBlender<double>;
