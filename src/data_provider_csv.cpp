#include "data_provider_csv.h"
#include "common.h"

#include <rv/io/ls.h>
#include <rv/ocv/io/csv.h>


template <typename Dtype>
DataProviderCsv<Dtype>::DataProviderCsv(const pose::DataProviderParameter& param) : 
    DataProvider<Dtype>(param) {
  
  std::vector<boost::filesystem::path> all_depth_paths;
  if(param.csv_param().recursive()) {
    rv::io::ListFilesRecursive(param.csv_param().data_path(), ".*_depth.csv", all_depth_paths);
  }
  else {
    rv::io::ListFiles(param.csv_param().data_path(), ".*_depth.csv", all_depth_paths);
  }
  std::cout << "[INFO] found " << all_depth_paths.size() << " depth images" << std::endl;
  all_depth_paths = this->sortReducePaths(all_depth_paths);
  std::cout << "[INFO] sorted and reduced to " << all_depth_paths.size() << " depth images" << std::endl;
  
  
  for(int depth_idx = 0; depth_idx < all_depth_paths.size(); ++depth_idx) {
    boost::filesystem::path depth_path = all_depth_paths[depth_idx];
    std::vector<boost::filesystem::path> anno_paths = lsAnnoPaths(depth_path);
    
    if(param.csv_param().anno_type() == pose::CsvParameter_AnnoType_ALL) {
      depth_paths_.push_back(depth_path);
      annos_.push_back(std::vector<cv::Vec3f>(param.n_pts(), cv::Vec3f(0.0, 0.0, 0.0)));
    }
    else {
      for(int anno_idx = 0; anno_idx < anno_paths.size(); ++anno_idx) {
        boost::filesystem::path anno_path = anno_paths[anno_idx];
        std::string annotator = getAnnotator(anno_path);
    
        if((param.csv_param().anno_type() == pose::CsvParameter_AnnoType_ANNO) || 
           (param.csv_param().anno_type() == pose::CsvParameter_AnnoType_VALID_ANNO && 
            annotator != "extrapolated" && annotator != "init")) {                
          std::vector<cv::Vec3f> anno = readAnno(anno_path, param.n_pts());
      
          depth_paths_.push_back(depth_path);
          annos_.push_back(anno);
        }
      }
    }
  }
  
  this->max_idx_ = depth_paths_.size();
  
  std::cout << "[INFO] label file " << param.csv_param().data_path() << " contained " << depth_paths_.size() << " annotated depth images" << std::endl;
  
  cv::Vec3f mean, mean_0, std, std_0;
  this->stat(annos_, mean, std, mean_0, std_0);
  std::cout << "[INFO] anno[0] = " << mean_0 << " +- " << std_0 << " | anno[all] = " << mean << " +- " << std << std::endl;
}

template <typename Dtype>
boost::filesystem::path DataProviderCsv<Dtype>::depthPath(int idx) const {
  return depth_paths_[idx];
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderCsv<Dtype>::depth_internal(int idx) {
  boost::filesystem::path depth_path = depth_paths_[idx];
  cv::Mat_<Dtype> dm = rv::ocv::ReadCsv<Dtype>(depth_path);
  
  cv::Mat_<Dtype> ir_img = ir_internal(idx);
  Dtype bg_value = this->param_.bg_value();
  this->param_.csv_param();
  int median_ksize = this->param_.csv_param().median_ksize();
  float min_ir = this->param_.csv_param().min_ir();
  
  for(int row = 0; row < dm.rows; ++row) {
    for(int col = 0; col < dm.cols; ++col) {
      Dtype d = dm(row, col);
      Dtype i = ir_img(row, col);
      
      if(d < 0 || d > bg_value || i < min_ir) {
        dm(row, col) = bg_value;
      }
    }
  }
  
  if(median_ksize > 0) {
    cv::medianBlur(dm, dm, median_ksize);
  }

  return dm;
}

template <typename Dtype>
cv::Mat_< Dtype > DataProviderCsv<Dtype>::ir_internal(int idx) {
  boost::filesystem::path depth_path = depth_paths_[idx];
  boost::filesystem::path ir_path = infraredPath(depth_path); 
  cv::Mat_<Dtype> ir_img = rv::ocv::ReadCsv<Dtype>(ir_path);
  
  return ir_img;
}

template <typename Dtype>
std::vector< cv::Vec3f > DataProviderCsv<Dtype>::gt_internal(int idx) const {
  return annos_[idx];
}

template <typename Dtype>
cv::Vec3f DataProviderCsv<Dtype>::hint2d_internal(int idx) const {
  return annos_[idx][0];
}



template <typename Dtype>
void DataProviderCsv<Dtype>::shuffle() {
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



template class DataProviderCsv<float>;
template class DataProviderCsv<double>;
