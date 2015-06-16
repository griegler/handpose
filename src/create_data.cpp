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

#include "create_data.h"
#include "common.h"

#include <rv/rand/rand.h>
#include <rv/ocv/linalg.h>
#include <rv/timer/cpu_timer.h>
#include <rv/timer/fps.h>
#include <rv/ocv/io/csv.h>

#include <boost/make_shared.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <omp.h>


template <typename Dtype>
void CreateData<Dtype>::data2Hdf5() const {
  static const bool VISUALIZE = false;

  const std::vector<int> patch_widths = patch_extraction_.patchWidths();
  std::vector<std::string> data_names(patch_widths.size());
  for(size_t patch_width_idx = 0; patch_width_idx < data_names.size(); ++patch_width_idx) {
    data_names[patch_width_idx] = (boost::format("data_%d") % patch_widths[patch_width_idx]).str();
  }
  
  boost::filesystem::path out_prefix = param_.out_prefix();
  boost::filesystem::path out_dir = out_prefix.parent_path();
  if(!boost::filesystem::exists(out_dir)) {
    boost::filesystem::create_directories(out_dir);
  }
  std::string prefix = boost::filesystem::basename(param_.out_prefix());
  boost::filesystem::path list_path = out_dir / (boost::format("%s.txt") % prefix).str();
  boost::filesystem::path hdf5_path = out_dir / (boost::format("%s_") % prefix).str();
  
  rv::io::H5BlockWriter<Dtype> hdf5; 
  for(size_t patch_width_idx = 0; patch_width_idx < data_names.size(); ++patch_width_idx) {
    hdf5.init(data_names[patch_width_idx], 1, patch_widths[patch_width_idx], patch_widths[patch_width_idx]);
  }
  hdf5.init("regression", data_provider_.nPts() * 3, 1, 1);
  hdf5.init("leftright", 1, 1, 1);
  
  if(param_.shuffle()) {
    data_provider_.shuffle();
  }
  
  Dtype bg_value = data_provider_.bgValue();
  bool normalizeDepth = patch_extraction_.normalizeDepth();
  
  rv::timer::CpuBatchedTimer& timer = rv::timer::CpuBatchedTimer::i();
  
  int n_invalid_annos = 0;
  
  int data_proder_max_idx = data_provider_.maxIdx();
  
  unsigned int n_max_annos = data_proder_max_idx * rotations_.size() * (param_.mirroring() ? 2 : 1);
  rv::timer::Fps fps;
  
  #pragma omp parallel for
  for(int data_provider_idx = 0; data_provider_idx < data_proder_max_idx; ++data_provider_idx) {
    
    timer.start("data_provider_");
    boost::filesystem::path depth_path = data_provider_.depthPath(data_provider_idx);
    cv::Mat_<Dtype> input_depth = data_provider_.depth(data_provider_idx);
    cv::Mat_<Dtype> input_ir = data_provider_.ir(data_provider_idx);
    std::vector<cv::Vec3f> input_anno = data_provider_.gt(data_provider_idx);
    timer.stop("data_provider_");
    
    std::cout << "[INFO] process: " << depth_path << std::endl;
    
    if(!boost::filesystem::exists(depth_path)) {
      std::cout << "[WARNING] depth map path does not exist: " << depth_path << std::endl;
      continue;
    }
    
    for(int mirror_idx = 0; (param_.mirroring() && mirror_idx < 2) || (!param_.mirroring() && mirror_idx < 1); ++mirror_idx) {
      cv::Mat_<Dtype> mirrored_depth;
      cv::Mat_<Dtype> mirrored_ir;
      std::vector<cv::Vec3f> mirrored_anno;
      
      timer.start("mirror");
      if(mirror_idx == 0) {
        mirrored_depth = input_depth;
        mirrored_ir = input_ir;
        mirrored_anno = input_anno;
      }
      else if(mirror_idx == 1) {
        mirror(input_depth, input_ir, input_anno, mirrored_depth, mirrored_ir, mirrored_anno);
      }
      timer.stop("mirror");
  
      for(size_t rot_idx = 0; rot_idx < rotations_.size(); ++rot_idx) {
        float rot = rotations_[rot_idx];
        
        timer.start("rotate");
        cv::Mat_<Dtype> depth;
        cv::Mat_<Dtype> ir;
        std::vector<cv::Vec3f> anno;
        if(rot == 0) {
          depth = mirrored_depth;
          anno = mirrored_anno;
        }
        else {
          rotate(mirrored_depth, mirrored_ir, mirrored_anno, depth, ir, anno, rot);
        }
        timer.stop("rotate");
        
        if(VISUALIZE) {
          cv::Mat_<cv::Vec3b> im3 = annoShow(depth, anno, bg_value);
          cv::imshow("anno", im3);
        }
        
        timer.start("hand_segmentation_");
        std::vector<HandSegmentationResult> segmentations = hand_segmentation_(depth, ir, anno[0]);
        timer.stop("hand_segmentation_");
        if(segmentations.size() == 0) {
          std::cout << "[WARNING] no hand found " << depth_path << " rotation " << rot << std::endl;
        }
        else {
          timer.start("patch_extraction_");
          std::vector<HandPatch<Dtype> > hand_patches = patch_extraction_(depth, segmentations[0]);
          if(!patch_extraction_.isReasonableHandPatch(hand_patches)) {
            #pragma omp critical (CreateData_data2hdf5_n_invalid_annos)
            {
              std::cout << "[WARNING] no reasonable hand_patch @ " << depth_path << " rotation: " << rot << std::endl;
              //cv::imshow("patch", hand_patches[0].patch_);
              //cv::waitKey(0);
              n_invalid_annos++;
            }
          }
          else {
            std::vector<cv::Vec3f> adjusted_anno = hand_patches[0].annoInHandPatch(anno, segmentations[0].roi_);
            timer.stop("patch_extraction_");
            
            if(!hand_patches[0].isReasonableCenteredAnno(adjusted_anno, normalizeDepth)) {
              #pragma omp critical (CreateData_data2hdf5_n_invalid_annos)
              {
                std::cout << "[WARNING] no reasonable annotation @ " << depth_path << " rotation: " << rot << std::endl;
                for(size_t adjusted_anno_idx = 0; adjusted_anno_idx < adjusted_anno.size(); ++adjusted_anno_idx) {
                  cv::Vec3f aa = adjusted_anno[adjusted_anno_idx];
                  std::cout << "    " << aa(0) << "/" << aa(1) << "/" << aa(2) << std::endl;
                }
                std::cout << std::endl;
                
                // cv::imshow("patch", hand_patches[0].patch_);
                // cv::waitKey(0);

                n_invalid_annos++;
              }
            }
            else {
              #pragma omp critical (CreateData_data2hdf5_hdf5)
              {
                timer.start("hdf5");
                for(size_t patch_width_idx = 0; patch_width_idx < data_names.size(); ++patch_width_idx) {
                  hdf5.add(data_names[patch_width_idx], hand_patches[patch_width_idx].patch_);
                }
                for(size_t anno_idx = 0; anno_idx < adjusted_anno.size(); ++anno_idx) {
                  hdf5.add("regression", adjusted_anno[anno_idx](0));
                  hdf5.add("regression", adjusted_anno[anno_idx](1));
                  hdf5.add("regression", adjusted_anno[anno_idx](2));
                }
                hdf5.add("leftright", mirror_idx);
                timer.stop("hdf5");
                
                if(VISUALIZE) {
                  for(size_t patch_width_idx = 0; patch_width_idx < data_names.size(); ++patch_width_idx) {
                    cv::Mat_<cv::Vec3b> im3 = handPatchShow(hand_patches[patch_width_idx], adjusted_anno);
                    cv::imshow(data_names[patch_width_idx], im3); 

                    std::string csv_name = std::string("create_data_") + data_names[patch_width_idx] + ".csv";
                    rv::ocv::WriteCsv(csv_name, hand_patches[patch_width_idx].patch_);
                  }
                }
        
                std::cout << "[INFO] written num: " << hdf5.totalNum("regression") << "/" << n_max_annos << std::endl;
                std::cout << "[INFO] fps = " << fps.Bump() << " | remaining h: " << fps.RemainingHours(n_max_annos) << std::endl;
                
                //split hdf5 file
                if(hdf5.currentNum("regression") % param_.samples_per_hdf5() == 0) {
                  hdf5.flushCurrent(hdf5_path.string());
                }
                
              } //omp critical
            } //isReasonableCenteredAnno
          } //isReasonableHandPatch
        } //else segmentation
        
        if(VISUALIZE) {
          cv::waitKey(0);
        }
        
      } //rotation for
    } //mirror for
    
  } //data_provider_ for
  
  //store hdf5            
  hdf5.flushCurrent(hdf5_path.string());
  hdf5.writeListFile(list_path.string());
  
  std::cout << "[INFO] CreateData: written " << hdf5.totalNum("regression") << " samples to h5" << std::endl;
  std::cout << "[INFO] CreateData: " << n_invalid_annos << " samples with non-reasonable anno" << std::endl;
}



template <typename Dtype>
void CreateData<Dtype>::createRotations() {
  rotations_.clear();
  
  rotations_.push_back(0.0);
  std::cout << "[INFO] CreateData: add rotation " << 0.0 << std::endl;
 
  if(param_.rotations()) {
    float from = param_.rotations_from();
    float to = param_.rotations_to();
    float step = param_.rotations_step();
    
    while(from <= to) {
      if( std::abs(from) > 0.001) { // don't add 0 again
        rotations_.push_back(from);
        std::cout << "[INFO] CreateData: add rotation " << from << std::endl;
      }
      from += step;
    }
  }
}


template <typename Dtype>
void CreateData<Dtype>::mirror(const cv::Mat_< Dtype >& depth, 
                               const cv::Mat_<Dtype>& ir,
                               const std::vector< cv::Vec3f >& anno, 
                               cv::Mat_< Dtype >& mirrored_depth, 
                               cv::Mat_<Dtype>& mirrored_ir,
                               std::vector< cv::Vec3f >& mirrored_anno) const {
    cv::flip(depth, mirrored_depth, 1);
    cv::flip(ir, mirrored_ir, 1);
    
    mirrored_anno.resize(anno.size());
    for(size_t idx = 0; idx < anno.size(); ++idx) {
        mirrored_anno[idx](0) = depth.cols - anno[idx](0) - 1;
        mirrored_anno[idx](1) = anno[idx](1);
        mirrored_anno[idx](2) = anno[idx](2);
    }
}

template <typename Dtype>
void CreateData<Dtype>::rotate(const cv::Mat_< Dtype >& depth, 
                               const cv::Mat_<Dtype>& ir,
                               const std::vector< cv::Vec3f >& anno, 
                               cv::Mat_< Dtype >& rotated_depth, 
                               cv::Mat_<Dtype> & rotated_ir,
                               std::vector< cv::Vec3f >& rotated_anno, 
                               float rot_deg) const {                                   
    int img_ct_x = depth.cols / 2;
    int img_ct_y = depth.rows / 2;
    
    int tx = img_ct_x - anno[0](0);
    int ty = img_ct_y - anno[0](1);
    cv::Mat_<float> T = (cv::Mat_<float>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::Mat_<float> R = cv::getRotationMatrix2D(cv::Point(img_ct_x, img_ct_y), rot_deg, 1.0);
    
    Dtype bg_value = data_provider_.bgValue();
    
    cv::Mat_<Dtype> tmp;
    cv::warpAffine(depth, tmp, T, cv::Size(depth.cols, depth.rows), CV_INTER_NN, cv::BORDER_CONSTANT, cv::Scalar(bg_value, bg_value, bg_value));
    cv::warpAffine(tmp, rotated_depth, R, cv::Size(depth.cols, depth.rows), CV_INTER_NN, cv::BORDER_CONSTANT, cv::Scalar(bg_value, bg_value, bg_value));
    
    cv::warpAffine(ir, tmp, T, cv::Size(ir.cols, ir.rows), CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::warpAffine(tmp, rotated_ir, R, cv::Size(ir.cols, ir.rows), CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    rotated_anno.resize(anno.size());
    for(int pt_idx = 0; pt_idx < anno.size(); ++pt_idx) {
        cv::Mat_<float> pt = (cv::Mat_<float>(3, 1) << anno[pt_idx](0) + tx, anno[pt_idx](1) + ty, 1.0);
        pt = R * pt;
        
        rotated_anno[pt_idx](0) = pt(0);
        rotated_anno[pt_idx](1) = pt(1);
        rotated_anno[pt_idx](2) = anno[pt_idx](2);
    }
}




template class CreateData<float>;
template class CreateData<double>; 
