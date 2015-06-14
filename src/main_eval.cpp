#include <iostream>

#include <opencv2/core/core.hpp>

#include <common.h>
#include <patch_extraction.h>
#include <make_data_provider.h>
#include <make_hand_segmentation.h>
#include <make_inference.h>
#include <make_projection.h>
#include <make_patch_extraction.h>
#include <make_preprocess.h>
#include <preprocess.h>

#include <rv/ocv/color.h>
#include <rv/ocv/convert.h>
#include <rv/ocv/visualize.h>
#include <rv/ocv/linalg.h>
#include <rv/ocv/caffe.h>
#include <rv/timer/cpu_timer.h>
#include <rv/stat/core.h>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;


typedef float Dtype;


void extractAlLayers(boost::shared_ptr<caffe::Net<Dtype> > net) {    
  std::vector<std::string> blob_names = net->blob_names();
  
  for(int idx = 0; idx < blob_names.size(); ++idx) {
    std::string blob_name = blob_names[idx];
    boost::shared_ptr<caffe::Blob<Dtype> > blob = net->blob_by_name(blob_name);
    int num = blob->num();
    int channels = blob->channels();
    int height = blob->height();
    int width = blob->width();
    
    if(height > 1 && width > 1) {
      for(int c = 0; c < channels; ++c) {
        cv::Mat_<Dtype> blob_mat = rv::ocv::blob2Mat(blob, 0, c);
        blob_mat = rv::ocv::clamp(blob_mat);
        boost::filesystem::path out_path = (boost::format("layer_%s_c%02d.png") % blob_name % c).str();
        
        cv::Mat_<cv::Vec3b> out_mat = rv::ocv::convert<cv::Vec3b, Dtype>(blob_mat, 255);
        cv::imwrite(out_path.string(), out_mat);
      }
    }
      
  }
}


int main(int argc, char** argv) {
  boost::filesystem::path data_provider_path;
  boost::filesystem::path hand_segmentation_path;
  boost::filesystem::path patch_extractor_path;
  boost::filesystem::path preprocess_path;
  boost::filesystem::path inference_path;

  boost::filesystem::path out_inf_path;
  boost::filesystem::path out_seg_path;

  boost::filesystem::path out_img_prefix;
  int timeout;
  float vis_scale;

  po::options_description desc;
  desc.add_options()
    ("data_provider_path", po::value<std::string>()->required(), "")
    ("hand_segmentation_path", po::value<std::string>()->required(), "")
    ("patch_extractor_path", po::value<std::string>()->required(), "")
    ("preprocess_path", po::value<std::string>()->required(), "")
    ("inference_path", po::value<std::string>()->required(), "")
    ("out_inf_path", po::value<std::string>()->required(), "")
    ("out_seg_path", po::value<std::string>()->default_value(""), "")
    ("out_img_prefix", po::value<std::string>()->default_value(""), "")
    ("timeout", po::value<int>()->default_value(10), "<0 no visualization, =0 wait, >0 no wait")
    ("vis_scale", po::value<float>()->default_value(1), "")
  ;
  
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    data_provider_path = vm["data_provider_path"].as<std::string>();
    hand_segmentation_path = vm["hand_segmentation_path"].as<std::string>();
    patch_extractor_path = vm["patch_extractor_path"].as<std::string>();
    preprocess_path = vm["preprocess_path"].as<std::string>();
    inference_path = vm["inference_path"].as<std::string>();

    out_inf_path = vm["out_inf_path"].as<std::string>();
    out_seg_path = vm["out_seg_path"].as<std::string>();

    out_img_prefix = vm["out_img_prefix"].as<std::string>();
    timeout = vm["timeout"].as<int>();
    vis_scale = vm["vis_scale"].as<float>();
  } catch(std::exception& e) { 
    std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cout << desc << std::endl; 
    return -1; 
  }
  
  //read config
  pose::DataProviderParameter data_provider_param;
  if(!readProtoFromTextFile(data_provider_path, &data_provider_param)) {
    std::cout << "ERROR: parsing data_provider_param" << std::endl;
    return -1;
  }
  pose::SegmentationParameter hand_segmentation_param;
  if(!readProtoFromTextFile(hand_segmentation_path, &hand_segmentation_param)) {
    std::cout << "ERROR: parsing hand_segmentation_param" << std::endl;
    return -1;
  }
  pose::PatchExtractionParameter patch_extraction_param;
  if(!readProtoFromTextFile(patch_extractor_path, &patch_extraction_param)) {
    std::cout << "ERROR: parsing patch_extraction_param" << std::endl;
    return -1;
  }
  pose::PreprocessParameter preprocess_param;
  if(!readProtoFromTextFile(preprocess_path, &preprocess_param)) {
    std::cout << "ERROR: parsing preprocess_param" << std::endl;
    return -1;
  }
  pose::InferenceParameter inference_param;
  if(!readProtoFromTextFile(inference_path, &inference_param)) {
    std::cout << "ERROR: parsing inference_param" << std::endl;
    return -1;
  }

  
  //Init DataProvider
  boost::shared_ptr<DataProvider<Dtype> > data_provider = makeDataProvider<Dtype>(data_provider_param);
  boost::shared_ptr<Projection> projection = data_provider->projection();
  
  //Init HandSegmentation
  boost::shared_ptr<HandSegmentation<Dtype> > hand_segmentation = makeHandSegmentation<Dtype>(hand_segmentation_param, *projection, data_provider->bgValue());
  
  //Init ExtractPatch
  boost::shared_ptr<PatchExtraction<Dtype> > patch_extraction = makePatchExtraction<Dtype>(patch_extraction_param, data_provider->bgValue());
      
  //Init Preprocess
  boost::shared_ptr<Preprocess<Dtype> > preprocess = makePreprocess<Dtype>(preprocess_param);
  
  //load TestPatch
  boost::shared_ptr<Inference<Dtype> > inference = makeInference<Dtype>(inference_param, data_provider->nPts());
      
  
  //start evaluation
  boost::filesystem::create_directories(out_inf_path.parent_path());
  std::ofstream result_out(out_inf_path.c_str());
  boost::shared_ptr<std::ofstream> name_out;
  // if(param.names_path() != "") {
  //   name_out = boost::make_shared<std::ofstream>(param.names_path().c_str());
  // }
  
  boost::shared_ptr<std::ofstream> segmentation_out;
  if(out_seg_path != "") {
    boost::filesystem::create_directories(out_seg_path.parent_path());
    segmentation_out = boost::make_shared<std::ofstream>(out_seg_path.c_str());
  }
  
  
  std::vector<cv::Vec3f> last_es;
  std::vector<cv::Vec3f> last_es_patch;
  std::vector<rv::stat::CumulativeStatistics<double> > stats(data_provider->gt().size());
  rv::stat::CumulativeStatistics<double> stat_segmentation;
  
  rv::timer::CpuBatchedTimer& timer = rv::timer::CpuBatchedTimer::i();
  int snapshot_idx = 0;
  for(;data_provider->hasNext(); data_provider->next()) {
    timer.start("data_provider_depth");
    cv::Mat_<Dtype> depth = data_provider->depth();
    timer.stop("data_provider_depth");
    
    timer.start("data_provider_ir");
    cv::Mat_<Dtype> ir = data_provider->ir();
    timer.stop("data_provider_ir");
    
    timer.start("data_provider_hint");
    cv::Vec3f hint_2d = data_provider->hint2d();        
    timer.stop("data_provider_hint");
    
    timer.start("data_provider_gt");
    std::vector<cv::Vec3f> gt = data_provider->gt();
    timer.stop("data_provider_gt");
    
    timer.start("data_provider_name");
    boost::filesystem::path name = data_provider->depthPath();
    timer.stop("data_provider_name");
            
    std::cout << "processing " << name << " - " << depth.rows << "/" << depth.cols << std::endl;
    if(depth.rows <= 0 || depth.cols <= 0) {
      continue;
    }
    
    timer.start("hand_segmentation");
    std::vector<HandSegmentationResult> segmentations = (*hand_segmentation)(depth, ir, hint_2d);
    std::cout << "  " << segmentations.size() << " segmentations" << std::endl;
    timer.stop("hand_segmentation");
    
    if(segmentations.size() > 0) {
      std::vector<HandPatch<Dtype> > hand_patches = (*patch_extraction)(depth, segmentations[0]);
      (*preprocess)(hand_patches);
      std::cout << "  " << hand_patches.size() << " hand patches " << std::endl;
      
      timer.start("test_patch");
      std::vector<cv::Vec3f> es_patch = (*inference)(hand_patches);
      std::vector<cv::Vec3f> es = hand_patches[0].annoFromHandPatch(es_patch, segmentations[0].roi_);
      std::cout << "  " << es_patch.size() << " estimated joint locations" << std::endl;
        
      last_es = es;
      last_es_patch = es_patch;  
      timer.stop("test_patch");  
      
      if(timeout >= 0) {
        static rv::ocv::ColorMap<Dtype>& cmap = rv::ocv::ColorMapCoolWarm<Dtype>::i();
        rv::ocv::imshow("hand_patch", cmap.Map(hand_patches[0].patch_, -3, 3), vis_scale);
        
//         std::string csv_name = std::string("eval_patch.csv");
//         rv::ocv::writeCsv(csv_name, hand_patches[0].patch_);
      }
    } 
    else {
      std::cout << "[WARNING] no valid hand found" << std::endl;
    }  
    
    //compute dists
    for(size_t last_es_idx = 0; last_es_idx < last_es.size(); ++last_es_idx) {
      cv::Vec3f g = projection->to3D(gt[last_es_idx]);
      cv::Vec3f e = projection->to3D(last_es[last_es_idx]);
      double dist = rv::ocv::dist2(g, e);
      stats[last_es_idx](dist);
    }
    
    //write
    timer.start("write");
    for(int row = 0; row < last_es.size(); ++row) {
      result_out << last_es[row](0) << " ";
      result_out << last_es[row](1) << " ";
      result_out << last_es[row](2) << " ";
    }
    result_out << std::endl;
    
    if(name_out) {
      (*name_out) << name << std::endl;
    }
    
    if(segmentation_out) {
      cv::Rect gt_rect = enclosingRect(gt);
      cv::Rect es_rect(-2, -2, 1, 1);
      if(segmentations.size() > 0) {
        es_rect = segmentations[0].roi_;
      }
      
      cv::Rect intersection = gt_rect & es_rect;
      float r = float(intersection.area()) / float(gt_rect.area() + es_rect.area() - intersection.area());
      
      (*segmentation_out) << gt_rect.x << " " << gt_rect.y << " " << gt_rect.width << " " << gt_rect.height << " " 
         << es_rect.x << " " << es_rect.y << " " << es_rect.width << " " << es_rect.height 
         << " " << r << std::endl;
         
      stat_segmentation.add(r);
    }
    timer.stop("write");
    
    
    //visualize
    if(timeout >= 0) {
      timer.start("visualize");
      cv::Mat_<cv::Vec3b> im3 = annoShow(depth.clone(), last_es, data_provider->bgValue());
      rv::ocv::imshow("anno", im3, vis_scale);

      if(out_img_prefix != "") {
        boost::filesystem::create_directories(out_img_prefix.parent_path());

        std::string snapshot_filename = (boost::format("%s_%08d.png") % out_img_prefix.string() % snapshot_idx).str();
        std::cout << "  save snapshot \"" << snapshot_filename << "\"" << std::endl;
        cv::imwrite(snapshot_filename, im3);
        snapshot_idx++;
      }
     
      cv::waitKey(timeout);
      timer.stop("visualize");
      
//    extractAlLayers(net);
    }
  } // for samples/images
  
  //print stats
  std::cout << "all: " << rv::stat::CumulativeStatistics<double>::mean(stats) 
    << " +- " << rv::stat::CumulativeStatistics<double>::std(stats) << std::endl;
  for(size_t idx = 0; idx < stats.size(); ++idx) {
    std::cout << "joint_" << idx << ": " << stats[idx].mean() << " +- " << stats[idx].std() << std::endl;
  }
  
  //close results file
  result_out.close();
  if(name_out) {
    name_out->close();
  }
  if(segmentation_out) {
    std::cout << "segmentation: " << stat_segmentation.mean() << " +- " << stat_segmentation.std()  << std::endl;
    segmentation_out->close();
  }
  std::cout << "results written to: " << out_inf_path << std::endl;
  
  timer.print(std::cout);
  
  return 0;
}
