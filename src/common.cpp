#include "common.h"

#include <iostream>

#include <boost/format.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <rv/ocv/convert.h>
#include <rv/ocv/visualize.h>
#include <rv/io/ls.h>


//------------------------------------------------------------------------------
void depthmap2pts(const cv::Mat_<float>& dm, const cv::Mat_<float>& ir, const cv::Mat_<cv::Vec3b>& img, const float& max_d, const float& min_ir,
                  std::vector<cv::Vec3f>& pts, std::vector<cv::Vec3f>& colors) {
    
    for(int row = 0; row < dm.rows; ++row) {
        for(int col = 0; col < dm.cols; ++col) {
            float d = dm(row, col);
            if(d < max_d && d > 0 && ir(row, col) >= min_ir) {
                pts.push_back(cv::Vec3f(col, row, d));
                colors.push_back(cv::Vec3f(img(row, col)[0] / 255.0, img(row, col)[1] / 255.0, img(row, col)[2] / 255.0));
            }
        }
    }
}




//------------------------------------------------------------------------------
std::vector<cv::Vec3f> readAnno(boost::filesystem::path anno_path, int n_joints) {
    std::vector<cv::Vec3f> joints(n_joints);
    
    std::ifstream fin(anno_path.c_str());
    
    for(size_t joint_idx = 0; joint_idx < joints.size(); ++joint_idx) {
        fin >> joints[joint_idx][0];
        fin >> joints[joint_idx][1];
        fin >> joints[joint_idx][2];
    }
    
    fin.close();
    
    return joints;
}


//------------------------------------------------------------------------------
void writePoints(std::ostream& out, const std::vector<cv::Vec3f>& pts) {
    for(size_t pts_idx = 0; pts_idx < pts.size(); ++pts_idx) {
        out << pts[pts_idx][0] << " "  << pts[pts_idx][1] << " "  << pts[pts_idx][2];
        if(pts_idx < pts.size() - 1) {
            out << " ";
        }
        else {
            out << std::endl;
        }
    }
}


//------------------------------------------------------------------------------
cv::Mat_<float> readExrDepth(const boost::filesystem::path& p) {
    cv::Mat_<cv::Vec3f> exrC3 = cv::imread(p.string(), -1);
    std::vector<cv::Mat_<float> > exrChannels;
    cv::split(exrC3, exrChannels);
    cv::Mat_<float> depth = exrChannels[0];
    
    return depth;
}


bool readProtoFromTextFile(const boost::filesystem::path& path, google::protobuf::Message* proto) {
  int fd = open(path.c_str(), O_RDONLY);
  if(fd < 0) return false;
  
  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  
  return success;
}



//------------------------------------------------------------------------------
void getIdTs(boost::filesystem::path depth_map_path, std::string& id, std::string& ts) {
    std::string depth_map_filename = depth_map_path.filename().string();
    
    boost::regex expression("(\\d+)_(\\d+)_depth.*");
    boost::smatch what;
    if(boost::regex_match(depth_map_filename, what, expression, boost::match_extra)) {
        id = what[1].str();
        ts = what[2].str();
    }
}

//------------------------------------------------------------------------------
std::string getAnnotator(boost::filesystem::path anno_path) {
    std::string anno_filename = anno_path.filename().string();
    
    boost::regex expression("\\d+_\\d+_anno_(.+).txt");
    boost::smatch what;
    std::string annotator = "unknown";
    if(boost::regex_match(anno_filename, what, expression, boost::match_extra)) {
        annotator = what[1].str();
    }
    
    return annotator;
}

//------------------------------------------------------------------------------
std::vector<boost::filesystem::path> lsAnnoPaths(boost::filesystem::path depth_map_path) {
    
    std::string id, ts;
    getIdTs(depth_map_path, id, ts);
    
    boost::format fmt("%s_%s_.*.txt");
    std::vector<boost::filesystem::path> anno_paths;
    rv::io::ListFiles(depth_map_path.parent_path(), (fmt % id % ts).str(), anno_paths);
    
    return anno_paths;
}

//------------------------------------------------------------------------------
boost::filesystem::path infraredPath(boost::filesystem::path depth_map_path) {
    std::string id, ts;
    getIdTs(depth_map_path, id, ts);
    
    boost::format fmt("%s_%s_ir.csv");
    return depth_map_path.parent_path() / (fmt % id % ts).str();
}

//------------------------------------------------------------------------------
boost::filesystem::path blenderRgbPath(boost::filesystem::path depth_map_path) {
    std::string id, ts, blender_id;
    
    std::string depth_map_filename = depth_map_path.filename().string();
    boost::regex expression("(\\d+)_(\\d+)_depth_(\\d+).*");
    boost::smatch what;
    if(boost::regex_match(depth_map_filename, what, expression, boost::match_extra)) {
        id = what[1].str();
        ts = what[2].str();
        blender_id = what[3].str();
    }
    
    boost::format fmt("%s_%s_rgb_%s.png");
    return depth_map_path.parent_path() / (fmt % id % ts % blender_id).str();
}



cv::Rect enclosingRect(const std::vector< cv::Vec3f >& pts) {
  float min_x = 1e9;
  float min_y = 1e9;
  float max_x = -1e9;
  float max_y = -1e9;
  
  for(size_t idx = 0; idx < pts.size(); ++idx) {
    float x = pts[idx](0);
    float y = pts[idx](1);
    
    if(x < min_x) min_x = x;
    if(y < min_y) min_y = y;
    if(x > max_x) max_x = x;
    if(y > max_y) max_y = y;
  }
  
  cv::Rect rect = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);  
  return rect;
}


