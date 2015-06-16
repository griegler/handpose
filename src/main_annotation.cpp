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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/make_shared.hpp>

#include <rv/io/ls.h>
#include <rv/ocv/convert.h>
#include <rv/ocv/io/csv.h>

#include <common.h>
#include <annotation_viewer.h>
#include <make_projection.h>

#include <pose.pb.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define SEQ_FORMAT_CSV 0
#define SEQ_FORMAT_BLENDER 1



//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string anno_name;
    boost::filesystem::path sequence_path;
    int joint_format;
    int sequence_format;
    int skip;
    bool skip_annotated;
    bool create_constraint_anno;
    int min_ir;
    int max_ir;
    int max_d;
    
    po::options_description desc;
    desc.add_options()
        ("anno_name,a", po::value<std::string>()->required(), "")
        ("sequence_path,s", po::value<std::string>()->required(), "")
        ("joint_format", po::value<int>()->default_value(0), "0 := 20 joints, 1 := 16 (DTang)")
        ("sequence_format", po::value<int>()->default_value(0), "0 := csv, 1 := blender exr")
        ("skip", po::value<int>()->default_value(0), "")
        ("skip_annotated", po::value<bool>()->default_value(true), "")
        ("create_constraint_anno", po::value<bool>()->default_value(false), "")
        ("min_ir", po::value<int>()->default_value(200), "is always used for depthmap2pts")
        ("max_ir", po::value<int>()->default_value(-1), "if -1, the ir img is clamped to [min(ir), max(ir)]")
        ("max_d", po::value<int>()->default_value(1500), "used for depthmap2pts")
    ;
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        po::notify(vm);
        
        anno_name = vm["anno_name"].as<std::string>();  
        sequence_path = vm["sequence_path"].as<std::string>();     
        joint_format = vm["joint_format"].as<int>();      
        sequence_format = vm["sequence_format"].as<int>();    
        skip = vm["skip"].as<int>();     
        skip_annotated = vm["skip_annotated"].as<bool>();    
        create_constraint_anno = vm["create_constraint_anno"].as<bool>();    
        
        min_ir = vm["min_ir"].as<int>();   
        max_ir = vm["max_ir"].as<int>();   
        max_d = vm["max_d"].as<int>();   
    } catch(std::exception& e) { 
        std::cout << "ERROR: " << e.what() << std::endl << std::endl; 
        std::cout << desc << std::endl; 
        return false; 
    }
    
    //----------
    //check existince of constraint anno and load it if needed
    std::vector<cv::Vec3f> constraint_anno; 
    boost::filesystem::path constraint_anno_path = sequence_path / "anno_constraint.txt";
    if(!create_constraint_anno) {
        if(boost::filesystem::exists(constraint_anno_path)) {
            if(joint_format == 0) { //mine_20
                constraint_anno = readAnno(constraint_anno_path, 20);
            }
            else if(joint_format == 1) { //dtang 16
                constraint_anno = readAnno(constraint_anno_path, 16);
            }
        }
        else {
            std::cout << "[ERROR] constraint anno not found at: " << constraint_anno_path << std::endl;
            return -1; 
        }
    }
    
    //----------
    //load calibration
    boost::filesystem::path calib_path = sequence_path / "calibration_depth.txt";
    if(!boost::filesystem::exists(calib_path)) {
        std::cout << "calibration file does not exist: " << calib_path << std::endl;
        return -1;
    }
    
    //Init Projection
    pose::ProjectionParameter projection_param;
    projection_param.set_type(pose::ProjectionParameter::ORTHOGRAPHIC);
    projection_param.set_calib_path(calib_path.string());
    boost::shared_ptr<Projection> projection = makeProjection(projection_param, calib_path);
    
    
    //----------
    //list depth map files
    std::vector<boost::filesystem::path> depth_map_paths;
    if(sequence_format == SEQ_FORMAT_CSV) {
        rv::io::ListFiles(sequence_path, ".*depth.csv", depth_map_paths);
    }
    else if(sequence_format == SEQ_FORMAT_BLENDER) {
        rv::io::ListFiles(sequence_path, ".*depth_.*.exr", depth_map_paths);
    }
    std::sort(depth_map_paths.begin(), depth_map_paths.end());
    
    //---------
    //process depth maps individually
    std::vector<cv::Vec3f> last_anno;
    int depth_map_idx = skip;
    while(depth_map_idx < depth_map_paths.size()) {
        boost::filesystem::path depth_map_path = depth_map_paths[depth_map_idx];
        std::cout << "depth map path: " << depth_map_path << std::endl;
        
        //--------
        //load annotations
        std::vector<boost::filesystem::path> anno_paths = lsAnnoPaths(depth_map_path);
        std::cout << "found " << anno_paths.size() << " annotations" << std::endl;
        
        //skip annotation if enabled and there exists an annotation
        if(skip_annotated && anno_paths.size() > 1) {
            depth_map_idx++;
            continue;
        }
        
        std::vector<std::vector<cv::Vec3f> > annos;
        if(last_anno.size() > 0) {
            annos.push_back(last_anno);
        }
        if(constraint_anno.size() > 0) {
            annos.push_back(constraint_anno);
        }
        for(size_t anno_path_idx = 0; anno_path_idx < anno_paths.size(); ++anno_path_idx) {
            boost::filesystem::path anno_path = anno_paths[anno_path_idx];
            std::cout << "  " << anno_path << std::endl;
            
            std::vector<cv::Vec3f> anno; 
            if(joint_format == 0) { //mine_20
                anno = readAnno(anno_path, 20);
            }
            else if(joint_format == 1) { //dtang 16
                anno = readAnno(anno_path, 16);
            }
            
            annos.push_back(anno);
        }
        
        //--------
        //load depthmap / ir / img
        cv::Mat_<float> dm, ir;
        cv::Mat_<cv::Vec3b> img;
        if(sequence_format == SEQ_FORMAT_CSV) {
            dm = rv::ocv::ReadCsv<float>(depth_map_path);
            cv::medianBlur(dm, dm, 3);
            ir = rv::ocv::ReadCsv<float>(infraredPath(depth_map_path));
            
            if(max_ir < 0) {
              img = rv::ocv::convert<cv::Vec3b, float>(rv::ocv::clamp(ir), 255);
            }
            else {
              img = rv::ocv::convert<cv::Vec3b, float>(rv::ocv::clamp(ir, float(min_ir), float(max_ir)), 255);
            }
        }
        else if(sequence_format == SEQ_FORMAT_BLENDER) {
            dm = readExrDepth(depth_map_path) * 1000;
            ir = cv::Mat_<float>::zeros(dm.rows, dm.cols);
            for(int row = 0; row < dm.rows; ++row) {
                for(int col = 0; col < dm.cols; ++col) {
                    if(dm(row, col) > 0.0 && dm(row, col) < 10000.0) {
                        ir(row, col) = 1000;
                    }
                }
            }
            img = cv::imread(blenderRgbPath(depth_map_path).string());
        }        
                
        // Read command lines arguments.
        QApplication application(argc, argv);

        // Instantiate the 3d viewer.
        std::vector<cv::Vec3f> pts3, pts3_colors;
        depthmap2pts(dm, ir, img, max_d, min_ir, pts3, pts3_colors); 
        projection->to3D(pts3, pts3);
        
        boost::shared_ptr<AnnotationViewer> viewer;
        if(constraint_anno.size() == 0) {
            viewer = boost::make_shared<Mine20AnnotationViewer>(pts3, pts3_colors, img, annos, annos[0], *projection);
        }
        else {
            viewer = boost::make_shared<Mine20AnnotationViewer>(pts3, pts3_colors, img, annos, constraint_anno, *projection);
        }
        
#ifdef __linux__
        viewer->setWindowTitle(depth_map_path.filename().c_str());
#endif
        viewer->show();
                
        // Run main loop.
        application.exec();
        
        
        //--------
        //catch result 
        std::vector<cv::Vec3f> anno = viewer->getAnno();
        last_anno = anno;
        
        //------
        //store result
        if(viewer->getChanged()) {
            std::string id, ts;
            getIdTs(depth_map_path, id, ts);
            
            boost::filesystem::path anno_path;
            if(create_constraint_anno) {
                anno_path = constraint_anno_path;
            }
            else {
                static boost::format fmt("%s_%s_anno_%s.txt");
                anno_path = depth_map_path.parent_path() / (fmt % id % ts % anno_name).str();
            }
            
            std::ofstream fout(anno_path.c_str());
            writePoints(fout, anno);
            fout.close();
            
            std::cout << "saved anno to " << anno_path << std::endl;
            if(create_constraint_anno) {
                return 0;
            }
        }
        
        //next/prev depthmap;
        depth_map_idx += viewer->getReturnDirection();
        if(depth_map_idx < 0) {
            depth_map_idx = 0;
        }
    }
    
    return 0;
}

