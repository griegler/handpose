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

#ifndef ANNOTATION_VIEWER_H
#define ANNOTATION_VIEWER_H

#include <qapplication.h>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/manipulatedFrame.h>

#include <opencv2/core/core.hpp>

#include "projection.h"


class AnnotationViewer : public QGLViewer {
public:
    AnnotationViewer(const std::vector<cv::Vec3f> pts3, 
        const std::vector<cv::Vec3f> pts3_colors, 
        const cv::Mat_<cv::Vec3b>& img, 
        const std::vector<std::vector<cv::Vec3f> >& anno_proposals,
        const std::vector<cv::Vec3f>& constraint_anno,
        const Projection& projection) :
            pts3_(pts3), pts3_colors_(pts3_colors), img_(img), 
            anno_proposals_(anno_proposals), constraint_anno_(constraint_anno), 
            projection_(projection), active_anno_proposal_(0), 
            active_joint_idx_(0), draw_3d_(false), free_edit_mode_(true), 
            project_constrained_depth_mode_(0), deviation_tolerance_(2.0), 
            changed_(false), size_3d_pts_(2.0), size_3d_annotation_(3.0) { 
                
        anno_ = anno_proposals_[active_anno_proposal_];
        
        color_pts_          = cv::Vec3f(119.0, 125.0, 210.0) / 255.0;
        color_joints_       = cv::Vec3f(221.0, 144.0, 144.0) / 255.0;
        color_joint_active_ = cv::Vec3f( 63.0, 135.0,  68.0) / 255.0;
        color_skelet_       = cv::Vec3f( 77.0,  77.0,  77.0) / 255.0;
        color_background_   = cv::Vec3f(255.0, 255.0, 255.0) / 255.0;
    }
    
    virtual ~AnnotationViewer() {}
    
protected : 
  virtual int parentJointIdx(int joint_idx) = 0;
    
  virtual void glColorVec3f(const cv::Vec3f& v);
  
  virtual float dist(const cv::Vec3f& v1, const cv::Vec3f& v2);
  
  virtual cv::Vec3f cartesian2spherical(const cv::Vec3f& c_cart, 
      const cv::Vec3f& p_cart);
  virtual cv::Vec3f spherical2cartesian(const cv::Vec3f& c_cart, 
      const cv::Vec3f& p_spher);
  
  virtual void moveAnnoX(int joint_idx, float t);
  virtual void moveAnnoY(int joint_idx, float t);
  virtual void moveAnnoZ(int joint_idx, float t);
  virtual void moveAnnoD(int joint_idx, float t);
  virtual void moveAnnoR(int joint_idx, float t);
  virtual void moveAnnoTheta(int joint_idx, float t);
  virtual void moveAnnoPhi(int joint_idx, float t);
  
  virtual void moveAnnoAllDependent(int joint_idx, const cv::Vec3f& t3);
  virtual void projectAnnoToConstraints();
  
  virtual void rotateAnno(float x, float y, float z);
  
    
  virtual void generateSphericalConstraints(); 
    
  virtual void init();
  
  
  virtual void drawHandSkeleton(const std::vector<cv::Vec3f>& anno3);
  virtual void drawPointWithCoords(const cv::Vec3f& pos, float radius, 
      const cv::Vec3f& color, bool coord);
  virtual void drawSphericalConstraint(const std::vector<cv::Vec3f>& anno3);
  virtual void drawIn3d();
  virtual void drawIn2d();
  
  virtual void draw();
  
  virtual void keyPressEvent(QKeyEvent* e);
  virtual void mousePressEvent(QMouseEvent* e);
  
  
  virtual QString helpString() const {
    return QString("...");
  }
  

public:
  virtual std::vector<cv::Vec3f> getAnno() const {
    return anno_;
  }
  
  virtual bool getChanged() const {
    return changed_;
  }
  
  virtual int getReturnDirection() const {
    return return_direction_;
  }
  
protected:    
    std::vector<cv::Vec3f> pts3_;
    std::vector<cv::Vec3f> pts3_colors_;
    const cv::Mat_<cv::Vec3b> img_;
    const std::vector<std::vector<cv::Vec3f> > anno_proposals_;
    const std::vector<cv::Vec3f> constraint_anno_;
    const Projection& projection_;
    
    int active_anno_proposal_;
    
    std::vector<cv::Vec3f> anno_;
    std::vector<float> spherical_constraints_;
    int active_joint_idx_;
    
    bool draw_3d_;
    bool free_edit_mode_;
    int project_constrained_depth_mode_;
    float deviation_tolerance_;
    
    bool changed_;
    int return_direction_;
    
    float size_3d_pts_;
    float size_3d_annotation_;
    
    cv::Vec3f color_pts_;
    cv::Vec3f color_joints_;
    cv::Vec3f color_joint_active_;
    cv::Vec3f color_skelet_;
    cv::Vec3f color_background_;
};


class Mine20AnnotationViewer : public AnnotationViewer {
public:
    Mine20AnnotationViewer(const std::vector<cv::Vec3f> pts3, 
        const std::vector<cv::Vec3f> pts3_colors, 
        const cv::Mat_<cv::Vec3b>& img, 
        const std::vector<std::vector<cv::Vec3f> >& anno_proposals,
        const std::vector<cv::Vec3f>& constraint_anno,
        const Projection& projection) :
        AnnotationViewer(pts3, pts3_colors, img, anno_proposals, 
            constraint_anno, projection) {}
    
protected:
    virtual int parentJointIdx(int joint_idx) {
        if(joint_idx == 4 || joint_idx == 8 || joint_idx == 12 
            || joint_idx == 16) {
          return 0;
        }
        else {
          return joint_idx - 1;
        }
    }
};

#endif
