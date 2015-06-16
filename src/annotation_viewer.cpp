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

#include "annotation_viewer.h"

#include <opencv2/highgui/highgui.hpp>
#include <boost/foreach.hpp>

#include <rv/ocv/convert.h>

#include <common.h>


//------------------------------------------------------------------------------
// Utility Functions
float AnnotationViewer::dist(const cv::Vec3f& v1, const cv::Vec3f& v2) {
    cv::Vec3f d = v1 - v2;
    return sqrt( d(0)*d(0) + d(1)*d(1) + d(2)*d(2) );
}

void AnnotationViewer::glColorVec3f(const cv::Vec3f& v) {
  glColor3f(v(0), v(1), v(2));
}


cv::Vec3f AnnotationViewer::cartesian2spherical(const cv::Vec3f& c_cart, const cv::Vec3f& p_cart) {
    //project point to center
    cv::Vec3f parent = c_cart;
    cv::Vec3f child = p_cart;
    
    cv::Vec3f t = parent;
    parent = parent - t;
    child = child - t;
    
    //convert to spherical coordinates
    cv::Vec3f p_spher;
    p_spher(0) = dist(parent, child);
    p_spher(1) = acos( child(2) / p_spher(0) );
    p_spher(2) = atan2(child(1), child(0));
    
    return p_spher;
}

cv::Vec3f AnnotationViewer::spherical2cartesian(const cv::Vec3f& c_cart, const cv::Vec3f& p_spher) {
    //convert to cartesian
    cv::Vec3f p_cart;
    p_cart(0) = p_spher(0) * sin(p_spher(1)) * cos(p_spher(2));
    p_cart(1) = p_spher(0) * sin(p_spher(1)) * sin(p_spher(2));
    p_cart(2) = p_spher(0) * cos(p_spher(1));
    
    //move points back
    p_cart = p_cart + c_cart;
    
    return p_cart;
}



void AnnotationViewer::moveAnnoX(int joint_idx, float t) {
    cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
    a3(0) += t;
    anno_[joint_idx] = projection_.to2D(a3);
    changed_ = true;
}

void AnnotationViewer::moveAnnoY(int joint_idx, float t) {
    cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
    a3(1) += t;
    anno_[joint_idx] = projection_.to2D(a3);
    changed_ = true;
}

void AnnotationViewer::moveAnnoZ(int joint_idx, float t) {
    cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
    a3(2) += t;
    anno_[joint_idx] = projection_.to2D(a3);
    changed_ = true;
}

void AnnotationViewer::moveAnnoD(int joint_idx, float t) {
    anno_[joint_idx](2) += t;
    changed_ = true;
}

void AnnotationViewer::moveAnnoR(int joint_idx, float t) {
    cv::Vec3f child3 = projection_.to3D(anno_[joint_idx]);
    cv::Vec3f parent3 = projection_.to3D(anno_[parentJointIdx(joint_idx)]);
    
    cv::Vec3f spherical = cartesian2spherical(parent3, child3);
    spherical(0) += t;
    
    anno_[joint_idx] = projection_.to2D(spherical2cartesian(parent3, spherical));
    changed_ = true;
}

void AnnotationViewer::moveAnnoTheta(int joint_idx, float t) {
    cv::Vec3f child3 = projection_.to3D(anno_[joint_idx]);
    cv::Vec3f parent3 = projection_.to3D(anno_[parentJointIdx(joint_idx)]);
    
    cv::Vec3f spherical = cartesian2spherical(parent3, child3);
    spherical(1) += t;
    
    anno_[joint_idx] = projection_.to2D(spherical2cartesian(parent3, spherical));
    changed_ = true;
}

void AnnotationViewer::moveAnnoPhi(int joint_idx, float t) {
    cv::Vec3f child3 = projection_.to3D(anno_[joint_idx]);
    cv::Vec3f parent3 = projection_.to3D(anno_[parentJointIdx(joint_idx)]);
    
    cv::Vec3f spherical = cartesian2spherical(parent3, child3);
    spherical(2) += t;
    
    anno_[joint_idx] = projection_.to2D(spherical2cartesian(parent3, spherical));
    changed_ = true;
}


void AnnotationViewer::moveAnnoAllDependent(int joint_idx, const cv::Vec3f& t3) {
    if(joint_idx == 0) { //if joint_idx is root node
        for(joint_idx = 1; joint_idx < anno_.size(); ++joint_idx) {
            cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
            a3 += t3;
            anno_[joint_idx] = projection_.to2D(a3);
        }
    }
    else {
        joint_idx++; //start with child
        while(parentJointIdx(joint_idx) != 0) {
            cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
            a3 += t3;
            anno_[joint_idx] = projection_.to2D(a3);
            joint_idx++;
        }
    }
    
    changed_ = true;
}

void AnnotationViewer::projectAnnoToConstraints() {
    for(size_t joint_idx = 1; joint_idx < anno_.size(); ++joint_idx) {
        cv::Vec3f child3 = projection_.to3D(anno_[joint_idx]);
        cv::Vec3f parent3 = projection_.to3D(anno_[parentJointIdx(joint_idx)]);
        
        cv::Vec3f child_spher = cartesian2spherical(parent3, child3);
        child_spher(0) = spherical_constraints_[joint_idx - 1];
        
        anno_[joint_idx] = projection_.to2D(spherical2cartesian(parent3, child_spher));
    }
    changed_ = true;
}


void AnnotationViewer::rotateAnno(float x, float y, float z) {
    cv::Vec3f t = projection_.to3D(anno_[0]);
    
    cv::Matx33f rotx(1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
    cv::Matx33f roty(cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
    cv::Matx33f rotz(cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);
    cv::Matx33f rot = rotx * roty * rotz;
    
    for(size_t joint_idx = 1; joint_idx < anno_.size(); ++joint_idx) {
        cv::Vec3f a3 = projection_.to3D(anno_[joint_idx]);
        a3 = a3 - t;
        
        a3 = rot * a3;
        
        a3 = a3 + t;
        anno_[joint_idx] = projection_.to2D(a3);
    }
    changed_ = true;
}





//------------------------------------------------------------------------------
// Init functions 

void AnnotationViewer::generateSphericalConstraints(){
    std::vector<cv::Vec3f> anno3d;
    projection_.to3D(constraint_anno_, anno3d);
    
    spherical_constraints_.resize(anno3d.size() - 1);
    
    for(int child_idx = 1; child_idx < anno3d.size(); ++child_idx) {
        int parent_idx = parentJointIdx(child_idx);
        spherical_constraints_[child_idx - 1] = dist(anno3d[child_idx], anno3d[parent_idx]);
    }
}


void AnnotationViewer::init() {
    //-------
    // Restore previous viewer state.
    restoreStateFromFile();

    //scene radius and center
    cv::Vec3f center(0,0,0);
    for(size_t i = 0; i < pts3_.size(); ++i) {
    center += pts3_[i];
    }
    center = center / float(pts3_.size());

    float radius = 0;
    std::vector<std::pair<float, int> > distances;
    distances.reserve(pts3_.size());

    for(size_t i = 0; i < pts3_.size(); ++i) {
        double dist = cv::norm(center, pts3_[i]);
        distances.push_back(std::make_pair(dist, i));
    }
    std::sort(distances.begin(), distances.end());

    int start_idx = (float) distances.size() * 0.03;
    int end_idx = (float) distances.size() * 0.97;

    center = cv::Vec3f(0, 0, 0);
    for (size_t i = start_idx; i < end_idx; i++) {
        center += pts3_[distances[i].second];
    }
    center =  center / float(end_idx - start_idx);
    distances.clear();

    for (size_t i = 0; i < pts3_.size(); i++) {
        double dist = cv::norm(center, pts3_[i]);
        distances.push_back(std::make_pair(dist, i));
    }
    std::sort(distances.begin(), distances.end());

    if (distances.size() > 0) {
        radius = distances[float(distances.size()) * 0.98].first;
    }

    setSceneCenter(qglviewer::Vec(center[0], center[1], center[2]));
    setSceneRadius(radius);

    std::cout << "scene center: " << center << std::endl;
    std::cout << "scene radius: " << radius << std::endl;

    //-------
    // load image to texture for 2d view 
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ); //twice needed!
    cv::flip(img_, img_, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, img_.cols, img_.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, img_.data);
    
    //set background color
    setBackgroundColor(QColor(color_background_(0) * 255, color_background_(1) * 255, color_background_(2) * 255));
    
    //-------
    generateSphericalConstraints();
}





//------------------------------------------------------------------------------
// draw methods
void AnnotationViewer::drawPointWithCoords(const cv::Vec3f& pos, float radius, const cv::Vec3f& color, bool coord) {
  glColorVec3f(color);

  glPushMatrix();
  glTranslatef(pos[0], pos[1], pos[2]);

  glBegin(GL_LINE_LOOP);
  GLUquadricObj* quadric = gluNewQuadric();
  gluQuadricDrawStyle(quadric, GLU_FILL);
  gluSphere(quadric, radius, 10, 10);
  gluDeleteQuadric(quadric);
  glEnd();

  glPopMatrix();

  //draw coord 
  if(coord) {
    float cylinder_radius = size_3d_annotation_ * 0.25;
    float cylinder_height = size_3d_annotation_ * 3;
    glColor3f(1, 0, 0);
    glPushMatrix();
    glTranslatef(pos[0], pos[1], pos[2]);
    glPushMatrix();
    glRotated(-90.0f, 0, 1.0f, 0.0f);
    GLUquadricObj *obj_x = gluNewQuadric();
    gluCylinder(obj_x, cylinder_radius, cylinder_radius, cylinder_height, 30, 30);
    glPopMatrix();
    glPopMatrix();

    glColor3f(0,1,0);
    glPushMatrix();
    glTranslatef(pos[0], pos[1], pos[2]);
    glPushMatrix();
    glRotated(-90.0f, 1.0f, 0.0f, 0.0f);
    GLUquadricObj *obj_y = gluNewQuadric();
    gluCylinder(obj_y, cylinder_radius, cylinder_radius, cylinder_height, 30, 30);
    glPopMatrix();
    glPopMatrix();

    glColor3f(0,0,1);
    glPushMatrix();
    glTranslatef(pos[0], pos[1], pos[2]);
    glPushMatrix();
    glRotated(-90.0f, 0.0f, 0.0f, 1.0f);
    GLUquadricObj *obj_z = gluNewQuadric();
    gluCylinder(obj_z, cylinder_radius, cylinder_radius, cylinder_height, 30, 30);
    glPopMatrix();
    glPopMatrix();
  }
}


void AnnotationViewer::drawHandSkeleton(const std::vector< cv::Vec3f >& anno3) {    
    //draw joint connections
    glLineWidth(size_3d_annotation_ * 1.5); 
    glBegin(GL_LINES);
    for(int child_idx = 1; child_idx < anno3.size(); ++child_idx) {
        int parent_idx = parentJointIdx(child_idx);
        cv::Vec3f parent = anno3[parent_idx];
        cv::Vec3f child = anno3[child_idx];
        float deviation = abs( dist(child, parent) - spherical_constraints_[child_idx - 1] );
        if(deviation > deviation_tolerance_) {
            glColorVec3f(color_skelet_); //TODO: define color
        }
        else {
            glColorVec3f(color_skelet_);
        }
        glVertex3f(parent[0], parent[1], parent[2]);
        glVertex3f(child[0], child[1], child[2]);
    }
    glEnd();
}

void AnnotationViewer::drawSphericalConstraint(const std::vector< cv::Vec3f >& anno3) { 
    //draw spherical constraint circles for active joint
    if(active_joint_idx_ == 0) {
        cv::Vec3f c = anno3[0];
        
        glDisable(GL_LIGHTING);
        
        //draw circles 
        glPointSize(2.5);
        glBegin(GL_POINTS);
                
        glColor3f(1, 0.5, 0.5);
        cv::Vec3f p = anno3[0] + cv::Vec3f(0, 50, 0);
        for(float alpha = 0.0; alpha < 2 * M_PI; alpha += 0.01) {
            cv::Matx33f rot(1, 0, 0, 0, cos(alpha), -sin(alpha), 0, sin(alpha), cos(alpha));
            cv::Vec3f plt = rot * (p - c) + c;
            glVertex3f(plt(0), plt(1), plt(2));
        }
        
        glColor3f(0.5, 1, 0.5);
        p = anno3[0] + cv::Vec3f(50, 0, 0);
        for(float alpha = 0.0; alpha < 2 * M_PI; alpha += 0.01) {
            cv::Matx33f rot(cos(alpha), 0, sin(alpha), 0, 1, 0, -sin(alpha), 0, cos(alpha));
            cv::Vec3f plt = rot * (p - c) + c;
            glVertex3f(plt(0), plt(1), plt(2));
        }
        
        glColor3f(0.5, 0.5, 1);
        p = anno3[0] + cv::Vec3f(50, 0, 0);
        for(float alpha = 0.0; alpha < 2 * M_PI; alpha += 0.01) {
            cv::Matx33f rot(cos(alpha), -sin(alpha), 0, sin(alpha), cos(alpha), 0, 0, 0, 1);
            cv::Vec3f plt = rot * (p - c) + c;
            glVertex3f(plt(0), plt(1), plt(2));
        }
        glEnd();
        
        glEnable(GL_LIGHTING);
    }
    else {
        //project point to center
        cv::Vec3f child = anno3[active_joint_idx_];
        cv::Vec3f parent = anno3[parentJointIdx(active_joint_idx_)];

        cv::Vec3f child_spher = cartesian2spherical(parent, child);

        glDisable(GL_LIGHTING);
        //draw circles 
        
        glPointSize(2.5);
        glBegin(GL_POINTS);    

        glColor3f(1, 1, 0.5);
        for(float theta = 0.0; theta <= M_PI; theta += 0.01) {
            cv::Vec3f plt = child_spher;
            plt(1) = theta;
            plt = spherical2cartesian(parent, plt);
            glVertex3f(plt(0), plt(1), plt(2));
        }

        glColor3f(0.5, 0.5, 1);
        for(float phi = 0.0; phi < 2 * M_PI; phi += 0.01) {
            cv::Vec3f plt = child_spher;
            plt(2) = phi;
            plt = spherical2cartesian(parent, plt);
            glVertex3f(plt(0), plt(1), plt(2));
        }

        glEnd();
        
        //draw cursor lines
        glLineWidth(size_3d_annotation_ * 3.5); 
        glBegin(GL_LINES);
        
        //theta
        cv::Vec3f p = child_spher;
        p(1) += 0.01;
        p = child + (spherical2cartesian(parent, p) - child) * 20;
        glColor3f(1, 1, 0.5);
        glVertex3f(child[0], child[1], child[2]);
        glVertex3f(p[0], p[1], p[2]);
        
        //phi
        p = child_spher;
        p(2) += 0.01;
        p = child + (spherical2cartesian(parent, p) - child) * 20;
        glColor3f(0.5, 0.5, 1);
        glVertex3f(child[0], child[1], child[2]);
        glVertex3f(p[0], p[1], p[2]);
        
        glEnd();
        
        glEnable(GL_LIGHTING);
    }
}

void AnnotationViewer::drawIn3d() {
    //draw depth points
    glDisable(GL_LIGHTING);
    glPointSize(size_3d_pts_);
    glBegin(GL_POINTS);    
    for(size_t idx = 0; idx < pts3_.size(); ++idx) {
        glColorVec3f(color_pts_);
//         glColor3f(pts3_colors_[idx][0], pts3_colors_[idx][1], pts3_colors_[idx][2]);
        glVertex3f(pts3_[idx][0], pts3_[idx][1], pts3_[idx][2]);
    }
    glEnd();

    //draw joints
    std::vector<cv::Vec3f> anno3;
    projection_.to3D(anno_, anno3);

    for(size_t idx = 0; idx < anno3.size(); ++idx) {
        if(idx == active_joint_idx_) {
            drawPointWithCoords(anno3[idx], size_3d_annotation_, color_joint_active_, true);
        }
        else {
            drawPointWithCoords(anno3[idx], size_3d_annotation_, color_joints_, false);
        }
    }
    drawHandSkeleton(anno3);

    //spherical movements
//     drawSphericalConstraint(anno3);
    
    glEnable(GL_LIGHTING);
}

void AnnotationViewer::drawIn2d() {
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);

    startScreenCoordinatesSystem(true);
    glNormal3f(0.0, 0.0, 1.0);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);        glVertex2i(0, 0);
    glTexCoord2f(0.0, 1.0);        glVertex2i(0, height());
    glTexCoord2f(1.0, 1.0);        glVertex2i(width(), height());
    glTexCoord2f(1.0, 0.0);        glVertex2i(width(), 0);
    glEnd();
    stopScreenCoordinatesSystem();
    
    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    
    
    //draw points
    startScreenCoordinatesSystem(true);
    
    glPointSize(3 * size_3d_annotation_);
    glBegin(GL_POINTS);    
    for(size_t idx = 0; idx < anno_.size(); ++idx) {
        int x = float(anno_[idx][0]) / float(img_.cols) * float(width());
        int y = float(anno_[idx][1]) / float(img_.rows) * float(height());

        if(idx == active_joint_idx_) {
            glColorVec3f(color_joint_active_);
        }
        else {
            glColorVec3f(color_joints_);
        }
        glVertex2i(x, height() - y);
    }
    glEnd();

    //draw skeleton
    glLineWidth(0.5 * size_3d_annotation_); 
    glBegin(GL_LINES);
    for(int child_idx = 1; child_idx < anno_.size(); ++child_idx) {
        int parent_idx = parentJointIdx(child_idx);
        
        cv::Vec3f child = anno_[child_idx];
        cv::Vec3f parent = anno_[parent_idx];
        cv::Vec3f child3 = projection_.to3D(child);
        cv::Vec3f parent3 = projection_.to3D(parent);
        float deviation = abs( dist(child3, parent3) - spherical_constraints_[child_idx - 1] );
        if(deviation > deviation_tolerance_) {
            glColorVec3f(color_skelet_); //TODO: define color
        }
        else {
            glColorVec3f(color_skelet_);
        }
        
        glVertex2i(float(parent[0]) / float(img_.cols) * float(width()), 
                   height() - float(parent[1]) / float(img_.rows) * float(height()));
        glVertex2i(float(child[0]) / float(img_.cols) * float(width()), 
                   height() - float(child[1]) / float(img_.rows) * float(height()));
    }
    glEnd();

    stopScreenCoordinatesSystem();
}

void AnnotationViewer::draw() {       
    if(draw_3d_) {
        drawIn3d();
    }
    else {
        drawIn2d();
    }
}





//------------------------------------------------------------------------------
// Event methods
void AnnotationViewer::keyPressEvent(QKeyEvent* e) {
    const Qt::KeyboardModifiers modifiers = e->modifiers();
    
    if(modifiers == Qt::NoButton) {
        if(free_edit_mode_) {
            // -x
            if(e->key() == Qt::Key_Left) {
                moveAnnoX(active_joint_idx_, 1.0);
            }
            // +x
            if(e->key() == Qt::Key_Right) {
                moveAnnoX(active_joint_idx_, -1.0);
            }
            // -y
            if(e->key() == Qt::Key_Down) {
                moveAnnoY(active_joint_idx_, -1.0);
            }
            // +y
            if(e->key() == Qt::Key_Up) {
                moveAnnoY(active_joint_idx_, 1.0);
            }
            // -z
            if(e->key() == Qt::Key_Plus) {
                moveAnnoZ(active_joint_idx_, -1.0);
            }
            // +z
            if(e->key() == Qt::Key_Minus) {
                moveAnnoZ(active_joint_idx_, 1.0);
            }
            // -d
            if(e->key() == Qt::Key_1) {
                moveAnnoD(active_joint_idx_, -1.0);
            }
            // +d
            if(e->key() == Qt::Key_2) {
                moveAnnoD(active_joint_idx_, 1.0);
            }
        }
        else { //constrained edit mode - spherical movement
            if(active_joint_idx_ == 0) {
                if(e->key() == Qt::Key_2) {
                    rotateAnno(-0.1, 0, 0);
                }
                if(e->key() == Qt::Key_1) {
                    rotateAnno(0.1, 0, 0);
                }
                if(e->key() == Qt::Key_Left) {
                    rotateAnno(0, 0.1, 0);
                }
                if(e->key() == Qt::Key_Right) {
                    rotateAnno(0, -0.1, 0);
                }
                if(e->key() == Qt::Key_Down) {
                    rotateAnno(0, 0, 0.1);
                }
                if(e->key() == Qt::Key_Up) {
                    rotateAnno(0, 0, -0.1);
                }
            }
            else {
                cv::Vec3f a3 = projection_.to3D(anno_[active_joint_idx_]);
                //-r
                if(e->key() == Qt::Key_Minus) {
                    moveAnnoR(active_joint_idx_, -0.5);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
                //+r
                if(e->key() == Qt::Key_Plus) {
                    moveAnnoR(active_joint_idx_, +0.5);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
                //-theta
                if(e->key() == Qt::Key_Left) {
                    moveAnnoTheta(active_joint_idx_, -0.05);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
                //+theta
                if(e->key() == Qt::Key_Right) {
                    moveAnnoTheta(active_joint_idx_, +0.05);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
                //-theta
                if(e->key() == Qt::Key_Down) {
                    moveAnnoPhi(active_joint_idx_, -0.05);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
                //+theta
                if(e->key() == Qt::Key_Up) {
                    moveAnnoPhi(active_joint_idx_, +0.05);
                    cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                    moveAnnoAllDependent(active_joint_idx_, t);
                }
            }
        }
        
        //commands for both edit modes
        if(e->key() == Qt::Key_G && active_joint_idx_ > 0) {
            int parent_idx = parentJointIdx(active_joint_idx_);
            cv::Vec3f parent3 = projection_.to3D(anno_[parent_idx]);
            cv::Vec3f child = anno_[active_joint_idx_];
            
            float r = spherical_constraints_[active_joint_idx_ - 1];
            
            float x = child(0);
            float y = child(1);
            
            float a = parent3(0);
            float b = parent3(1);
            float c = parent3(2);
            
            float fx = projection_.K()(0, 0);
            float fy = projection_.K()(1, 1);
            float px = projection_.K()(0, 2);
            float py = projection_.K()(1, 2);
            
            //don't ask
            float root = -(a*a)*(fx*fx)*(fy*fy)-(b*b)*(fx*fx)*(fy*fy)-(a*a)*(fx*fx)*(py*py)-(b*b)*(fy*fy)*(px*px)-(c*c)*(fx*fx)*(py*py)-(c*c)*(fy*fy)*(px*px)+(fx*fx)*(fy*fy)*(r*r)-(a*a)*(fx*fx)*(y*y)-(b*b)*(fy*fy)*(x*x)-(c*c)*(fy*fy)*(x*x)-(c*c)*(fx*fx)*(y*y)+(fx*fx)*(py*py)*(r*r)+(fy*fy)*(px*px)*(r*r)+(fy*fy)*(r*r)*(x*x)+(fx*fx)*(r*r)*(y*y)+(a*a)*(fx*fx)*py*y*2.0+(b*b)*(fy*fy)*px*x*2.0+(c*c)*(fy*fy)*px*x*2.0+(c*c)*(fx*fx)*py*y*2.0-(fy*fy)*px*(r*r)*x*2.0-(fx*fx)*py*(r*r)*y*2.0+a*c*fx*(fy*fy)*px*2.0+b*c*(fx*fx)*fy*py*2.0-a*c*fx*(fy*fy)*x*2.0-b*c*(fx*fx)*fy*y*2.0+a*b*fx*fy*px*py*2.0-a*b*fx*fy*py*x*2.0-a*b*fx*fy*px*y*2.0+a*b*fx*fy*x*y*2.0;
            float z1 = (c*2.0+(a*(px-x)*2.0)/fx+(b*(py-y)*2.0)/fy)/(1.0/(fx*fx)*pow(px-x,2.0)*2.0+1.0/(fy*fy)*pow(py-y,2.0)*2.0+2.0);
            float z2 = (fx*fy*(sqrt(root)+c*fx*fy+a*fy*px+b*fx*py-a*fy*x-b*fx*y))/((fx*fx)*(fy*fy)+(fx*fx)*(py*py)+(fy*fy)*(px*px)+(fy*fy)*(x*x)+(fx*fx)*(y*y)-(fy*fy)*px*x*2.0-(fx*fx)*py*y*2.0);
            float z3 = -(fx*fy*(sqrt(root)-c*fx*fy-a*fy*px-b*fx*py+a*fy*x+b*fx*y))/((fx*fx)*(fy*fy)+(fx*fx)*(py*py)+(fy*fy)*(px*px)+(fy*fy)*(x*x)+(fx*fx)*(y*y)-(fy*fy)*px*x*2.0-(fx*fx)*py*y*2.0);

            if(project_constrained_depth_mode_ % 3 == 0 || root < 0) {
                anno_[active_joint_idx_](2) = z1;
            }
            else if(project_constrained_depth_mode_ % 3 == 1) {
                anno_[active_joint_idx_](2) = z2;
            }else if(project_constrained_depth_mode_ % 3 == 2) {
                anno_[active_joint_idx_](2) = z3;
            }
            
            project_constrained_depth_mode_++;
            changed_ = true;
        }
    }
    else if(modifiers == Qt::ControlModifier) { //move all points with ctrl
        // -x
        if(e->key() == Qt::Key_Left) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoX(joint_idx, 1.0);
            }
        }
        // +x
        if(e->key() == Qt::Key_Right) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoX(joint_idx, -1.0);
            }
        }
        // -y
        if(e->key() == Qt::Key_Down) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoY(joint_idx, -1.0);
            }
        }
        // +y
        if(e->key() == Qt::Key_Up) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoY(joint_idx, 1.0);
            }
        }
        // -z
        if(e->key() == Qt::Key_Plus) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoZ(joint_idx, -1.0);
            }
        }
        // +z
        if(e->key() == Qt::Key_Minus) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoZ(joint_idx, 1.0);
            }
        }
        // -d
        if(e->key() == Qt::Key_1) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoD(joint_idx, -1.0);
            }
        }
        // +d
        if(e->key() == Qt::Key_2) {
            for(int joint_idx = 0; joint_idx < anno_.size(); ++joint_idx) {
                moveAnnoD(joint_idx, 1.0);
            }
        }
    }
    
    
    
    
    
    
        
    
    
    //select next point
    if(e->key() == Qt::Key_Space) {
        if(modifiers == Qt::NoButton) {
            active_joint_idx_ = (active_joint_idx_ + 1) % anno_.size();
        }
        if(modifiers == Qt::ControlModifier) {
            active_joint_idx_ = (active_joint_idx_ - 1) < 0 ? anno_.size() - 1 : active_joint_idx_ - 1;
        }
    }
    
    //init anno with prev_joints
    if(e->key() == Qt::Key_I) {
        active_anno_proposal_ = (active_anno_proposal_ + 1) % anno_proposals_.size();
        anno_ = anno_proposals_[active_anno_proposal_];
    }
    
    //change pts rendered size in 3d
    if(e->key() == Qt::Key_V) {
        if(modifiers == Qt::NoButton) {
            size_3d_pts_ += 0.1;
        }
        if(modifiers == Qt::ControlModifier) {
            size_3d_annotation_ += 0.1;
        }
    }
    if(e->key() == Qt::Key_C) {
        if(modifiers == Qt::NoButton) {
            size_3d_pts_ -= 0.1;
        }
        if(modifiers == Qt::ControlModifier) {
            size_3d_annotation_ -= 0.1;
        }
    }
    
    //change in free-edit/expert mode
    if(e->key() == Qt::Key_X) {
        free_edit_mode_ = !free_edit_mode_;
        if(free_edit_mode_) {
            std::cout << "enabled free edit mode" << std::endl;
        }
        else {
            std::cout << "disabled free edit mode" << std::endl;
        }
    }
    
    //change view
    if(e->key() == Qt::Key_L) {
        draw_3d_ = !draw_3d_;
    }
    
    //next frame
    if(e->key() == Qt::Key_M && modifiers == Qt::NoButton) {
        return_direction_ = 1;
        this->close();
    }
    //previous frame
    if(e->key() == Qt::Key_N && modifiers == Qt::NoButton) {
        return_direction_ = -1;
        this->close();
    }
    
    //make a snap shot of the scene
    if(e->key() == Qt::Key_S && modifiers == Qt::ControlModifier) {
      saveSnapshot(false);
    }
    
    //in any case update gl - redraw
    updateGL();
}



void AnnotationViewer::mousePressEvent(QMouseEvent* e) {
    if(draw_3d_) { //in 3d use the normal navigation
        QGLViewer::mousePressEvent(e);
    }
    else { 
        
        if(e->button() == Qt::LeftButton) { //in 2d control the active point
            cv::Vec3f a3 = projection_.to3D(anno_[active_joint_idx_]);
                
            qreal x = e->localPos().x();
            qreal y = e->localPos().y();

            anno_[active_joint_idx_](0) = x / float(width()) * img_.cols;
            anno_[active_joint_idx_](1) = y / float(height()) * img_.rows;

            if(!free_edit_mode_) {
                if(active_joint_idx_ > 0) {
                    int parent_joint_idx = parentJointIdx(active_joint_idx_);

                    cv::Vec3f parent = projection_.to3D(anno_[parent_joint_idx]);
                    cv::Vec3f child = projection_.to3D(anno_[active_joint_idx_]);

                    cv::Vec3f child_spher = cartesian2spherical(parent, child);
                    child_spher(0) = spherical_constraints_[active_joint_idx_ - 1];
                    
                    child = spherical2cartesian(parent, child_spher);
                    anno_[active_joint_idx_] = projection_.to2D(child);
                }
                
                cv::Vec3f t = projection_.to3D(anno_[active_joint_idx_]) - a3;
                moveAnnoAllDependent(active_joint_idx_, t);
            }
            
            changed_ = true;
        }
        
    }
    
    //in any case update gl - redraw
    updateGL();
}

