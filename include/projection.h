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

#ifndef PROJECTION_H
#define PROJECTION_H

#include <opencv2/core/core.hpp>

namespace cv {
  typedef Matx<double, 1, 8> Matx18d;
}


class Projection {
public:
    Projection(const cv::Matx33d& K, const cv::Matx18d d) : K_(K), d_(d) {};
    virtual ~Projection() {}
    
    virtual cv::Vec3f to2D(const cv::Vec3f& v3) const = 0;
    virtual cv::Vec3f to3D(const cv::Vec3f& v2) const = 0;
    
    virtual void to2D(const std::vector<cv::Vec3f>& v3, std::vector<cv::Vec3f>& v2) const;
    virtual void to3D(const std::vector<cv::Vec3f>& v2, std::vector<cv::Vec3f>& v3) const;
    
    virtual cv::Matx33d K() const { return K_; }
    virtual cv::Matx18d d() const { return d_; }
    
protected:
    const cv::Matx33d K_;
    const cv::Matx18d d_;
};


class OrthographicProjection : public Projection {
public:
    OrthographicProjection(const cv::Matx33d& K, const cv::Matx18d d) : 
      Projection(K, d) {};
    virtual ~OrthographicProjection() {}
    
    virtual cv::Vec3f to2D(const cv::Vec3f& v3) const;
    virtual cv::Vec3f to3D(const cv::Vec3f& v2) const;
};


class ProjectiveProjection : public Projection {
public:
    ProjectiveProjection(const cv::Matx33d& K, const cv::Matx18d d) : 
      Projection(K, d) {};
    virtual ~ProjectiveProjection() {}
    
    virtual cv::Vec3f to2D(const cv::Vec3f& v3) const;
    virtual cv::Vec3f to3D(const cv::Vec3f& v2) const;
};

#endif
