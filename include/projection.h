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
