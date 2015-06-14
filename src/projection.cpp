#include "projection.h"


void Projection::to2D(const std::vector< cv::Vec3f >& v3, std::vector< cv::Vec3f >& v2) const {
    v2.resize(v3.size());
    
    for(size_t pts_idx = 0; pts_idx < v3.size(); ++pts_idx) {
        v2[pts_idx] = to2D(v3[pts_idx]);
    }
}

void Projection::to3D(const std::vector< cv::Vec3f >& v2, std::vector< cv::Vec3f >& v3) const {
    v3.resize(v2.size());
    
    for(size_t pts_idx = 0; pts_idx < v2.size(); ++pts_idx) {
        v3[pts_idx] = to3D(v2[pts_idx]);
    }
}


//------------------------------------------------------------------------------
cv::Vec3f OrthographicProjection::to2D(const cv::Vec3f& v3) const {
    cv::Vec3f v2;
        
    v2[0] = K_(0, 0) * v3(0) / v3(2) + K_(0, 2);
    v2[1] = K_(1, 1) * v3(1) / v3(2) + K_(1, 2);
    v2[2] = v3(2);
       
    return v2;
}

cv::Vec3f OrthographicProjection::to3D(const cv::Vec3f& v2) const {
    cv::Vec3f v3;
    
    v3[0] = v2(2) * (v2(0) - K_(0, 2)) / K_(0, 0);
    v3[1] = v2(2) * (v2(1) - K_(1, 2)) / K_(1, 1);
    v3[2] = v2(2);
    
    return v3;
}


//------------------------------------------------------------------------------
cv::Vec3f ProjectiveProjection::to2D(const cv::Vec3f& v3) const {
    //TODO
    cv::Vec3f v2;
        
    v2[0] = K_(0, 0) * v3(0) / v3(2) + K_(0, 2);
    v2[1] = K_(1, 1) * v3(1) / v3(2) + K_(1, 2);
    v2[2] = v3(2);
       
    return v2;
}

cv::Vec3f ProjectiveProjection::to3D(const cv::Vec3f& v2) const {
    //TODO
    cv::Vec3f v3;
    
    v3[0] = v2(2) * (v2(0) - K_(0, 2)) / K_(0, 0);
    v3[1] = v2(2) * (v2(1) - K_(1, 2)) / K_(1, 1);
    v3[2] = v2(2);
    
    return v3;
}
