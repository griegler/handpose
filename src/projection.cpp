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
