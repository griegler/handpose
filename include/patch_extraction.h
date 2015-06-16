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

#ifndef EXTRACT_PATCH_H
#define EXTRACT_PATCH_H

#include "hand_patch.h"
#include "hand_segmentation_result.h"

template <typename Dtype>
class PatchExtraction {
public:
    PatchExtraction(const std::vector<int>& patch_widths, Dtype bg_value, bool normalize_depth) : 
        patch_widths_(patch_widths), bg_value_(bg_value), normalize_depth_(normalize_depth) {}
    virtual ~PatchExtraction() {}
    
    std::vector<HandPatch<Dtype> > operator()(const cv::Mat_<Dtype>& depth, const HandSegmentationResult& segmentation) const;
    
    bool isReasonableHandPatch(const std::vector<HandPatch<Dtype> >& patches) const;
    
    const std::vector<int> patchWidths() const { return patch_widths_; }
    bool normalizeDepth() const { return normalize_depth_; }
    
private:
    const std::vector<int> patch_widths_;
    Dtype bg_value_;
    const bool normalize_depth_;
};

#endif
