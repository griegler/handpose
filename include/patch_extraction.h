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