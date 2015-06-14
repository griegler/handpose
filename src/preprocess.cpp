#include "preprocess.h"

template <typename Dtype>
void Preprocess<Dtype>::operator()(std::vector<HandPatch<Dtype> >& hand_patch) const {
    if(subtract_mean_) {
        hand_patch[0].patch_ -= mean_;
    }
}


template class Preprocess<float>;
template class Preprocess<double>;