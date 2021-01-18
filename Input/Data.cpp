#include "Data.hpp"

Data::Data(uint32_t featuresSize) : featuresSize_(featuresSize) {
    //Resize feature based on features size
    features_.resize(featuresSize_);
}

Data::~Data() {
    features_.clear();
}

uint8_t Data::getLabel() {
    return label_;
}

uint32_t Data::getFeatureSize() {
    return features_.size();
}

std::vector<double> Data::getFeatures() {
    return features_;
}
