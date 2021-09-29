#include "RMSprop.hpp"

RMSprop::RMSprop(const std::vector<std::shared_ptr<ILayer>>& layers, optimizerParams params)
    : learningRate_(params.learnRate), decay_(params.decay), layers_(layers)
    ,  epsilon_(params.epsilon), rho_(params.rho) {
    currentLearningRate_ = params.learnRate;
    iterations_ = 0;
    //Build and initialize cache for each layer
    for(uint32_t layer = 0; layer < layers_.size(); ++layer) {
        auto cache = std::make_shared<LayerCache>();
        cache->weight = Eigen::MatrixXd::Zero(
            layers_[layer]->getWeights().rows(), layers_[layer]->getWeights().cols());
        cache->bias = Eigen::MatrixXd::Zero(
            layers_[layer]->getBiases().rows(), layers_[layer]->getBiases().cols());
        caches_.push_back(cache);
    }
}

RMSprop::~RMSprop() {
}

void RMSprop::preUpdate() {
    if(decay_) {
        currentLearningRate_ =  learningRate_ * (1.0/(1 + decay_  * iterations_));
    }
}

void RMSprop::update(uint32_t layer) {
    Eigen::MatrixXd weightCache;
    Eigen::VectorXd biasCache;
    weightCache = rho_ * caches_[layer]->weight.array() + (1 - rho_) *
        layers_[layer]->getWeightsGradients().array().square();
    caches_[layer]->weight = weightCache;

    biasCache = rho_ * caches_[layer]->bias.array() + (1- rho_) *
        layers_[layer]->getBiasesGradients().array().square();
    caches_[layer]->bias = biasCache;

    layers_[layer]->getWeights() = layers_[layer]->getWeights().array() -currentLearningRate_ *
        layers_[layer]->getWeightsGradients().array() / (weightCache.array().sqrt() + epsilon_);
    layers_[layer]->getBiases() = layers_[layer]->getBiases().array() -currentLearningRate_ *
        layers_[layer]->getBiasesGradients().array() /  (biasCache.array().sqrt() + epsilon_);
}

void RMSprop::postUpdate() {
    iterations_++;
}
