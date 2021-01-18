#include "SGD.hpp"

SGD::SGD(const std::vector<std::shared_ptr<DenseLayer>>& layers, optimizerParams params)
    : learningRate_(params.learnRate), decay_(params.decay)
    , momentum_(params.momentum), layers_(layers) {
    currentLearningRate_ = params.learnRate;
    iterations_ = 0;
    for(uint32_t layer = 0; layer < layers_.size(); ++layer) {
        auto momentum = std::make_shared<LayerMomentums>();
        momentum->weight = Eigen::MatrixXd::Zero(
            layers_[layer]->getWeights().rows(), layers_[layer]->getWeights().cols());
        momentum->bias = Eigen::MatrixXd::Zero(
            layers_[layer]->getBiases().rows(), layers_[layer]->getBiases().cols());
        momentums_.push_back(momentum);
    }
}

SGD::~SGD() {
}

void SGD::preUpdate() {
    if(decay_) {
        currentLearningRate_ =  learningRate_ * (1.0/(1 + decay_  * iterations_));
    }
}

void SGD::update(uint32_t layer) {
    Eigen::MatrixXd weightUpdates;
    Eigen::VectorXd biasUpdates;
    if(momentum_) {
        weightUpdates = momentum_ * momentums_[layer]->weight -
                        currentLearningRate_ * layers_[layer]->getWeightsGradients();
        momentums_[layer]->weight = weightUpdates;

        biasUpdates = momentum_ * momentums_[layer]->bias -
                      currentLearningRate_ * layers_[layer]->getBiasesGradients();
        momentums_[layer]->bias = biasUpdates;
    } else {
        weightUpdates = - currentLearningRate_ * layers_[layer]->getWeightsGradients();
        biasUpdates = - currentLearningRate_ * layers_[layer]->getBiasesGradients();
    }
    layers_[layer]->getWeights() += weightUpdates;
    layers_[layer]->getBiases() += biasUpdates;
}

void SGD::postUpdate() {
    iterations_++;
}
