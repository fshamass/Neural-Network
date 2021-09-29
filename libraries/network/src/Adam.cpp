#include "Adam.hpp"

Adam::Adam(const std::vector<std::shared_ptr<DenseLayer>>& layers, optimizerParams params)
    : learningRate_(params.learnRate), decay_(params.decay), layers_(layers)
    ,  epsilon_(params.epsilon), beta1_(params.beta1), beta2_(params.beta2) {
    currentLearningRate_ = params.learnRate;
    iterations_ = 0;
    //Build and initialize and caches momentums for each layer
    for(uint32_t layer = 0; layer < layers_.size(); ++layer) {
        auto cache = std::make_shared<LayerCache>();
        cache->weight = Eigen::MatrixXd::Zero(
            layers_[layer]->getWeights().rows(), layers_[layer]->getWeights().cols());
        cache->bias = Eigen::MatrixXd::Zero(
            layers_[layer]->getBiases().rows(), layers_[layer]->getBiases().cols());
        caches_.push_back(cache);

        auto momentum = std::make_shared<LayerMomentums>();
        momentum->weight = Eigen::MatrixXd::Zero(
            layers_[layer]->getWeights().rows(), layers_[layer]->getWeights().cols());
        momentum->bias = Eigen::MatrixXd::Zero(
            layers_[layer]->getBiases().rows(), layers_[layer]->getBiases().cols());
        momentums_.push_back(momentum);
    }
}

Adam::~Adam() {
}

void Adam::preUpdate() {
    if(decay_) {
        currentLearningRate_ =  learningRate_ * (1.0/(1 + decay_  * iterations_));
    }
}

void Adam::update(uint32_t layer) {
    Eigen::MatrixXd weightCacheCorrected;
    Eigen::VectorXd biasCacheCorrected;
    Eigen::MatrixXd weightMomentumCorrected;
    Eigen::VectorXd biasMomentumCorrected;

    momentums_[layer]->weight = beta1_ * momentums_[layer]->weight.array() +
        (1 - beta1_) * layers_[layer]->getWeightsGradients().array();
    momentums_[layer]->bias = beta1_ * momentums_[layer]->bias.array() +
        (1 - beta1_) * layers_[layer]->getBiasesGradients().array();

    weightMomentumCorrected = momentums_[layer]->weight.array()/(1 - pow(beta1_, (iterations_+ 1)));
    biasMomentumCorrected = momentums_[layer]->bias.array()/(1 - pow(beta1_, (iterations_+ 1)));

    caches_[layer]->weight = beta2_ * caches_[layer]->weight.array() +
        (1 - beta2_) * layers_[layer]->getWeightsGradients().array().square();
    caches_[layer]->bias = beta2_ * caches_[layer]->bias.array() +
        (1 - beta2_) * layers_[layer]->getBiasesGradients().array().square();

    weightCacheCorrected = caches_[layer]->weight.array()/(1 - pow(beta2_, ((iterations_+ 1))));
    biasCacheCorrected = caches_[layer]->bias.array()/(1 - pow(beta2_, ((iterations_+ 1))));

    layers_[layer]->getWeights() = layers_[layer]->getWeights().array() - currentLearningRate_ *
        weightMomentumCorrected.array()/(weightCacheCorrected.array().sqrt() + epsilon_);
    layers_[layer]->getBiases() = layers_[layer]->getBiases().array() - currentLearningRate_ *
        biasMomentumCorrected.array()/(biasCacheCorrected.array().sqrt() + epsilon_);
}

void Adam::postUpdate() {
    iterations_++;
}
