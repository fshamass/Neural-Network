#include "DropoutLayer.hpp"

DropoutLayer::DropoutLayer(uint32_t numNeurons, uint32_t batchSize, double dropoutRate)
    : numNeurons_(numNeurons), batchSize_(batchSize) {
    successRate_ = 1 - dropoutRate;
    type_ = LayerType::DROPOUT;

    distribution_ = std::bernoulli_distribution(successRate_);
    output_.resize(batchSize, numNeurons);
    gradientsMask_.resize(batchSize, numNeurons);
    inputGradients_.resize(batchSize, numNeurons);
}

DropoutLayer::~DropoutLayer() {
}

void DropoutLayer::forward(Eigen::MatrixXd& input) {
    Eigen::VectorXd dist(numNeurons_);
    //Generate distribution for all neurons
    for(int i =0;i<numNeurons_;++i) {
        dist(i) = (distribution_(generator_))?1/successRate_:0;
    }
    gradientsMask_ = dist.rowwise().replicate(batchSize_).transpose();
    output_ = gradientsMask_.array() * input.array();
}

void DropoutLayer::backward(Eigen::MatrixXd& prevGradients) {
    inputGradients_ = prevGradients.array() * prevGradients.array();
}

Eigen::MatrixXd& DropoutLayer::getActivOutput() {
    return output_;
}

Eigen::MatrixXd& DropoutLayer::getInputGradients() {
    return inputGradients_;
}

Eigen::MatrixXd& DropoutLayer::getWeights() {
    return weights_;
}

Eigen::MatrixXd& DropoutLayer::getWeightsGradients() {
    return weightsGradients_;
}

Eigen::VectorXd& DropoutLayer::getBiases() {
    return bias_;
}

Eigen::VectorXd& DropoutLayer::getBiasesGradients() {
    return biasGradients_;
}

uint32_t DropoutLayer::getOutputSize() {
    return numNeurons_;
}

double DropoutLayer::getRegularizedLoss() {
    return 0.0;
}

LayerType DropoutLayer::getLayerType() {
    return type_;
}
