#include "Neuron.hpp"

Neuron::Neuron(uint32_t numInputs, uint32_t numNeurons, uint32_t batchSize)
    : numInputs_(numInputs), numNeurons_(numNeurons) {
    input_.resize(batchSize, numInputs);
    weights_.resize(numInputs, numNeurons);
    output_.resize(batchSize, numNeurons);
    weightsGradients_.resize(numInputs, numNeurons);
    biasGradients_.resize(numNeurons);
    inputGradients_.resize(batchSize, numInputs);
}

Neuron::~Neuron() {
}

void Neuron::init() {
    for(uint32_t neuron =0; neuron < numNeurons_; ++neuron) {
        weights_.col(neuron) = Eigen::VectorXd::Random(numInputs_) +
            Eigen::VectorXd::Constant(numInputs_,1) * 0.01;
    }
    bias_ = Eigen::VectorXd::Zero(numNeurons_);
    #ifdef __DEBUG__
    std::cout << "Neuron Weights: \n" << weights_ << std::endl;
    #endif
}

void Neuron::forward(Eigen::MatrixXd& input) {
    input_ = input;
    output_ = input * weights_ + bias_.transpose().replicate(input.rows(),1);
}

void Neuron::backward(Eigen::MatrixXd& prevGradients) {
    weightsGradients_ = input_.transpose() * prevGradients;
    inputGradients_ = prevGradients * weights_.transpose();
    biasGradients_ = prevGradients.colwise().sum();
}

Eigen::MatrixXd& Neuron::getOutput() {
    return output_;
}

Eigen::MatrixXd& Neuron::getInputGradients() {
    return inputGradients_;
}

Eigen::MatrixXd& Neuron::getWeights() {
    return weights_;
}

Eigen::MatrixXd& Neuron::getWeightsGradients() {
    return weightsGradients_;
}

Eigen::VectorXd& Neuron::getBiases() {
    return bias_;
}

Eigen::VectorXd& Neuron::getBiasesGradients() {
    return biasGradients_;
}
