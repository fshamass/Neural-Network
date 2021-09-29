#include "Neuron.hpp"

Neuron::Neuron(uint32_t numInputs, uint32_t numNeurons, uint32_t batchSize)
    : numInputs_(numInputs), numNeurons_(numNeurons) {
    input_.resize(numInputs, batchSize);
    weights_.resize(numNeurons, numInputs);
    output_.resize(numNeurons, batchSize);
    weightsGradients_.resize(numNeurons, numInputs);
    biasGradients_.resize(numNeurons);
    inputGradients_.resize(numInputs, batchSize);
}

Neuron::~Neuron() {
}

void Neuron::init() {
    for(uint32_t input =0; input < numInputs_; ++input) {
        weights_.col(input) = Eigen::VectorXd::Random(numNeurons_) +
            Eigen::VectorXd::Constant(numNeurons_,1) * 0.01;
    }
    bias_ = Eigen::VectorXd::Zero(numNeurons_);

    #ifdef __DEBUG__
    std::cout << "Neuron Weights: \n" << weights_ << std::endl;
    std::cout << "Neuron Weights size :" << weights_.rows() << " x " << weights_.cols() << std::endl;
    #endif
}

void Neuron::forward(Eigen::MatrixXd& input) {
    input_ = input;
    output_ = weights_ * input;
    //Broadcasting to add every column in Z to bias vector
    output_.colwise() += bias_;
}

void Neuron::backward(Eigen::MatrixXd& prevGradients) {
    weightsGradients_ = prevGradients * input_.transpose();
    inputGradients_   = weights_.transpose() * prevGradients;
    biasGradients_ = prevGradients.rowwise().sum();
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
