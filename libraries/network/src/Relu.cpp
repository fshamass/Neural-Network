#include "Relu.hpp"

Relu::Relu(uint32_t numNeurons, uint32_t batchSize) {
    output_.resize(numNeurons, batchSize);
    gradients_.resize(numNeurons, batchSize);
}

Relu::~Relu() {
}

void Relu::forward(Eigen::MatrixXd& input) {
    output_ = input.unaryExpr([](double x) {return ((x < 0) ? 0 : x);});
}

void Relu::backward(Eigen::MatrixXd& preGradients) {
    gradients_ = Eigen::MatrixXd::Zero(preGradients.rows(), preGradients.cols());
    gradients_ = output_.binaryExpr(preGradients, [](double x, double y) {return ((x==0)?x:y);});
}

Eigen::MatrixXd& Relu::getOutput() {
    return output_;
}

Eigen::MatrixXd& Relu::getGradients() {
    return gradients_;
}
