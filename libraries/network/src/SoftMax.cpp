#include "SoftMax.hpp"

SoftMax::SoftMax(uint32_t numNeurons, uint32_t batchSize) {
    output_.resize(numNeurons, batchSize);
    gradients_.resize(numNeurons, batchSize);
}

SoftMax::~SoftMax() {
}

void SoftMax::forward(Eigen::MatrixXd& input) {
    double sum;
    Eigen::MatrixXd inputCopy = input.array().exp();
    for(uint32_t col = 0; col < output_.cols(); ++col) {
        sum = 0;
        //Work with offset for smaller numbers
        sum = inputCopy.col(col).sum();
        output_.col(col) = inputCopy.col(col).unaryExpr([sum](double x) {return x/sum;});
    }
}

void SoftMax::backward(Eigen::MatrixXd& preGradients) {
    Eigen::MatrixXd jacobian_matrix;
    for(uint32_t col = 0; col < output_.cols(); ++col) {
        jacobian_matrix = output_.col(col).matrix().asDiagonal();
        jacobian_matrix =
            jacobian_matrix - (output_.col(col).matrix() * output_.col(col).matrix().transpose());
        gradients_.col(col) = jacobian_matrix * preGradients.col(col).matrix();
    }
}

Eigen::MatrixXd& SoftMax::getOutput() {
    return output_;
}

Eigen::MatrixXd& SoftMax::getGradients() {
    return gradients_;
}
