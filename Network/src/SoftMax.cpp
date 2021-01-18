#include "SoftMax.hpp"

SoftMax::SoftMax(uint32_t numNeurons, uint32_t batchSize) {
    output_.resize(batchSize, numNeurons);
    gradients_.resize(batchSize, numNeurons);
}

SoftMax::~SoftMax() {
}

void SoftMax::forward(Eigen::MatrixXd& input) {
    double sum;
    Eigen::MatrixXd inputCopy = input.array().exp();
    for(uint32_t row = 0; row < output_.rows(); ++row) {
        sum = 0;
        //Work with offset for smaller numbers
        sum = inputCopy.row(row).sum();
        output_.row(row) = inputCopy.row(row).unaryExpr([sum](double x) {return x/sum;});
    }
}

void SoftMax::backward(Eigen::MatrixXd& preGradients) {
    Eigen::MatrixXd jacobian_matrix;
    for(uint32_t row = 0; row < output_.rows(); ++row) {
        jacobian_matrix = output_.row(row).matrix().asDiagonal();
        jacobian_matrix =
            jacobian_matrix - output_.row(row).matrix().transpose() * (output_.row(row).matrix());
        gradients_.row(row) = jacobian_matrix * preGradients.row(row).matrix().transpose();
    }
}

Eigen::MatrixXd& SoftMax::getOutput() {
    return output_;
}

Eigen::MatrixXd& SoftMax::getGradients() {
    return gradients_;
}
