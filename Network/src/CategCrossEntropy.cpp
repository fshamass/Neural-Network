#include "CategCrossEntropy.hpp"

CategCrossEntropy::CategCrossEntropy(uint32_t numClasses, uint32_t batchSize) {
    output_.resize(batchSize);
    gradients_.resize(batchSize, numClasses);
    netLoss_ = 0;
    netAccuracy_ = 0;
}

CategCrossEntropy::~CategCrossEntropy() {
}

void CategCrossEntropy::forward(Eigen::MatrixXd& predictedOutput, Eigen::VectorXd& targetOutput) {
    double netAccuracy = 0;
    double netLoss = 0;
    for(uint32_t sample = 0; sample < predictedOutput.rows(); ++sample) {
        uint32_t pos;
        predictedOutput.row(sample).maxCoeff(&pos);
        if(pos == targetOutput(sample)) {
            netAccuracy++;
        }
        //Clip to protect against 0 or 1 values and get loss of predicted output
        output_(sample) = (predictedOutput(sample, targetOutput[sample]) <= 0) ?
            -log(1e-7) : (predictedOutput(sample, targetOutput[sample]) >= 1) ?
            -log(1-1e-7) : -log(predictedOutput(sample, targetOutput[sample]));
        netLoss += output_(sample);
    }
    netAccuracy_ = netAccuracy/predictedOutput.rows();
    netLoss_ = netLoss/predictedOutput.rows();
}

void CategCrossEntropy::backward(Eigen::MatrixXd& lossOutput, Eigen::VectorXd& targetOutput) {
    gradients_ = Eigen::MatrixXd::Zero(gradients_.rows(), gradients_.cols());
    for(uint32_t row = 0; row < lossOutput.rows(); ++row) {
        gradients_(row, targetOutput(row)) =
            (-1/lossOutput(row, targetOutput(row)))/lossOutput.rows();
    }
}

void CategCrossEntropy::addRegularization(double regLoss){
    netLoss_ += regLoss;
}

double CategCrossEntropy::getAveNetLoss() {
    return netLoss_;
}

double CategCrossEntropy::getNetAccuracy() {
    return netAccuracy_;
}

Eigen::MatrixXd& CategCrossEntropy::getLossGradients() {
    return gradients_;
}

Eigen::VectorXd& CategCrossEntropy::getOutput() {
    return output_;
}
