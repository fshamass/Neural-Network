#ifndef __CATEGCROSSENTROPY__
#define __CATEGCROSSENTROPY__

#include <iostream>
#include <cmath>
#include "INetLoss.hpp"

class CategCrossEntropy: public INetLoss {
public:
    CategCrossEntropy(uint32_t numClasses, uint32_t batchSize);
    ~CategCrossEntropy();
    //Passth the output from activation of last layer
    void forward(Eigen::MatrixXd& predictedOutput, Eigen::VectorXd& targetOutput) override;
    //Calculate gradients from loss function
    void backward(Eigen::MatrixXd& lossOutput, Eigen::VectorXd& targetOutput) override;
    //Add L1/L2 Regularization if needed
    void addRegularization(double regLoss) override;
    //Get average network loss
    double getAveNetLoss() override;
    //Get network accuracy
    double getNetAccuracy() override;
    //Get Loss Gradients
    Eigen::MatrixXd& getLossGradients() override;
    Eigen::VectorXd& getOutput() override;
private:
    Eigen::VectorXd output_;
    Eigen::MatrixXd gradients_;
    double netLoss_;
    double netAccuracy_;
};
#endif