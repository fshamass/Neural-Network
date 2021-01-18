#ifndef __SOFTMAX__
#define __SOFTMAX__

#include <iostream>
#include "IActivation.hpp"

class SoftMax : public IActivation {
public:
    //Number of neurons (equals num of classes) and batch size
    SoftMax(uint32_t numNeurons, uint32_t batchSize);
    ~SoftMax();
    void forward(Eigen::MatrixXd& input) override;
    void backward(Eigen::MatrixXd& preGradients) override;
    Eigen::MatrixXd& getOutput() override;
    Eigen::MatrixXd& getGradients() override;
private:
    Eigen::MatrixXd output_;
    Eigen::MatrixXd gradients_;
};
#endif