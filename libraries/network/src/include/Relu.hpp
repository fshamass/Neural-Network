#ifndef __RELU__
#define __RELU__

#include <iostream>
#include "IActivation.hpp"

class Relu : public IActivation {
public:
    Relu(uint32_t numNeurons, uint32_t batchSize);
    ~Relu();
    void forward(Eigen::MatrixXd& input) override;
    void backward(Eigen::MatrixXd& preGradients) override;
    Eigen::MatrixXd& getOutput() override;
    Eigen::MatrixXd& getGradients() override;
private:
    Eigen::MatrixXd output_;
    Eigen::MatrixXd gradients_;
};
#endif