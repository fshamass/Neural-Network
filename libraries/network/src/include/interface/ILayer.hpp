#ifndef __ILAYER_HPP__
#define __ILAYER_HPP__

#include <Eigen/Dense>
#include "NeuralNetDefines.hpp"

class ILayer {
public:
    virtual void forward(Eigen::MatrixXd& input) = 0;
    virtual void backward(Eigen::MatrixXd& prevGradients) = 0;
    virtual Eigen::MatrixXd& getActivOutput() = 0;
    virtual Eigen::MatrixXd& getInputGradients() = 0;
    virtual Eigen::MatrixXd& getWeights() = 0;
    virtual Eigen::MatrixXd& getWeightsGradients() = 0;
    virtual Eigen::VectorXd& getBiases() = 0;
    virtual Eigen::VectorXd& getBiasesGradients() = 0;
    virtual uint32_t getOutputSize() = 0;
    virtual double getRegularizedLoss() = 0;
    virtual LayerType getLayerType() = 0;
};
#endif