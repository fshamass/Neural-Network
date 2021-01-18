#ifndef __DENSELAYER__
#define __DENSELAYER__
#include <iostream>
#include <memory>

#include "Neuron.hpp"
#include "IActivation.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "CategCrossEntropy.hpp"
#include "NeuralNetDefines.hpp"


class DenseLayer {
public:
    //Constructor takes params needed to construct dense layer
    DenseLayer(DenseLayerParams& denseLayerParams);
    ~DenseLayer();
    void forward(Eigen::MatrixXd& input);
    void backward(Eigen::MatrixXd& prevGradients);
    Eigen::MatrixXd& getActivOutput();
    Eigen::MatrixXd& getInputGradients();
    Eigen::MatrixXd& getWeights();
    Eigen::MatrixXd& getWeightsGradients();
    Eigen::VectorXd& getBiases();
    Eigen::VectorXd& getBiasesGradients();
    uint32_t getOutputSize();
    double getRegularizationLoss();

private:
    uint32_t numInputs_;
    std::shared_ptr<IActivation> activation_;
    std::shared_ptr<Neuron> neurons_;
    double    weightRegularizerL1_;
    double    biasRegularizerL1_;
    double    weightRegularizerL2_;
    double    biasRegularizerL2_;
    double    regularizationLoss_;

    void      calculateRegularizationLoss();
};
#endif