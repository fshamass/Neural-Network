#ifndef __DENSELAYER__
#define __DENSELAYER__
#include <iostream>
#include <memory>

#include "Neuron.hpp"
#include "IActivation.hpp"
#include "ILayer.hpp"
#include "NeuralNetDefines.hpp"

class DenseLayer : public ILayer {
public:
    //Constructor takes params needed to construct dense layer
    DenseLayer(DenseLayerParams& params, uint32_t numInputs, uint32_t batchSize);
    ~DenseLayer();
    void forward(Eigen::MatrixXd& input) override;
    void backward(Eigen::MatrixXd& prevGradients) override;
    Eigen::MatrixXd& getActivOutput() override;
    Eigen::MatrixXd& getInputGradients() override;
    Eigen::MatrixXd& getWeights() override;
    Eigen::MatrixXd& getWeightsGradients() override;
    Eigen::VectorXd& getBiases() override;
    Eigen::VectorXd& getBiasesGradients() override;
    uint32_t getOutputSize() override;
    double getRegularizedLoss() override;
    LayerType getLayerType() override;
private:
    uint32_t numInputs_;
    std::shared_ptr<IActivation> activation_;
    std::shared_ptr<Neuron> neurons_;
    double  weightRegularizerL1_;
    double  biasRegularizerL1_;
    double  weightRegularizerL2_;
    double  biasRegularizerL2_;
    double  regularizationLoss_;
    void    calculateRegularizationLoss();
    LayerType type_;
};
#endif