#ifndef __DROPOUT_LAYER_HPP__
#define __DROPOUT_LAYER_HPP__

#include <random>
#include "ILayer.hpp"

class DropoutLayer: public ILayer {
public:
    DropoutLayer(uint32_t numNeurons, uint32_t batchSize, double dropoutRate);
    ~DropoutLayer();
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
    uint32_t numNeurons_;
    uint32_t batchSize_;
    double successRate_;
    Eigen::MatrixXd output_;
    Eigen::MatrixXd inputGradients_;
    Eigen::MatrixXd gradientsMask_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd bias_;
    Eigen::MatrixXd weightsGradients_;
    Eigen::VectorXd biasGradients_;
    std::default_random_engine generator_;
    std::bernoulli_distribution distribution_;
    LayerType type_;
};
#endif