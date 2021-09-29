#include "DenseLayer.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "CategCrossEntropy.hpp"

DenseLayer::DenseLayer(DenseLayerParams& params, uint32_t numInputs, uint32_t batchSize) {
    numInputs_ = numInputs;
    type_ = LayerType::DENSE;
    neurons_ = std::make_shared<Neuron>(numInputs_, params.numNeurons, batchSize);
    neurons_->init();

    weightRegularizerL1_ = params.weightRegularizerL1;
    biasRegularizerL1_   = params.biasRegularizerL1;
    weightRegularizerL2_ = params.weightRegularizerL2;
    biasRegularizerL2_   = params.biasRegularizerL2;

    std::shared_ptr<IActivation> activ;
    switch(params.activation) {
        case Activation::RELU: {
            activ =
                std::make_shared<Relu>(params.numNeurons, batchSize);
        } break;
        case Activation::SOFTMAX: {
            activ =
                std::make_shared<SoftMax>(params.numNeurons, batchSize);
        } break;
        default : {
            //Throw an exception if activation function is not set
            throw "Invalid Activation function";
        } break;
    }
    activation_ = activ;

}

DenseLayer::~DenseLayer() {
    neurons_ = nullptr;
    activation_ = nullptr;
}

void DenseLayer::forward(Eigen::MatrixXd& input) {
    neurons_->forward(input);
    activation_->forward(neurons_->getOutput());
    calculateRegularizationLoss();
}

void DenseLayer::calculateRegularizationLoss() {
    regularizationLoss_ = 0.0;
    if(weightRegularizerL1_) {
        regularizationLoss_ += neurons_->getWeights().array().abs().sum() * weightRegularizerL1_;
    }
    if(biasRegularizerL1_) {
        regularizationLoss_ += neurons_->getBiases().array().abs().sum() * biasRegularizerL1_;
    }
    if(weightRegularizerL2_) {
        regularizationLoss_ += neurons_->getWeights().array().square().sum() * weightRegularizerL2_;
    }
    if(biasRegularizerL2_) {
        regularizationLoss_ += neurons_->getBiases().array().square().sum() * biasRegularizerL2_;
    }
}

void DenseLayer::backward(Eigen::MatrixXd& prevGradients) {
    activation_->backward(prevGradients);
    neurons_->backward(activation_->getGradients());
}

Eigen::MatrixXd& DenseLayer::getInputGradients() {
    return neurons_->getInputGradients();
}

Eigen::MatrixXd& DenseLayer::getActivOutput() {
    if(activation_) {
        return activation_->getOutput();
    } else {
        //Throw an exception if activation function is not set
        throw "Activation function of layer is not set!";
    }
}

uint32_t DenseLayer::getOutputSize() {
    return activation_->getOutput().rows();
}

Eigen::MatrixXd& DenseLayer::getWeights() {
    return neurons_->getWeights();
}

Eigen::MatrixXd& DenseLayer::getWeightsGradients() {
    return neurons_->getWeightsGradients();
}

Eigen::VectorXd& DenseLayer::getBiases() {
    return neurons_->getBiases();
}

Eigen::VectorXd& DenseLayer::getBiasesGradients() {
    return neurons_->getBiasesGradients();
}

double DenseLayer::getRegularizedLoss() {
    return regularizationLoss_;
}

LayerType DenseLayer::getLayerType() {
    return type_;
}
