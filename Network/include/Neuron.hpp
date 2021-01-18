#ifndef __NEURON__
#define __NEURON__

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include "Relu.hpp"

class Neuron {
public:
    //Constructor takes Num of inputs, num of neurons, and batch size
    Neuron(uint32_t numInputs, uint32_t numNeurons, uint32_t batchSize);
    ~Neuron();
    //Method to randomly initialize weights
    void init();
    //Method to do linear algebra operations
    void forward(Eigen::MatrixXd& input);
    void backward(Eigen::MatrixXd& prevGradients);
    Eigen::MatrixXd& getOutput();
    Eigen::MatrixXd& getInputGradients();
    Eigen::MatrixXd& getWeights();
    Eigen::MatrixXd& getWeightsGradients();
    Eigen::VectorXd& getBiases();
    Eigen::VectorXd& getBiasesGradients();
private:
    Eigen::MatrixXd input_;
    Eigen::MatrixXd weights_;
    Eigen::VectorXd bias_;
    Eigen::MatrixXd output_;
    Eigen::MatrixXd weightsGradients_;
    Eigen::VectorXd biasGradients_;
    Eigen::MatrixXd inputGradients_;
    uint32_t numInputs_;
    uint32_t numNeurons_;
};
#endif