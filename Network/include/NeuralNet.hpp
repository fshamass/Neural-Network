#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include "DenseLayer.hpp"
#include "Data.hpp"
#include "DataHandler.hpp"
#include "INetLoss.hpp"
#include "IOptimizer.hpp"
#include "SGD.hpp"
#include "RMSprop.hpp"
#include "Adam.hpp"
#include "MatplotlibHelper.hpp"
#include "NeuralNetDefines.hpp"

class NeuralNet {
public:
    //Number of Features and Batch Size
    NeuralNet(uint32_t numFeatures, uint32_t batchSize);
    ~NeuralNet();
    void setTrainData(std::vector<std::shared_ptr<Data>>& trainData);
    void setTestData(std::vector<std::shared_ptr<Data>>& testData);
    void setValidData(std::vector<std::shared_ptr<Data>>& validData);
    //New layer will always go to bottom of list
    //Number of Neurons and Activation Function
    void addLayer(uint32_t numNeurons, Activation activation,
        double weightRegularizerL1=0, double biasRegularizerL1=0,
        double weightRegularizerL2=0, double biasRegularizerL2=0);
    void setLossFunction(LossFunc lossFunc);
    void setOptimizer(Optimizer optimizer, optimizerParams params);
    void train(uint32_t epochs);
private:
    uint32_t numFeatures_;
    uint32_t batchSize_;
    std::vector<std::shared_ptr<Data>> trainData_;
    std::vector<std::shared_ptr<Data>> validData_;
    std::vector<std::shared_ptr<Data>> testData_;
    std::shared_ptr<INetLoss> netLoss_;
    std::shared_ptr<IOptimizer> optimizer_;
    double accuracy_;
    std::vector<std::shared_ptr<DenseLayer>> network_;

    void shuffleData(std::vector<std::shared_ptr<Data>>& data);
    bool isNetworkValid();
    void populateInputsAndTargets(std::vector<std::shared_ptr<Data>>& data,
        uint32_t startIdx, uint32_t endIdx, Eigen::MatrixXd& netInput, Eigen::VectorXd& targets);
    void forwardPass(Eigen::MatrixXd& input);
    void calculateNetLoss(Eigen::VectorXd& targets);
    void addRegularizationLoss();
    void backPropagate(Eigen::VectorXd& targets);
    void optimize();
};
#endif