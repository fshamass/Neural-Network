#ifndef __NEURAL_NET_IMPL__
#define __NEURAL_NET_IMPL__

#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include "NeuralNet.hpp"
#include "DenseLayer.hpp"
#include "DataHandler.hpp"
#include "INetLoss.hpp"
#include "IOptimizer.hpp"
#include "SGD.hpp"
#include "RMSprop.hpp"
#include "Adam.hpp"
#include "MatplotlibHelper.hpp"
#include "NeuralNetDefines.hpp"

class NeuralNetImpl : public NeuralNet {
public:
    //Number of Features and Batch Size
    NeuralNetImpl(uint32_t numFeatures, uint32_t batchSize);
    ~NeuralNetImpl();
    void setTrainData(std::vector<std::shared_ptr<DataHandler::Data>>& trainData) override;
    void setTestData(std::vector<std::shared_ptr<DataHandler::Data>>& testData) override;
    void setValidData(std::vector<std::shared_ptr<DataHandler::Data>>& validData) override;
    //New layer will always go to bottom of list
    //Number of Neurons and Activation Function
    void addLayer(uint32_t numNeurons, Activation activation,
        double weightRegularizerL1=0, double biasRegularizerL1=0,
        double weightRegularizerL2=0, double biasRegularizerL2=0) override;
    void setLossFunction(LossFunc lossFunc) override;
    void setOptimizer(Optimizer optimizer, optimizerParams params) override;
    void train(uint32_t epochs) override;
    std::vector<double>& getTrainLoss() override;
    std::vector<double>& getValidLoss() override;
    std::vector<double>& getTrainAcc()  override;
    std::vector<double>& getValidAcc()  override;
private:
    uint32_t numFeatures_;
    uint32_t batchSize_;
    std::vector<std::shared_ptr<DataHandler::Data>> trainData_;
    std::vector<std::shared_ptr<DataHandler::Data>> validData_;
    std::vector<std::shared_ptr<DataHandler::Data>> testData_;
    std::vector<double> trainLoss_;
    std::vector<double> validLoss_;
    std::vector<double> trainAcc_;
    std::vector<double> validAcc_;
    std::shared_ptr<INetLoss> netLoss_;
    std::shared_ptr<IOptimizer> optimizer_;
    double accuracy_;
    std::vector<std::shared_ptr<DenseLayer>> network_;

    void shuffleData(std::vector<std::shared_ptr<DataHandler::Data>>& data);
    bool isNetworkValid();
    void populateInputsAndTargets(std::vector<std::shared_ptr<DataHandler::Data>>& data,
        uint32_t startIdx, uint32_t endIdx, Eigen::MatrixXd& netInput, Eigen::VectorXd& targets);
    void forwardPass(Eigen::MatrixXd& input);
    void calculateNetLoss(Eigen::VectorXd& targets);
    void addRegularizationLoss();
    void backPropagate(Eigen::VectorXd& targets);
    void optimize();
};
#endif