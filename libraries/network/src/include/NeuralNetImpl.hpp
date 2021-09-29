#ifndef __NEURAL_NET_IMPL__
#define __NEURAL_NET_IMPL__

#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include "NeuralNet.hpp"
#include "DenseLayer.hpp"
#include "DropoutLayer.hpp"
#include "DataHandler.hpp"
#include "INetLoss.hpp"
#include "IOptimizer.hpp"
#include "SGD.hpp"
#include "RMSprop.hpp"
#include "Adam.hpp"
#include "NeuralNetDefines.hpp"

class NeuralNetImpl : public NeuralNet {
public:
    //Number of Features and Batch Size
    NeuralNetImpl(uint32_t numFeatures, uint32_t batchSize);
    ~NeuralNetImpl();
    void setTrainData(DataHandler::dataPtrVector& trainData) override;
    void setTestData(DataHandler::dataPtrVector& testData) override;
    void setValidData(DataHandler::dataPtrVector& validData) override;
    //New layer will always go to bottom of list
    //Number of Neurons and Activation Function
    void addDenseLayer(DenseLayerParams params) override;
    void addDropoutLayer( double dropoutRate) override;
    void setLossFunction(LossFunc lossFunc) override;
    void setOptimizer(Optimizer optimizer, optimizerParams params) override;
    void fit(uint32_t epochs) override;
    void evaluate() override;

    std::vector<double>& getTrainLoss() override;
    std::vector<double>& getValidLoss() override;
    std::vector<double>& getTrainAcc()  override;
    std::vector<double>& getValidAcc()  override;
private:
    uint32_t numFeatures_;
    uint32_t batchSize_;
    double accuracy_;
    DataHandler::dataPtrVector trainData_;
    DataHandler::dataPtrVector validData_;
    DataHandler::dataPtrVector testData_;
    std::vector<double> trainLoss_;
    std::vector<double> validLoss_;
    std::vector<double> trainAcc_;
    std::vector<double> validAcc_;
    std::shared_ptr<INetLoss> netLoss_;
    std::shared_ptr<IOptimizer> optimizer_;
    std::vector<std::shared_ptr<ILayer>> network_;
    Eigen::VectorXd targets_;
    Eigen::MatrixXd netInput_;


    void shuffleData(DataHandler::dataPtrVector& data);
    bool isNetworkValid();
    void populateInputsAndTargets(DataHandler::dataPtrVector& data,
        uint32_t startIdx, uint32_t endIdx);
    void forwardPass(bool skipDropout);
    void calculateNetLoss();
    void addRegularizationLoss();
    void backPropagate();
    void optimize();
};
#endif