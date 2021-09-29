#ifndef __INEURAL_NET__
#define __INEURAL_NET__

#include <memory>
#include <vector>
#include "DataHandler.hpp"
#include "NeuralNetDefines.hpp"

class NeuralNet {
public:
    static NeuralNet& getInstance(uint32_t numFeatures, uint32_t batchSize);
    static void cleanup();
    virtual void setTrainData(std::vector<std::shared_ptr<DataHandler::Data>>& trainData) = 0;
    virtual void setTestData(std::vector<std::shared_ptr<DataHandler::Data>>& testData) = 0;
    virtual void setValidData(std::vector<std::shared_ptr<DataHandler::Data>>& validData) = 0;
    //New layer will always go to bottom of list
    //Number of Neurons and Activation Function
    virtual void addLayer(uint32_t numNeurons, Activation activation,
        double weightRegularizerL1=0, double biasRegularizerL1=0,
        double weightRegularizerL2=0, double biasRegularizerL2=0) = 0;
    virtual void setLossFunction(LossFunc lossFunc) = 0;
    virtual void setOptimizer(Optimizer optimizer, optimizerParams params) = 0;
    virtual void train(uint32_t epochs) = 0;
    virtual std::vector<double>& getTrainLoss() = 0;
    virtual std::vector<double>& getValidLoss() = 0;
    virtual std::vector<double>& getTrainAcc()  = 0;
    virtual std::vector<double>& getValidAcc()  = 0;
protected:
    NeuralNet();
    ~NeuralNet();
private:
    NeuralNet(const NeuralNet&);
};

#endif