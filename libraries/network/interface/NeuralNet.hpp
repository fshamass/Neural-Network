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
    virtual void setTrainData(DataHandler::dataPtrVector& trainData) = 0;
    virtual void setTestData(DataHandler::dataPtrVector& testData) = 0;
    virtual void setValidData(DataHandler::dataPtrVector& validData) = 0;
    //New layer will always go to bottom of list
    //Number of Neurons and Activation Function
    virtual void addDenseLayer(DenseLayerParams params) = 0;
    //Dropout layer with dropout percentage
    virtual void addDropoutLayer( double dropoutRate) = 0;
    virtual void setLossFunction(LossFunc lossFunc) = 0;
    virtual void setOptimizer(Optimizer optimizer, optimizerParams params) = 0;
    virtual void fit(uint32_t epochs) = 0;
    virtual void evaluate() = 0;
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