#include "NeuralNetImpl.hpp"
#include "CategCrossEntropy.hpp"

std::shared_ptr<NeuralNet> netPtr = nullptr;

NeuralNetImpl::NeuralNetImpl(uint32_t numFeatures, uint32_t batchSize)
: numFeatures_(numFeatures), batchSize_(batchSize) {
    accuracy_ = 0;

    targets_.resize(batchSize_);
    netInput_.resize(numFeatures_, batchSize_);
}

NeuralNet::NeuralNet() {
}

NeuralNet::~NeuralNet() {
}

NeuralNet& NeuralNet::getInstance(uint32_t numFeatures, uint32_t batchSize) {
    if(netPtr == nullptr) {
        netPtr = std::make_shared<NeuralNetImpl>(numFeatures, batchSize);
    }
    return *netPtr;
}

NeuralNetImpl::~NeuralNetImpl() {
    network_.clear();
}

void NeuralNet::cleanup() {
    netPtr = nullptr;
}

void NeuralNetImpl::setTrainData(DataHandler::dataPtrVector& trainData) {
    trainData_ = trainData;
}

void NeuralNetImpl::setTestData(DataHandler::dataPtrVector& testData) {
    testData_  = testData;
}

void NeuralNetImpl::setValidData(DataHandler::dataPtrVector& validData) {
    validData_ = validData;
}

void NeuralNetImpl::shuffleData(DataHandler::dataPtrVector& data) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

bool NeuralNetImpl::isNetworkValid() {
    bool retValue = true;
    if(trainData_.size() == 0) {
        std::cout << "Error: Training Data is not set" << std::endl;
        retValue = false;
    }
    if(batchSize_ > trainData_.size()) {
        std::cout << "Error: Batch size can't be higher than training data size" << std::endl;
        retValue = false;
    }
    if(validData_.size() == 0) {
        std::cout << "Error: Validation Data is not set" << std::endl;
        retValue = false;
    }
    if(network_.size() < 2) {
        std::cout << "Error: At least 2 layers needed for Neural Network" << std::endl;
        retValue = false;
    }
    if((LayerType::DROPOUT == network_[0]->getLayerType()) ||
       (LayerType::DROPOUT == network_[network_.size() - 1]->getLayerType())) {
        std::cout << "Error: Dropout Layer cannot be 1st or last layer" << std::endl;
        retValue = false;
    }
    if(netLoss_ == nullptr) {
        std::cout << "Error: Cost Function for the network is not set" << std::endl;
        retValue = false;
    }
    if(optimizer_ == nullptr) {
        std::cout << "Error: Optimizer for the network is not set" << std::endl;
        retValue = false;
    }
    return retValue;
}

void NeuralNetImpl::addDenseLayer(DenseLayerParams params) {
    uint32_t numInputs;
    //Check compatibility
    if(network_.empty()) {
        //First layer input is number of features
        numInputs = numFeatures_;
    }
    else {
        //Output size of last layer should be new layer inputs
        numInputs = (network_.back())->getOutputSize();
    }
    auto layer = std::make_shared<DenseLayer>(params, numInputs, batchSize_);
    network_.push_back(layer);
}

void NeuralNetImpl::addDropoutLayer( double dropoutRate) {
    if(network_.empty()) {
        throw "Dropout Layer cannot be first layer";
    }
    uint32_t numInputs = (network_.back())->getOutputSize();
    auto layer = std::make_shared<DropoutLayer>(numInputs, batchSize_, dropoutRate);
    network_.push_back(layer);
}

void NeuralNetImpl::setLossFunction(LossFunc lossFunc) {
    std::shared_ptr<INetLoss> lossFunction;
    switch(lossFunc) {
        case LossFunc::CATEGORICAL_CROSS_ENTROPY: {
            lossFunction = std::make_shared<CategCrossEntropy>(
                (network_.back())->getActivOutput().rows(), batchSize_);
        } break;
        default: {
            //Throw an exception if loss function is not set
            throw "Invalid Loss function";
        } break;
    }
    netLoss_ = lossFunction;
}

void NeuralNetImpl::setOptimizer(Optimizer optimizer, optimizerParams params) {
    switch(optimizer) {
        case Optimizer::SGD: {
            optimizer_ = std::make_shared<SGD>(network_, params);
        } break;
        case Optimizer::RMSprop: {
            optimizer_ = std::make_shared<RMSprop>(network_, params);
        }break;
        case Optimizer::Adam: {
            optimizer_ = std::make_shared<Adam>(network_, params);
        }break;
        default: {
        }
    }
}

void NeuralNetImpl::forwardPass(bool skipDropout) {
    //First layer will process input to Network
    network_[0]->forward(netInput_);
    //Pass data through all layers
    for(uint32_t layer = 1; layer < network_.size(); ++layer) {
        if((skipDropout == true) && (LayerType::DROPOUT == network_[layer]->getLayerType())) {
            //No check is made for out of boundary, dropout layer cannot be last layer
            //This check is already made in network validation
            layer++;
            network_[layer]->forward(network_[layer-2]->getActivOutput());
        } else {
            //Process activation output of previous layer
            network_[layer]->forward(network_[layer-1]->getActivOutput());
        }
    }
}

void NeuralNetImpl::populateInputsAndTargets(DataHandler::dataPtrVector& data,
    uint32_t startIdx, uint32_t endIdx) {
    //Collect batch of data
    for( uint32_t sample = startIdx; sample < endIdx; ++sample) {
        netInput_.col(sample - startIdx) =
            Eigen::Map<Eigen::VectorXd>(data[sample]->features.data(),
                                        (data[sample]->features).size());
        targets_(sample - startIdx) = data[sample]->label;
    }
}

void NeuralNetImpl::calculateNetLoss() {
    //Get network error
    netLoss_->forward((network_.back())->getActivOutput(), targets_);
}

void NeuralNetImpl::addRegularizationLoss() {
    double regularizedLoss = 0.0;
    //Get regularization loss for all layers
    for(int i = 0; i < network_.size(); ++i) {
        regularizedLoss += network_[i]->getRegularizedLoss();
    }
    //Add regularization loss to network loss
    netLoss_->addRegularization(regularizedLoss);
}

void NeuralNetImpl::backPropagate() {
    //Back propagation - last layer
    netLoss_->backward((network_.back())->getActivOutput(), targets_);
    network_[network_.size()-1]->backward(netLoss_->getLossGradients());
    //Back propagation - Subsequent layers
    for(int32_t layer = network_.size() - 2; layer >= 0; --layer) {
        network_[layer]->backward(network_[layer + 1]->getInputGradients());
    }
}

void NeuralNetImpl::optimize() {
    //Optimization of weights and biases
    optimizer_->preUpdate();
    for(uint32_t layer = 0; layer < network_.size(); ++layer) {
        //Skip optimization for dropout layers
        if(LayerType::DENSE == network_[layer]->getLayerType()) {
            optimizer_->update(layer);
        }
    }
    optimizer_->postUpdate();
}

void NeuralNetImpl::fit(uint32_t epochs) {
    double aveNetworkError , aveNetAccuracy , networkLoss;

    //Declare vectors to hold accuracy and loss data per epoch
    //Size is 2 for validation and accuracy
    std::vector<double> accData(2);
    std::vector<double> lossData(2);

    if(!isNetworkValid()) {
        throw "Invalid network configurations";
    }

    for(uint32_t epoch = 0; epoch < epochs; ++epoch) {
        //Run model validation on new epoch
        populateInputsAndTargets(validData_, 0, validData_.size());
        //Skip dropout layer during validation
        forwardPass(true);
        calculateNetLoss();
        //Store network loss on validation set
        lossData[0] = netLoss_->getAveNetLoss();
        validLoss_.push_back(netLoss_->getAveNetLoss());
        //Store network accuracy on validation set
        accData[0] = netLoss_->getNetAccuracy();
        validAcc_.push_back(netLoss_->getNetAccuracy());
        //Shuffle input data with every epoch for best randomization
        shuffleData(trainData_);
        for(uint32_t idx = 0; idx <= trainData_.size() - batchSize_; idx += batchSize_) {
            populateInputsAndTargets(trainData_, idx, idx+batchSize_);
            forwardPass(false);
            calculateNetLoss();
            backPropagate();
            optimize();
        }
        //Store network loss on last batch of training data (end of epoch)
        lossData[1] = netLoss_->getAveNetLoss();
        trainLoss_.push_back(netLoss_->getAveNetLoss());
        accData[1] = netLoss_->getNetAccuracy();
        trainAcc_.push_back(netLoss_->getNetAccuracy());
        std::cout << "epoch: " << epoch << " , ";
        std::cout << "accuracy: " << std::setprecision(3) << netLoss_->getNetAccuracy() << " , ";
        std::cout << "loss: " << std::setprecision(3) << netLoss_->getAveNetLoss() << std::endl;
    }
}

void NeuralNetImpl::evaluate() {

    //Run model on test data series
    populateInputsAndTargets(validData_, 0, validData_.size());
    forwardPass(true);
    calculateNetLoss();

    std::cout << "Test Data , " ;
    std::cout << "accuracy: " << std::setprecision(3) << netLoss_->getNetAccuracy() << " , ";
    std::cout << "loss: " << std::setprecision(3) << netLoss_->getAveNetLoss() << std::endl;
}

std::vector<double>& NeuralNetImpl::getTrainLoss() {
    return trainLoss_;
}

std::vector<double>& NeuralNetImpl::getTrainAcc() {
    return trainAcc_;
}

std::vector<double>& NeuralNetImpl::getValidLoss() {
    return validLoss_;
}

std::vector<double>& NeuralNetImpl::getValidAcc() {
    return validAcc_;
}