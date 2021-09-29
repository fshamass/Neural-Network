#include "NeuralNetImpl.hpp"

std::shared_ptr<NeuralNet> netPtr = nullptr;

NeuralNetImpl::NeuralNetImpl(uint32_t numFeatures, uint32_t batchSize)
: numFeatures_(numFeatures), batchSize_(batchSize) {
    accuracy_ = 0;
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

void NeuralNetImpl::setTrainData(std::vector<std::shared_ptr<DataHandler::Data>>& trainData) {
    trainData_ = trainData;
}

void NeuralNetImpl::setTestData(std::vector<std::shared_ptr<DataHandler::Data>>& testData) {
    testData_  = testData;
}

void NeuralNetImpl::setValidData(std::vector<std::shared_ptr<DataHandler::Data>>& validData) {
    validData_ = validData;
}

void NeuralNetImpl::shuffleData(std::vector<std::shared_ptr<DataHandler::Data>>& data) {
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

void NeuralNetImpl::addLayer(uint32_t numNeurons, Activation activation,
    double weightRegularizerL1, double biasRegularizerL1,
    double weightRegularizerL2, double biasRegularizerL2) {
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
    #ifdef __DEBUG__
    std::cout << "New DenseLayer added" << std::endl;
    #endif

    DenseLayerParams denseLayerParams;
    denseLayerParams.numInputs = numInputs;
    denseLayerParams.numNeurons = numNeurons;
    denseLayerParams.batchSize = batchSize_;
    denseLayerParams.activation = activation;
    denseLayerParams.weightRegularizerL1 = weightRegularizerL1;
    denseLayerParams.biasRegularizerL1 = biasRegularizerL1;
    denseLayerParams.weightRegularizerL2 = weightRegularizerL2;
    denseLayerParams.biasRegularizerL2 = biasRegularizerL2;
    auto layer = std::make_shared<DenseLayer>(denseLayerParams);
    network_.push_back(layer);
}

void NeuralNetImpl::setLossFunction(LossFunc lossFunc) {
    std::shared_ptr<INetLoss> lossFunction;
    switch(lossFunc) {
        case LossFunc::CATEGORICAL_CROSS_ENTROPY: {
            lossFunction = std::make_shared<CategCrossEntropy>(
                (network_.back())->getActivOutput().cols(), batchSize_);
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

void NeuralNetImpl::forwardPass(Eigen::MatrixXd& input) {
    //First layer will process input to Network
    network_[0]->forward(input);
    //Pass data through all layers
    for(uint32_t layer = 1; layer < network_.size(); ++layer) {
        //Process activation output of previous layer
        network_[layer]->forward(network_[layer-1]->getActivOutput());
    }
}

void NeuralNetImpl::populateInputsAndTargets(std::vector<std::shared_ptr<DataHandler::Data>>& data,
    uint32_t startIdx, uint32_t endIdx, Eigen::MatrixXd& netInput, Eigen::VectorXd& targets) {
    //Collect batch of data
    for( uint32_t sample = startIdx; sample < endIdx; ++sample) {
        netInput.row(sample - startIdx) =
            Eigen::Map<Eigen::VectorXd>(data[sample]->features.data(),
                                        (data[sample]->features).size());
        targets(sample - startIdx) = data[sample]->label;
    }
    #ifdef __DEBUG__
    std::cout << "Network Input: \n" << netInput << std::endl;
    std::cout << "Target input:  \n" << targets.transpose() << std::endl;
    #endif
}

void NeuralNetImpl::calculateNetLoss(Eigen::VectorXd& targets) {
    //Get network error
    netLoss_->forward((network_.back())->getActivOutput(), targets);
}

void NeuralNetImpl::addRegularizationLoss() {
    double regularizationLoss = 0.0;
    //Get regularization loss for all layers
    for(int i = 0; i < network_.size(); ++i) {
        regularizationLoss += network_[i]->getRegularizationLoss();
    }
    //Add regularization loss to network loss
    netLoss_->addRegularization(regularizationLoss);
}

void NeuralNetImpl::backPropagate(Eigen::VectorXd& targets) {
    //Back propagation - last layer
    netLoss_->backward((network_.back())->getActivOutput(), targets);
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
        optimizer_->update(layer);
    }
    optimizer_->postUpdate();
}

void NeuralNetImpl::train(uint32_t epochs) {
    double aveNetworkError , aveNetAccuracy , networkLoss;
    Eigen::VectorXd targets(batchSize_);
    Eigen::MatrixXd netInput(batchSize_, numFeatures_);

    //Declare vectors to hold accuracy and loss data per epoch
    //Size is 2 for validation and accuracy
    std::vector<double> accData(2);
    std::vector<double> lossData(2);

    if(!isNetworkValid()) {
        throw "Invalid network configurations";
    }

    for(uint32_t epoch = 0; epoch < epochs; ++epoch) {
        //Run model validation on new epoch
        populateInputsAndTargets(validData_, 0, validData_.size(), netInput, targets);
        forwardPass(netInput);
        calculateNetLoss(targets);
        //Store network loss on validation set
        lossData[0] = netLoss_->getAveNetLoss();
        validLoss_.push_back(netLoss_->getAveNetLoss());
        //Store network accuracy on validation set
        accData[0] = netLoss_->getNetAccuracy();
        validAcc_.push_back(netLoss_->getNetAccuracy());
        //Shuffle input data with every epoch for best randomization
        shuffleData(trainData_);
        //std::cout << "epoch: " << epoch << " , ";
        for(uint32_t idx = 0; idx <= trainData_.size() - batchSize_; idx += batchSize_) {
            #ifdef __DEBUG__
            std::cout << "Processing Batch: "<< static_cast<int>((idx/batchSize_) + 1) << std::endl;
            #endif
            populateInputsAndTargets(trainData_, idx, idx+batchSize_, netInput, targets);
            forwardPass(netInput);
            calculateNetLoss(targets);
            backPropagate(targets);
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
    //Run model on test data series
    populateInputsAndTargets(validData_, 0, testData_.size(), netInput, targets);
    forwardPass(netInput);
    calculateNetLoss(targets);

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