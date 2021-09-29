#include "SpiralDataHandler.hpp"
#include <eigen3/Eigen/Core>
#include <iostream>

std::shared_ptr<SpiralDataHandler> handlerPtr = nullptr;

DataHandler& DataHandler::getInstance(uint32_t numSamples, uint32_t numClasses) {
    if(handlerPtr == nullptr) {
        handlerPtr = std::make_shared<SpiralDataHandler>(numSamples, numClasses);
    }
    return *handlerPtr;
}

DataHandler::DataHandler() {
}

DataHandler::~DataHandler() {
}

void DataHandler::cleanup() {
    handlerPtr = nullptr;
}

SpiralDataHandler::SpiralDataHandler(uint32_t numSamples, uint32_t numClasses)
    : numSamples_(numSamples), numClasses_(numClasses) {
    if(numSamples % numClasses) {
        throw "Number of samples should be multiples of number of classes";
    }
    dataStore_.resize(numSamples_);
    samplesPerClass_ = numSamples/numClasses;
    dataMin_ = std::numeric_limits<double>::max();
    dataMax_ = std::numeric_limits<double>::min();
    generateData();
}

SpiralDataHandler::~SpiralDataHandler() {
  dataStore_.clear();
  trainingData_.clear();
  testData_.clear();
  validationData_.clear();
}

void SpiralDataHandler::generateData() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // instance of class std::normal_distribution with zero mean and stddev of 1
    std::normal_distribution<float> d(0, 1);
    std::vector<double> radius = linspace(0.0, 1.0, samplesPerClass_);
    std::vector<double> theta(samplesPerClass_);
    double rand_theta;
    double x_coord;
    double y_coord;
    uint32_t sampleIdx = 0;
    for(uint8_t class_num = 0; class_num < numClasses_; ++class_num) {
        theta.clear();
        theta = linspace(class_num*4, (class_num+1)*4, numSamples_);
        for(uint32_t sample = 0; sample < samplesPerClass_; ++sample) {
            //rand_theta = theta[sample] + d(gen) * 0.2;
            rand_theta = theta[sample] + d(gen) * 0.05;
            x_coord = radius[sample] * sin(rand_theta * 2.5);
            y_coord = radius[sample] * cos(rand_theta * 2.5);
            //Find min/max to be used in normalization later
            if(y_coord < dataMin_) {
              dataMin_ = y_coord;
            }
            if(y_coord > dataMax_) {
              dataMax_ = y_coord;
            }
            dataStore_[sampleIdx] = std::make_shared<Data>();
            dataStore_[sampleIdx]->features = std::vector<double>{x_coord, y_coord};
            dataStore_[sampleIdx]->label = class_num;
            sampleIdx++;
        }
    }
    //Shuffle data to get randomization
    std::shuffle(dataStore_.begin(), dataStore_.end(), gen);
}

std::vector<double> SpiralDataHandler::linspace(double start, double end, int num)
{
  std::vector<double> linspaced;

  if (num == 0) { return linspaced; }
  if (num == 1)
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end);
  return linspaced;
}

void SpiralDataHandler::splitData(double trainPct, double testPct, double validPct)
{
    // Training Data: will hold shared pointer to data array vector
    uint32_t trainIdx = 0;
    uint32_t testIdx = 0;
    uint32_t validIdx = 0;

    //Check validity of splitting data to training, validation, and testing
    if((trainPct + testPct + validPct) != 1) {
        throw "Training, validation, and testing percentages must add up to 1";
    }
    if(fmod(numSamples_ * trainPct,1) != 0) {
        throw "Training data percentage doesn't yield to integer";
    }
    trainingData_.resize(numSamples_ * trainPct);

    if(fmod(numSamples_ * testPct,1) != 0) {
        throw "Test data percentage doesn't yeld to integer";
    }
    testData_.resize(numSamples_ * testPct);

    if(fmod(numSamples_ * validPct,1) != 0) {
        throw "Validation data percentage doesn't yeld to integer";
    }
    validationData_.resize(numSamples_ * validPct);

    //Populate training, validation, and test vectors
    for(uint32_t storeIdx = 0; storeIdx < dataStore_.size(); storeIdx++) {
        if((storeIdx % samplesPerClass_) < (samplesPerClass_ * trainPct)) {
            trainingData_[trainIdx++] = dataStore_[storeIdx];
        }
        else if((storeIdx % samplesPerClass_) <
            (samplesPerClass_ * trainPct + samplesPerClass_ * testPct)) {
            testData_[testIdx++] = dataStore_[storeIdx];
        }
        else {
            validationData_[validIdx++] = dataStore_[storeIdx];
        }
    }
}

void SpiralDataHandler::normalizeData() {
    for(uint32_t idx = 0; idx < numSamples_; ++idx) {
        dataStore_[idx]->features[1] =
            static_cast<double>(dataStore_[idx]->features[1] - dataMin_)/(dataMax_ - dataMin_);
    }
}

uint32_t SpiralDataHandler::getNumLabels() {
    return numClasses_;
}

DataHandler::dataPtrVector& SpiralDataHandler::getAllData() {
  return dataStore_;
}
DataHandler::dataPtrVector& SpiralDataHandler::getTrainData() {
    return trainingData_;
}

DataHandler::dataPtrVector& SpiralDataHandler::getTestData() {
    return testData_;
}

DataHandler::dataPtrVector& SpiralDataHandler::getValidData() {
    return validationData_;
}
