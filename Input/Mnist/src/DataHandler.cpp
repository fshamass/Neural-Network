#include "../include/DataHandler.h"
#include <eigen3/Eigen/Core>

DataHandler::DataHandler(std::string featurePath, std::string labelPath)
: featurePath_(featurePath), labelPath_(labelPath) {
    dataArraySize_ = 0;
    trainDataArraySize_ = 0;
    testDataArraySize_ = 0;
    validDataArraySize_ = 0;
    dataMin_ = std::numeric_limits<double>::max();
    dataMax_ = std::numeric_limits<double>::min();
}

DataHandler::~DataHandler()
{
  dataStore_.clear();
  trainingData_.clear();
  testData_.clear();
  validationData_.clear();
}

void DataHandler::generateData() {
    readFeatureData(featurePath_);
    readLabelData(labelPath_);
}

void DataHandler::readFeatureData(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  uint32_t num_rows = 0;
  uint32_t num_cols = 0;

  uint8_t bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f)
  {
    int i = 0;
    while(i < 4)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            break;
          case 1:
            num_images = format(bytes);
            break;
          case 2:
            num_rows = format(bytes);
            break;
          case 3:
            num_cols = format(bytes);
            break;
        }
      }
      i++;
    }
    printf("Done getting file header.\n");
    dataArraySize_ = num_images;
    for(i = 0; i < dataArraySize_; i++) {
        auto dataPtr = std::make_shared<Data>(num_rows * num_cols);
        if(fread(&dataPtr->features_[0], sizeof(uint8_t), dataPtr->getFeatureSize(), f)) {
            std::cout << "Error encountered while loading data element from file .. exit" << std::endl;
            exit(1);
        }
        auto [min, max] = std::minmax_element(begin(dataPtr->features_), end(dataPtr->features_));
        if(*min < dataMin_) {
            dataMin_ = *min;
        }
        if(*max > dataMax_) {
            dataMax_ = *max;
        }
        dataStore_.push_back(dataPtr);
    }
    std::cout << "Successfully read " << dataStore_.size() << " data entries." << std::endl;
    std::cout << "Min value: " << dataMin_ << " Max value: " << dataMax_ << std::endl;
  }
  else {
      std::cout << "Invalid Input File Path" << std::endl;
      exit(1);
  }
}

void DataHandler::readLabelData(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  uint8_t bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f) {
    int i = 0;
    while(i < 2)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            break;
          case 1:
            num_images = format(bytes);
            break;
        }
      }
      i++;
    }
    //std::cout << "Debug: magic number = " << magic << std::endl;
    //std::cout << "Debug: num of lables = " << num_images << std::endl;

    if(num_images != dataArraySize_) {
        std::cout << "Error: Number of labels does not match number of images... exit" << std::endl;
        exit(1);
    }
    uint8_t lables[dataArraySize_];
    if(!fread(lables, sizeof(uint8_t), dataArraySize_, f))
    {
        std::cout << "Error encountered while loading lables from file .. exit" << std::endl;
        exit(1);
    }

    for(int j = 0; j < dataArraySize_; j++)
    {
        dataStore_[j]->label_ = lables[j];
    }

    std::cout << "Done getting Label header." << std::endl;
}
  else
  {
     std::cout << "Invalid Label File Path" << std::endl;
     exit(1);
  }
}

void DataHandler::splitData()
{
    trainDataArraySize_ = dataStore_.size() * TRAIN_SET_PERCENT;
    testDataArraySize_ = dataStore_.size() * TEST_SET_PERCENT;
    int valid_size = dataStore_.size() * VALID_SET_PERCENT;

    // Training Data: will hold shared pointer to data array vector
    int count = 0;
    int index = 0;
    while(count < trainDataArraySize_)
    {
        trainingData_.push_back(dataStore_[index++]);
        count++;
    }

    // Test Data: will hold shared pointer to data array vector
    count = 0;
    while(count < testDataArraySize_)
    {
        testData_.push_back(dataStore_[index++]);
        count++;
    }

    // Validation Data: will hold shared pointer to data array vector
    count = 0;
    while(count < valid_size)
    {
        validationData_.push_back(dataStore_[index++]);
        count++;
    }

    std::cout << "Training Data Size: " << trainingData_.size() << std::endl;
    std::cout << "Test Data Size: " << testData_.size() << std::endl;
    std::cout << "Validation Data Size: " << validationData_.size() << std::endl;
}

uint32_t DataHandler::getNumDataSamples()
{
  return dataArraySize_;
}

uint32_t DataHandler::getNumTrainingSamples()
{
  return trainDataArraySize_;
}

uint32_t DataHandler::getNumTestSamples()
{
  return testDataArraySize_;
}

uint32_t DataHandler::getNumValidationSamples()
{
  return validDataArraySize_;
}

uint32_t DataHandler::format(const unsigned char* bytes)
{
  return (uint32_t)((bytes[0] << 24) |
                    (bytes[1] << 16)  |
                    (bytes[2] << 8)   |
                    (bytes[3]));
}

void DataHandler::normalizeData() {
    for(uint32_t i = 0; i < dataArraySize_; ++i) {
        for(uint32_t j = 0; j < dataStore_[i]->features_.size(); ++j) {
            dataStore_[i]->features_[j] =
                static_cast<double>(dataStore_[i]->features_[j] - dataMin_)/(dataMax_ - dataMin_);
        }
    }
}

std::vector<std::shared_ptr<Data>> DataHandler::getTrainData() {
    return trainingData_;
}

std::vector<std::shared_ptr<Data>> DataHandler::getTestData() {
    return testData_;
}

std::vector<std::shared_ptr<Data>> DataHandler::getValidData() {
    return validationData_;
}
