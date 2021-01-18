#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include "fstream"
#include "stdint.h"
#include "Data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <random>
#include <math.h>


class DataHandler
{
public:
    DataHandler(std::string featurePath, std::string labelPath);
    ~DataHandler();

    //Will generate data structures for feature and lables
    void generateData();
    //Will normalize spiral data, if needed, it should be called after generateData
    void normalizeData();
    //Will split data read from files above to training, testing, and validation data
    void splitData();

    uint32_t getNumDataSamples();
    uint32_t getNumTrainingSamples();
    uint32_t getNumTestSamples();
    uint32_t getNumValidationSamples();

    uint32_t format(const unsigned char* bytes);

    std::vector<std::shared_ptr<Data>> getTrainData();
    std::vector<std::shared_ptr<Data>> getTestData();
    std::vector<std::shared_ptr<Data>> getValidData();

private:
    const double TRAIN_SET_PERCENT = 0.1;
    const double TEST_SET_PERCENT = 0.075;
    const double VALID_SET_PERCENT = 0.005;
    uint32_t dataArraySize_;
    uint32_t trainDataArraySize_;
    uint32_t testDataArraySize_;
    uint32_t validDataArraySize_;
    std::vector<std::shared_ptr<Data>> dataStore_;
    std::vector<std::shared_ptr<Data>> trainingData_;
    std::vector<std::shared_ptr<Data>> testData_;
    std::vector<std::shared_ptr<Data>> validationData_;
    std::string featurePath_;
    std::string labelPath_;
    //Find min/max of data to be used in normalization later
    double dataMin_;
    double dataMax_;
    void readFeatureData(std::string path);
    void readLabelData(std::string path);
};

#endif
