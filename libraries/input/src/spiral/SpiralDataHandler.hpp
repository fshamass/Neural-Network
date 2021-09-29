#ifndef __DATA_HANDLER__
#define __DATA_HANDLER__

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <random>
#include <cmath>
#include <math.h>
#include <limits>
#include "fstream"
#include "stdint.h"
#include "DataHandler.hpp"

/***********************************************************************
 * Handeles generation of spiral data groups.
 * Data is generated at construction of class as pointers to Data class
 * It also offers API to split and optionally normalize data to
 * training, validation, and test groups based on percentages provided.
 * Class throws following exceptions:
 *   Number of samples should be multiples of number of classes
 *   Training, validation, and testing percentages must add up to 1
 *   Training data percentage doesn't yield to integer
 *   Test data percentage doesn't yeld to integer
 *   Validation data percentage doesn't yeld to integer
 * *********************************************************************/

class SpiralDataHandler : public DataHandler {
public:
    SpiralDataHandler(uint32_t numSamples, uint32_t numClasses);
    ~SpiralDataHandler();

    //Normalize spiral data, if needed. It should be called after class construction
    //and before splitting data between training, validations and testing
    void normalizeData() override;
    //Will split spiral data to training, testing, and validation data
    void splitData(double trainPct, double testPct, double validPct) override;

    //getter APIs for all vectors
    uint32_t getNumLabels() override;
    std::vector<std::shared_ptr<DataHandler::Data>>& getAllData() override;
    std::vector<std::shared_ptr<DataHandler::Data>>& getTrainData() override;
    std::vector<std::shared_ptr<DataHandler::Data>>& getTestData() override;
    std::vector<std::shared_ptr<DataHandler::Data>>& getValidData() override;

private:
    void generateData();
    std::vector<double> linspace(double start, double end, int num);
    uint32_t numSamples_;
    uint32_t numClasses_;
    uint32_t samplesPerClass_;
    //Find min/max of data to be used in normalization later
    double dataMin_;
    double dataMax_;
    //Vectors to hold pointers to different data sets
    std::vector<std::shared_ptr<DataHandler::Data>> dataStore_;
    std::vector<std::shared_ptr<DataHandler::Data>> trainingData_;
    std::vector<std::shared_ptr<DataHandler::Data>> testData_;
    std::vector<std::shared_ptr<DataHandler::Data>> validationData_;

};

#endif
