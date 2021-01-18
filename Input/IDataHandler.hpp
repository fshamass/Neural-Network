#ifndef __IDATAHANDLER__
#define __IDATAHANDLER__

#include <vector>
#include <memory>
#include "Data.hpp"

//Interface for Data handeling to work with Neural Network
class IDataHandler {
public:
    //Will normalize spiral data, if needed, it should be called after generateData
    virtual void normalizeData() = 0;
    //Will split spiral data to training, testing, and validation data
    virtual void splitData(double trainPct, double testPct, double validPct) = 0;

    virtual uint32_t getNumLabels() = 0;
    virtual std::vector<std::shared_ptr<Data>>& getAllData() = 0;
    virtual std::vector<std::shared_ptr<Data>>& getTrainData() = 0;
    virtual std::vector<std::shared_ptr<Data>>& getTestData() = 0;
    virtual std::vector<std::shared_ptr<Data>>& getValidData() = 0;
};
#endif