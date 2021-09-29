#ifndef __IDATAHANDLER__
#define __IDATAHANDLER__

#include <vector>
#include <memory>

//Interface for interacting with Input Data to be used with Neural Network
class DataHandler {
public:
    //Struct to hold information about each input data entry
    struct Data {
        //Features of input data entry
        std::vector<double> features;
        //Label associated with input data entry
        uint8_t label;
    };
    static DataHandler& getInstance(uint32_t numSamples, uint32_t numClasses);
    static void cleanup();
    //Will normalize data, if needed, it should be called after generateData
    virtual void normalizeData() = 0;
    //Will split data to training, testing, and validation data
    virtual void splitData(double trainPct, double testPct, double validPct) = 0;

    virtual uint32_t getNumLabels() = 0;
    virtual std::vector<std::shared_ptr<DataHandler::Data>>& getAllData() = 0;
    virtual std::vector<std::shared_ptr<DataHandler::Data>>& getTrainData() = 0;
    virtual std::vector<std::shared_ptr<DataHandler::Data>>& getTestData() = 0;
    virtual std::vector<std::shared_ptr<DataHandler::Data>>& getValidData() = 0;
protected:
    DataHandler();
    ~DataHandler();
private:
    DataHandler(const DataHandler&) = delete;
};

#endif