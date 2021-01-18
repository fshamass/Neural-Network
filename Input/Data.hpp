#ifndef __DATA__
#define __DATA__

#include <vector>
#include <iostream>
#include "stdint.h"
#include "stdio.h"

class DataHandler;
//A container to encapsulate data entry features and label
class Data
{
public:
    //Configurable - Number of features per data sample
    Data(uint32_t featuresSize);
    ~Data();
    uint8_t getLabel();
    uint32_t getFeatureSize();
    std::vector<double> getFeatures();
    friend class DataHandler;
private:
    std::vector<double> features_;
    uint8_t label_;
    uint32_t featuresSize_;
};
#endif
