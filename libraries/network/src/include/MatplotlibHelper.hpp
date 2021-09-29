#ifndef __MatplotlibHelper__
#define __MatplotlibHelper__

#include <iostream>

class MatplotlibHelper {
public:
    //Constructor of class takes following arguments:
    //Max number of samples on X-Axis to have fixed size plot
    //Label that need to be shown on top of the plot
    //Pairs of series label and color to be used
    MatplotlibHelper(int xAxisMaxLimit, std::string title,
        std::vector<std::pair<std::string, std::string>> attributes);
    //Size of this vector must equal to number of pair class initialized with
    void draw(std::vector<double> data);
private:
    std::vector<std::vector<double>> data_;
    std::vector<std::pair<std::string, std::string>> attributes_;
    int id_;;
};
#endif