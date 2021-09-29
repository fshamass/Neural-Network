#include "MatplotlibHelper.hpp"

namespace plt = matplotlibcpp;
static int plotId = 5;

MatplotlibHelper::MatplotlibHelper(int xAxisMaxLimit, std::string title,
    std::vector<std::pair<std::string, std::string>> attributes) : attributes_(attributes) {
    id_ = plotId++;
    plt::figure(id_);
    plt::title(title.c_str());
    plt::xlim(0, xAxisMaxLimit);
    data_.resize(attributes.size());
    for(int i=0;i<attributes.size();++i) {
        plt::plot(data_[i], {{"label", attributes_[i].first},{"color", attributes_[i].second}});
    }
    plt::legend();
}

void MatplotlibHelper::draw(std::vector<double> data) {
    plt::figure(id_);
    for(int i =0;i<data.size();++i) {
        data_[i].push_back(data[i]);
        plt::plot(data_[i], {{"label", attributes_[i].first},{"color", attributes_[i].second}});
    }
    plt::draw();
    plt::pause(0.05);
}