#ifndef __SGD__
#define __SGD__

#include <memory>
#include <vector>
#include "IOptimizer.hpp"
#include "ILayer.hpp"
#include "NeuralNetDefines.hpp"

class SGD : public IOptimizer {
public:
    SGD(const std::vector<std::shared_ptr<ILayer>>& layers, optimizerParams params);
    ~SGD();
    void preUpdate() override;
    void update(uint32_t layer) override ;
    void postUpdate() override;
private:
    //Every layer has own momentums
    struct LayerMomentums {
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
    };
    double learningRate_;
    double currentLearningRate_;
    double decay_;
    double momentum_;
    uint64_t iterations_;
    std::vector<std::shared_ptr<LayerMomentums>> momentums_;
    const std::vector<std::shared_ptr<ILayer>> layers_;
};
#endif
