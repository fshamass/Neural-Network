#ifndef __ADAM__
#define __ADAM__

#include <memory>
#include <vector>
#include "IOptimizer.hpp"
#include "ILayer.hpp"
#include "NeuralNetDefines.hpp"

class Adam : public IOptimizer {
public:
    Adam(const std::vector<std::shared_ptr<ILayer>>& layers, optimizerParams params);
    ~Adam();
    void preUpdate() override;
    void update(uint32_t layer) override ;
    void postUpdate() override;
private:
    //Every layer has own cache
    struct LayerCache {
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
    };
    //Every layer has own momentums
    struct LayerMomentums {
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
    };
    double learningRate_;
    double currentLearningRate_;
    double decay_;
    double epsilon_;
    double beta1_;
    double beta2_;
    uint64_t iterations_;
    std::vector<std::shared_ptr<LayerCache>> caches_;
    std::vector<std::shared_ptr<LayerMomentums>> momentums_;
    const std::vector<std::shared_ptr<ILayer>> layers_;
};
#endif
