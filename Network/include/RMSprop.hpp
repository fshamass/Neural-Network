#ifndef __RMSPROP__
#define __RMSPROP__

#include <memory>
#include <vector>
#include "IOptimizer.hpp"
#include "DenseLayer.hpp"

class RMSprop : public IOptimizer {
public:
    RMSprop(const std::vector<std::shared_ptr<DenseLayer>>& layers, optimizerParams params);
    ~RMSprop();
    void preUpdate() override;
    void update(uint32_t layer) override ;
    void postUpdate() override;
private:
    //Every layer has own cache
    struct LayerCache {
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
    };
    double learningRate_;
    double currentLearningRate_;
    double decay_;
    double epsilon_;
    double rho_;
    uint64_t iterations_;
    std::vector<std::shared_ptr<LayerCache>> caches_;
    const std::vector<std::shared_ptr<DenseLayer>> layers_;
};
#endif
