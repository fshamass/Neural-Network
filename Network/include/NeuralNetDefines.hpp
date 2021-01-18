#ifndef __NEURALNETDEFINES__
#define __NEURALNETDEFINES__

enum class Activation {
    RELU = 0,
    SOFTMAX = 1
};

enum class LossFunc {
    CATEGORICAL_CROSS_ENTROPY = 0,
};

enum class Optimizer {
    SGD     = 0,
    RMSprop = 1,
    Adam    = 2,
};

struct DenseLayerParams {
    uint32_t    numInputs;
    uint32_t    numNeurons;
    uint32_t    batchSize;
    double      weightRegularizerL1;
    double      biasRegularizerL1;
    double      weightRegularizerL2;
    double      biasRegularizerL2;
    Activation  activation;
};

struct optimizerParams {
    optimizerParams() : learnRate(0.001), decay(0.0), momentum(0.0)
    , epsilon(1e-7), rho(0.9), beta1(0.9), beta2(0.999) {}
    double learnRate  ;    /*   Learning Rate                         */
    double decay      ;    /*   Learning Rate Decay                   */
    double momentum   ;    /*   Momentum - used with SGD              */
    double epsilon    ;    /*   Used with RMSProp                     */
    double rho        ;    /*   Gradient Moving Ave used with RMSProp */
    double beta1      ;    /*   Used with Adam                        */
    double beta2      ;    /*   Used with Adam                        */
};

#endif