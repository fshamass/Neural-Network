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
    double learnRate  = 0.001;    /*   Learning Rate                         */
    double decay      = 0.0  ;    /*   Learning Rate Decay                   */
    double momentum   = 0.0  ;    /*   Momentum - used with SGD              */
    double epsilon    = 1e-7 ;    /*   Used with RMSProp                     */
    double rho        = 0.9  ;    /*   Gradient Moving Ave used with RMSProp */
    double beta1      = 0.9  ;    /*   Used with Adam                        */
    double beta2      = 0.999;    /*   Used with Adam                        */
};

#endif