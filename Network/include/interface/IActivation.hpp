#ifndef __IACTIVATION__
#define __IACTIVATION__

#include <iostream>
#include <Eigen/Dense>

class IActivation {
public:
    virtual void forward(Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd& getOutput(void) = 0;
    virtual void backward(Eigen::MatrixXd& preGradients) = 0;
    virtual Eigen::MatrixXd& getGradients() = 0;
};
#endif