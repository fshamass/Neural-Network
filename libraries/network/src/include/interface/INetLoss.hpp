#ifndef __INETLOSS__
#define __INETLOSS__

#include <vector>
#include <Eigen/Dense>

//Interface to calculate network loss and accuracy per data batch
class INetLoss {
public:
    //Pass the output from activation of last layer
    virtual void forward(Eigen::MatrixXd& predictedOutput, Eigen::VectorXd& targetOutput) = 0;
    //Calculate gradients from loss function
    virtual void backward(Eigen::MatrixXd& predictedOutput, Eigen::VectorXd& targetOutput) = 0;
    //Add L1/L2 Regularization if needed
    virtual void addRegularization(double regLoss) = 0;
    //Get average network loss
    virtual double getAveNetLoss() = 0;
    //Get network accuracy
    virtual double getNetAccuracy() = 0;
    virtual Eigen::VectorXd& getOutput() = 0;
    //Get Loss Gradients
    virtual Eigen::MatrixXd& getLossGradients() = 0;
};
#endif