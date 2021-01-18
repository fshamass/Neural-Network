#include <iostream>
#include <chrono>
#include "Neuron.hpp"
#include "NeuralNet.hpp"
#include "matplotlibcpp.h"
#include "DataHandler.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plotSprialDataSets(std::vector<std::vector<std::shared_ptr<Data>>> data) {
    std::vector<double> xCoord, yCoord;
    plt::figure(4);
    plt::title("Input Spiral Data Sets");
    for(auto& series:data){
        xCoord.clear();
        yCoord.clear();
        for(auto& elem:series) {
            xCoord.push_back(elem->getFeatures()[0]);
            yCoord.push_back(elem->getFeatures()[1]);
        }
        plt::scatter(xCoord, yCoord);
    }
    plt::draw();
    plt::pause(0.05);
}

int main() {
//  Eigen::Matrix3f m3;
//  m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
//  Eigen::Matrix3f m4;
//  m4 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
//  Eigen::Matrix3f m5;
//  m5 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
//  Eigen::Matrix3f m6 = m4.array().square() /  m5.array() + 0.1;
//  std::cout << m3.cwiseProduct(m3) << std::endl;
//  std::cout << m3.array().square().sum() << std::endl;
//  return 1;
//  std::cout << m3.array()/pow(2,2) << std::endl;
//  std::cout <<  m6 << std::endl;
//  std::cout << m3.cwiseSqrt().unaryExpr([](double x) {return x + 0.1;}) << std::endl;
//  std::cout << m3/m4 << std::endl;
//  return 1;
//x = Eigen::VectorXd::LinSpaced(3, 1, 10);
//std::cout << "x:\n" << x << std::endl;
//updateBias();
//std::cout << "x:\n" << x << std::endl;
//std::cout << "x transposed:\n" << x.transpose() << std::endl;
//std::cout << "x transposed replicate:\n" << x.transpose().replicate(3,1) << std::endl;
//return 1;

    try {
        //Create 300 samples for 3 classes
        //Each Sample is shared_ptr to Data structure which has features size 2
        // (x and y coordinates of each point) and label (group 1, 2 or 3).
        DataHandler dataHandler(6000, 3);
        //Split data into 150 samples for training, 75 samples for test and 75 samples for validation

        dataHandler.splitData(0.9, 0.05, 0.05);

        std::cout << "Number of training samples: "
                << dataHandler.getTrainData().size() << std::endl;
        std::cout << "Number of validation samples: "
                << dataHandler.getValidData().size() << std::endl;
        std::cout << "Number of test samples: "
                << dataHandler.getTestData().size() << std::endl;

        std::vector<std::vector<std::shared_ptr<Data>>> data;
        data.push_back(dataHandler.getTrainData());
        data.push_back(dataHandler.getValidData());
        data.push_back(dataHandler.getTestData());
        plotSprialDataSets(data);

        NeuralNet* neuralNet = new NeuralNet(2,300);
        neuralNet->setTrainData(dataHandler.getTrainData());
        neuralNet->setTestData(dataHandler.getTestData());
        neuralNet->setValidData(dataHandler.getValidData());

        neuralNet->addLayer(6, Activation::RELU, 0, 0, 50, 50);
        neuralNet->addLayer(3, Activation::SOFTMAX);
        neuralNet->setLossFunction(LossFunc::CATEGORICAL_CROSS_ENTROPY);
        optimizerParams params;
        /* Params for SGD
        params.learnRate = 1;
        params.decay = 1e-3;
        params.momentum = 0.5;
        neuralNet->setOptimizer(Optimizer::SGD, params);
        */

        /* Params for RMS Prop
        params.learnRate = 0.001;
        params.decay = 1e-4;
        params.epsilon = 1e-7;
        params.rho = 0.9;
        neuralNet->setOptimizer(Optimizer::RMSprop, params);
        */

        /* Params for Adam */
        params.learnRate = 0.05;
        params.decay = 1e-5;
        neuralNet->setOptimizer(Optimizer::Adam, params);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        neuralNet->train(100);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Run Time = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[sec]" << std::endl;
    } catch (const char* msg) {
        std::cout << "Exception occurred: " << msg << std::endl;
    }
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
//plt::plot(dataHandler.x_coord, dataHandler.y_coord);
//std::vector<double> class1x;
//std::vector<double> class1y;
//for(int i=0;i<dataHandler.getNumTrainingSamples();++i)
//{
//    std::vector<std::shared_ptr<Data>> tmp = dataHandler.getTrainData();
//    class1x.push_back((tmp[i]->getFeatures())[0]);
//    class1y.push_back((tmp[i]->getFeatures())[1]);
//}
//plt::scatter(class1x, class1y);
/*
std::vector<double> class2x(&dataHandler.x_coord[100], &dataHandler.x_coord[200]);
std::vector<double> class2y(&dataHandler.y_coord[100], &dataHandler.y_coord[200]);
plt::scatter(class2x, class2y);

std::vector<double> class3x(&dataHandler.x_coord[200], &dataHandler.x_coord[300]);
std::vector<double> class3y(&dataHandler.y_coord[200], &dataHandler.y_coord[300]);
plt::scatter(class3x, class3y);
*/
//plt::title("Three Groups Of Spiral Data");
//plt::save("sprialData.png");
//plt::show();

//int pos;
//Eigen::VectorXd v1;
//v1 << 1.5, 0.026, 4.015, 2.91, 6.0;
//std::cout << "Max coeff is : " << v1.maxCoeff(&pos) << " At position: " << pos << std::endl;

//    plt::plot({1,3,2,4});
//    plt::show();

//Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(3, 1, 10);
//std::cout << "x:\n" << x << std::endl;
//
//Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(5, 6.0, 23.4);
//std::cout << "y:\n" << y << std::endl;
//
//Eigen::VectorXd z = Eigen::VectorXd::LinSpaced(1, 0.0, 2.0);
//std::cout << "z:\n" << z << std::endl;
//
//std::cout << "y(2):\n" << y(2) << std::endl;

//  Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(200, 0, 6);
//  Eigen::VectorXd y, z;

  // y = exp(sin(x)), z = exp(cos(z))
    //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> mymap(NULL);
    //new (&(mymap.row(0))) Eigen::Map<Eigen::RowVectorXd>(&v1[0], 1, v1.size());
    //mymap.resize(2,10);
    //mymap << v1;
    //Eigen::MatrixXd mymap(2,10);
    //Eigen::Map<Eigen::RowVectorXd> v(v0.data(),v0.size());
    //new (&v) Eigen::Map<Eigen::RowVectorXd>(v1.data(),v1.size());
    //mymap << v;
    //Eigen::VectorXd::Map(&v1[0], v1.size()) = mymap.row(0);
    //mymap << Eigen::Map<Eigen::VectorXd>(&v1[0], v1.size());
    //Eigen::VectorXi::Map(&v2[0], v2.size()) = mymap.row(1);
    //mymap << v2;
    //Eigen::VectorXi mymap = Eigen::Map(&v1[0], v1.size());
    //std::cout << mymap << std::endl;
//VectorXd::Map(&v2[0], v1.size()) = v1;
//    double n1 = 1.56527e-07 + 0.00263076 * 2 + 0.0151121 * 3;
//    double n2 = 0.009173 + 0.0106553 * 2 + 0.00437918 * 3;
//    double n3 = 0.000940892 + 0.0135773 * 2 + 0.0135859 * 3;
//    double nw = n1 * 0.0186939 + n2 * 0.00767004 + n3 * 0.0103883;
//    std::cout << "Manual calc is: " << nw << std::endl;

//   Neuron* neuron = new Neuron(4);
//   neuron->init();
//   Eigen::VectorXd v2(4);
//   v2 << 1000.0,2000.0,3000.0,4000.0;
//   neuron->forward(v2);
//   std::cout << "Dot product is: " << neuron->getOutput() << std::endl;

//   Eigen::VectorXd v1(4);
//   v1 << 1.0, 2.0, 3.0, 4.0;
//   SoftMax* softMax = new SoftMax(4);
//   softMax->processInput(v1);
//   std::cout << v1.array().exp() << std::endl;

// Dot product is: 87.29

//Eigen::VectorXd v1(4);
//v1 << 1.56527e-07, 0.00263076, 0.0151121, 0.009173;
//Eigen::VectorXd v2(4);
//v2 << 1000.0,2000.0,3000.0,4000.0;
//std::cout << "Dot product is: " << v1.dot(v2) << std::endl;

//    auto dataHandle = std::make_shared<DataHandler>();
//    dataHandle->readData("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
//
//    dataHandle->printDataSample(2);
//using namespace Eigen;
//using namespace std;//
//  Matrix3f m3;
//  m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
//  Matrix4f m4 = Matrix4f::Identity();
//  Vector4i v4(1, 2, 3, 4);
//  std::cout << "m3\n" << m3 << "\n m3 rows = " <<m3.rows() << " cols: " << m3.cols() << " m4:\n"
//    << m4 << "\nv4:\n" << v4 << std::endl;
// g++ -std=c++17 -I./include -o main ./src/*

//    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> inData(input.data(), input.size());
//    Eigen::Matrix<double, 1, 1> dotMat = weights_ * inData;

//Eigen::Vector3i v = Eigen::Vector3i::Random().transpose();
//Eigen::MatrixXd v = Eigen::MatrixXd::Zero(1,3);
//v << 1,2,3;
//std::cout << "Here is the vector v:" << std::endl << v << std::endl;
//std::cout << "v.rowwise().replicate(5) = ..." << std::endl;
//std::cout << v.rowwise().replicate(5) << std::endl;
//std::cout << "v.colwise().replicate(5) = ..." << std::endl;
//std::cout << v.colwise().replicate(5) << std::endl;
//std::cout << v.replicate(3,1) << std::endl;
//return 0;
//    Eigen::MatrixXd mymap(4,4);
//    mymap << 1,2,0,3,0,2,4,1,0,0,6,7,8,9,3,0;
//    std::cout << "mymap: \n" << mymap << std::endl;
//    Eigen::MatrixXd mymap2 = mymap.setZero();
//    std::cout << "mymap2: \n" << mymap2 << std::endl;
//    Eigen::MatrixXd mymap2 = mymap.row(0).matrix().asDiagonal();
//    std::cout << "as diagonal: \n" << mymap2 << std::endl;
//    Eigen::MatrixXd mymap3 = mymap.unaryExpr([](double x) {return log(x);});
//    std::cout << "as exp: \n" << mymap3 << std::endl;
//    std::cout << "mymap: \n" << mymap << std::endl;

//    std::cout << mymap - mymap.rowwise().maxCoeff();
//    std::cout << mymapVector3i v = Vector3i::Random();
//    Eigen::MatrixXd temp = mymap.exp();
//    temp = temp.exp();
//    std::cout << temp << std::endl;
//    return 0;
//    Eigen::MatrixXd mymap2(4,4);
//    Eigen::VectorXd row0(3);
//    mymap2 << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
//    std::cout << mymap.unaryExpr([&mymap2](double x) {mymap2.unaryExpr([&x](double y) {return ((x==0) ? x : y);});}) << std::endl;
//    std::cout << mymap.binaryExpr(mymap2, [](double x, double y) {return ((x==0)?x:y);});
//    return 0;
//   mymap = mymap.unaryExpr([](double x) {return (x >= 1) ? 1-1e-7 : x;});
//    row0 << 0.7 , 0.1 , 0.2;
//    std::cout << "Dot prod of 2 mat: \n" << m0.dot(m1);

//    std::cout << "Dot Prod: \n" <<m1.dot(m0) << std::endl;
//    std::cout << (row0.array() * col0.array()).colwise().sum() << std::endl;
//    std::cout << "Initial row0: \n" << row0 << std::endl;
//    std::cout << "row0 attrib: \n" << row0.matrix().rows() << row0.matrix().cols() << std::endl;
//    std::cout << "matrix row0: \n" << (row0.matrix()) * (row0.matrix().transpose()) << std::endl;
//    std::cout << "Dot Product: \n" << Eigen::Matrix<double,4,1>(row0).dot(Eigen::Matrix<double,1,4>(col0)) << std::endl;
//    std::cout << Eigen::Map<Eigen::Matrix<double,4,1>>(row0).dot(Eigen::Map<Eigen::Matrix<double,1,4>>(col0)) << std::endl;

//    return 0;
//    mymap.row(0) = row0;
//    Eigen::VectorXd row1;
//    row1 << 11,12,13,14;
//    mymap.row(1) = row1;
//    std::cout << mymap << std::endl;
//    mymap << 0.5,0.3,1,0.7,0,0.3,0.2,0,1;
//    std::cout << mymap << std::endl;
//    mymap = mymap.cwiseMin(1e-7).cwiseMax(1-1e-7);
//    mymap = mymap.unaryExpr([](double x) {return (x <= 0) ? -log(1e-7) : (x >= 1) ? -log(1-1e-7) : -log(x);});
//   mymap = mymap.unaryExpr([](double x) {return (x >= 1) ? 1-1e-7 : x;});
//    std::cout << std::endl;
//    std::cout << std::setprecision(10) << mymap << std::endl;


    std::cout << "Press any key to continue .. " << std::endl;
    int c=getchar();
}
