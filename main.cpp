#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <random>
#include "DataHandler.hpp"
#include "NeuralNet.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plotSprialDataSets(DataHandler::dataPtrVector data,
    std::string label, std::string color) {
    std::vector<double> xCoord, yCoord;
    for(auto& elem:data) {
        xCoord.push_back(elem->features[0]);
        yCoord.push_back(elem->features[1]);
    }
    plt::scatter(xCoord, yCoord, 1.0, {{"label", label},{"color", color}} );
}

int main() {
    try {
        //Create 300 samples for 3 classes
        //Each Sample is shared_ptr to Data structure which has features size 2
        // (x and y coordinates of each point) and label (group 1, 2 or 3).
        //Split data into 150 samples for training, 75 samples for test and 75 samples for
        //validation

        auto& dataHandler = DataHandler::getInstance(6000, 3);
        dataHandler.splitData(0.9, 0.05, 0.05);

        std::cout << "Number of training samples: "
                << dataHandler.getTrainData().size() << std::endl;
        std::cout << "Number of validation samples: "
                << dataHandler.getValidData().size() << std::endl;
        std::cout << "Number of test samples: "
                << dataHandler.getTestData().size() << std::endl;

        auto& neuralNet = NeuralNet::getInstance(2, 300);
        neuralNet.setTrainData(dataHandler.getTrainData());
        neuralNet.setTestData(dataHandler.getTestData());
        neuralNet.setValidData(dataHandler.getValidData());

        //Add layers with number of neurons, activation and regularization
        neuralNet.addDenseLayer({6, Activation::RELU, 0.0, 0.0, 0.0, 0.0});
        neuralNet.addDenseLayer({3, Activation::SOFTMAX, 0.0, 0.0, 0.0, 0.0});
        //Set loss function
        neuralNet.setLossFunction(LossFunc::CATEGORICAL_CROSS_ENTROPY);

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
        neuralNet.setOptimizer(Optimizer::Adam, params);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        neuralNet.fit(100);
        neuralNet.evaluate();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Run Time = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[sec]" << std::endl;

//        plt::backend("WebAgg");
        plt::figure(1);
        plt::title("Input Spiral Data Sets");
        plotSprialDataSets(dataHandler.getTrainData(), "Train Data", "green");
        plotSprialDataSets(dataHandler.getValidData(), "Valid Data", "red");
        plotSprialDataSets(dataHandler.getTestData(),  "Test Data", "blue");
        plt::legend();
        //plt::show();

        plt::figure(2);
        plt::title("Training vs Validation Accuracy");
        plt::plot(neuralNet.getTrainAcc(), {{"label", "Train Acc"},{"color", "red"}});
        plt::plot(neuralNet.getValidAcc(), {{"label", "Valid Acc"},{"color", "blue"}});
        plt::legend();

        plt::figure(3);
        plt::title("Training vs Validation Loss");
        plt::plot(neuralNet.getTrainLoss(), {{"label", "Train Acc"},{"color", "red"}});
        plt::plot(neuralNet.getValidLoss(), {{"label", "Valid Acc"},{"color", "blue"}});
        plt::legend();

        plt::show();

        DataHandler::cleanup();
        NeuralNet::cleanup();

    } catch (const char* msg) {
        std::cout << "Exception occurred: " << msg << std::endl;
    }
    std::cout << "Press any key to continue .. " << std::endl;
    int c=getchar();
}
