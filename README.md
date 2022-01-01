# Neural Network Framework
## Description
Learning project to get insight of how neuron learn and the process of backpropagation and optimization.<br>
There are sophisticated frameworks out there such as Keras/Caffee but I wanted to code it from scratch to deeply understand what is going on behind the scene instead of
just using the framwork APIs. <br>
Since my ultiamte goal is to learn self driving cars, I only implemented multiclass classification framework.
I learned a lot from this project and got deeper understanding of neural network learing challenges by playing with hyper parameters.

## Table of Contents
- [Organization](#organization)
- [Design](#design)
- [Sample Run](#sample-run)
- [License](#license)

## Organization
Code directory structure is organized as the following:<br><br>
![DirectoryDiagram](https://user-images.githubusercontent.com/29670728/147845259-ae43a9e2-d3ec-4695-8459-d0e6176efe84.png) <br><br>
On top level:<br>
*libraries* directory contains 2 subdirectories:
- input: Contains code for handeling input data to the network. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to fine tune inout data (suc as normalizeing data, dividing data sets to training, validation and testing sets, etc)
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate inout.so
- network: contains code for neural network implementation. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to configure and train neural network.
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate network.so.<br>
   
   
*main.cpp* file is the application that build, configure, and train neural network.
Building main.cpp will generate NeuralNet application that interacts with both shared libraries.
The thought behind this partitioning is to avoid unnecessary code compilation. That is, for instance if experiementing with neural network configurations and/or hyper parameters, only main.cpp need to be rebuilt. 

## Design

![NeuralNet_3](https://user-images.githubusercontent.com/29670728/147845114-7e009818-b16d-4d2e-aa2c-9dc8d01c3853.png)<br><br>
Application level:
- Application uses DataHandler interface to get instance that implements APIs included in the interface.
- Application uses NeuralNet interface to get instance that implements APIs included in the interface.<br>

Internal implementation level: 
- DataHandler interface is used to abstract operations to be performed on network input data.
- ILayer, INetLoss, and IOptimizer interfaces are used to abstract layer, network loss and optimizer componenents functionality. *NeuralNetImpl* class ised those interfaces to train network regardless of waht type of layer, network loss or optimizer is used.
- Furthur more, IActivation interface is used to abstract activation function used. Layer, uses IActivation interface for forward and backward pass regadless or what activation function is used.


## Sample Run
![all_input_data](https://user-images.githubusercontent.com/29670728/147860633-137886ce-f19c-4355-b567-8011433066d8.png)
![network_loss](https://user-images.githubusercontent.com/29670728/147860641-165e469e-8940-4fbe-aa52-5afcbb978d41.png)
![network_accuracy](https://user-images.githubusercontent.com/29670728/147860644-2782a4c4-cd0a-4b53-af29-52192a1c4ae2.png)

