# Neural Network Framework
## Description
Learning project to get insight of how neuron learn and the process of back propagation and optimization.<br>
There are sophisticated frameworks out there such as Keras/Caffee but I wanted to code everything from scratch to deeply understand what is going on behind the scene instead of just using the framework APIs. <br>
Since my passion is self driving cars, I only implemented the relative part of Neural Network - multi-class classification.
I learned a lot from this project and got deeper understanding of neural network learning challenges by playing with hyper parameters.

## Table of Contents
- [Code Structure](#code-structure)
- [Design](#design)
- [Sample Run](#sample-run)
- [Usage](#usage)
- [Next Steps](#next-steps)
- [License](#license)

## Code Structure
Code directory structure is organized as the following:<br><br>
![DirectoryDiagram](https://user-images.githubusercontent.com/29670728/147845259-ae43a9e2-d3ec-4695-8459-d0e6176efe84.png) <br><br>
On top level:<br>
*libraries* directory contains 2 subdirectories:
- input: Contains code for handling input data to the network. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to fine tune inout data (such as normalizing data, dividing data sets to training, validation and testing sets, etc)
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate inout.so
- network: contains code for neural network implementation. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to configure and train neural network.
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate network.so.<br>


*main.cpp* file is the application that build, configure, and train neural network.
Building main.cpp will generate NeuralNet application that interacts with both shared libraries.
The thought behind this partitioning is to avoid unnecessary code compilation. That is, for instance if experimenting with neural network configurations and/or hyper parameters, only main.cpp need to be rebuilt.

## Design

![NeuralNet_3](https://user-images.githubusercontent.com/29670728/147845114-7e009818-b16d-4d2e-aa2c-9dc8d01c3853.png)<br><br>
Application level:
- Application uses DataHandler interface to get instance that implements APIs included in the interface.
- Application uses NeuralNet interface to get instance that implements APIs included in the interface.<br>

Internal implementation level:
- DataHandler interface is used to abstract operations to be performed on network input data.
- ILayer, INetLoss, and IOptimizer interfaces are used to abstract layer, network loss and optimizer components functionality. *NeuralNetImpl* class use those interfaces to train network regardless of what type of layer, network loss or optimizer is used.
- Further more, IActivation interface is used to abstract activation function used. Layer, uses IActivation interface for forward and backward pass regardless or what activation function is used.


## Sample Run

Figures below show sample run for classifying 3 classes of spiral data. Settings used are the following: <br>
Total number of data samples = 6000 <br>
Training samples (90% of total samples) = 5400 <br>
Validation samples (5% of total samples) = 600 <br>
Test samples (5% of total samples) = 300 <br>
Batch size = 300 samples <br>
Network: 2 input neurons (x and y of each point) -> 6 hidden layer neurons w/ Relu activation -> 3 output layer neurons w/ softmax activation <br>
Loss Function: Categorical Cross Entropy <br>
Optimizer: Adam with following params: <br>
- Learning rate = 0.05 <br>
- Learning rate decay = 1e-5 <br>
- beta1 = 0.9 <br>
- beta2 = 0.999 <br>
- epsilon = 1e-7 <br>

Test set results: Accuracy = 98.7, Loss = 3.19

![all_input_data](https://user-images.githubusercontent.com/29670728/147860633-137886ce-f19c-4355-b567-8011433066d8.png)
![network_loss](https://user-images.githubusercontent.com/29670728/147860641-165e469e-8940-4fbe-aa52-5afcbb978d41.png)
![network_accuracy](https://user-images.githubusercontent.com/29670728/147860644-2782a4c4-cd0a-4b53-af29-52192a1c4ae2.png)

## Usage
Follow steps below to build and run application:
1. define env variable PYTHON_INSTALL_DIR. This is the Python installation directory needed for matplotlib library (used in plotting).
2. Add path to shared libraries (libraries/input/lib and libraries/network/lib) to LD_LIBRARY_PATH. If on MACOS, add to DYLD_LIBRARY_PATH.
For instance on my Mac Book Pro with Anaconda, I have the following
```
export PYTHON_INSTALL_DIR=/opt/anaconda3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHON_INSTALL_DIR/lib
export DYLD_LIBRARY_PATH=/Users/faiqshamass/Documents/Tutorials/ML_Code/Neural-Network/libraries/input/lib:/Users/faiqshamass/Documents/Tutorials/ML_Code/Neural-Network/libraries/network/lib
```
3. Build data input library (currently only spiral data is supported)
```
cd <workspace root>/libraries/input
make spiral
```
4. Build network library
```
cd <workspace root>/libraries/network
make
```
5. Build application
```
cd <workspace root>
make
```
6. Run application
```
cd <workspace root>
./NeuralNet
```

## Next Steps
- Need to experiment with layers and neurons to find out where bias vs variance occur and see if that is related to input type to this network.
- Visualize data distribution between layers. Capture and plot histogram to show input data distribution (especially after Relu activation)
- Neural network design is all driven by input data. I believe network design could be concluded by inspecting input data. What exactly to look for in input data is what I need to figure out.

## License
The code should be used for educational purposes only. It should not be sold or used in any commercial product without permission from author.
