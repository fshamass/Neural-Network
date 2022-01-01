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
- [Credits](#credits)
- [License](#license)

## Organization
Code directory structure is organized as the following:<br><br>
![DirectoryDiagram](https://user-images.githubusercontent.com/29670728/147845259-ae43a9e2-d3ec-4695-8459-d0e6176efe84.png) <br><br>
There are 2 main directories:
- input: Contains code for handeling input data to the network. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to fine tune inout data (suc as normalizeing data, dividing data sets to training, validation and testing sets, etc)
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate inout.so
- network: contains code for neural network implementation. This directory contains 2 sub directories
   - interface: Contains the interface class used by application to configure and train neural network.
   - src: Contains the internal implementation of the interface APIs stated above
   There is also makefile to build this code and generate network.so.

## Design
Design was based 
