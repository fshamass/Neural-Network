CC=g++
SRC := $(ML_PROJ_ROOT)/Network/src
CFLAGS := -std=c++11 -g -DMNIST
INCLUDE_DIR := -I $(ML_PROJ_ROOT)/Network/include -I /usr/local/include/eigen3 \
	-I $(PYTHON_INSTALL_DIR)/include/python3.8 -I$(ML_PROJ_ROOT)/Input/$(INPUT_TYPE)/include \
	-I$(ML_PROJ_ROOT)/Input -I $(PYTHON_INSTALL_DIR)/lib/python3.8/site-packages/numpy/core/include \
	-I $(ML_PROJ_ROOT)/Network/include/interface

all: main

main : obj $(ML_PROJ_ROOT)/main.cpp Neuron.o Relu.o SoftMax.o DenseLayer.o NeuralNet.o CategCrossEntropy.o \
	MatplotlibHelper.o SGD.o RMSProp.o Adam.o
	$(CC) $(CFLAGS) $(PWD)/main.cpp -o main $(INCLUDE_DIR) -L $(PYTHON_INSTALL_DIR)/lib -lpython3.8 \
	-L $(ML_PROJ_ROOT)/Input/lib -ldata $(ML_PROJ_ROOT)/Network/obj/*.o

obj:
	if [ ! -d "./Network/obj" ]; then mkdir Network/obj; fi

Neuron.o : $(SRC)/Neuron.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Neuron.cpp -o $(ML_PROJ_ROOT)/Network/obj/Neuron.o $(INCLUDE_DIR)

Relu.o : $(SRC)/Relu.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Relu.cpp -o $(ML_PROJ_ROOT)/Network/obj/Relu.o $(INCLUDE_DIR)

SoftMax.o : $(SRC)/SoftMax.cpp
	$(CC) -c $(CFLAGS) $(SRC)/SoftMax.cpp -o $(ML_PROJ_ROOT)/Network/obj/SoftMax.o $(INCLUDE_DIR)

DenseLayer.o : $(SRC)/DenseLayer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/DenseLayer.cpp -o $(ML_PROJ_ROOT)/Network/obj/DenseLayer.o $(INCLUDE_DIR)

NeuralNet.o : $(SRC)/NeuralNet.cpp
	$(CC) -c $(CFLAGS) $(SRC)/NeuralNet.cpp -o $(ML_PROJ_ROOT)/Network/obj/NeuralNet.o $(INCLUDE_DIR)

CategCrossEntropy.o: $(SRC)/CategCrossEntropy.cpp
	$(CC) -c $(CFLAGS) $(SRC)/CategCrossEntropy.cpp -o $(ML_PROJ_ROOT)/Network/obj/CategCrossEntropy.o $(INCLUDE_DIR)

MatplotlibHelper.o: $(SRC)/MatplotlibHelper.cpp
	$(CC) -c $(CFLAGS) $(SRC)/MatplotlibHelper.cpp -o $(ML_PROJ_ROOT)/Network/obj/MatplotlibHelper.o $(INCLUDE_DIR)

SGD.o: $(SRC)/SGD.cpp
	$(CC) -c $(CFLAGS) $(SRC)/SGD.cpp -o $(ML_PROJ_ROOT)/Network/obj/SGD.o $(INCLUDE_DIR)

RMSProp.o: $(SRC)/RMSProp.cpp
	$(CC) -c $(CFLAGS) $(SRC)/RMSProp.cpp -o $(ML_PROJ_ROOT)/Network/obj/RMSProp.o $(INCLUDE_DIR)

Adam.o: $(SRC)/Adam.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Adam.cpp -o $(ML_PROJ_ROOT)/Network/obj/Adam.o $(INCLUDE_DIR)

clean :
	rm -rf main ./Network/obj