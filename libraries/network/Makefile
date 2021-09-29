CC=g++
SRC := src
CFLAGS := -std=c++11 -g
LIB_NETWORK := libnetwork.so
INCLUDE_DIR := -I src/include -I /usr/local/include/eigen3 \
	-I$(PYTHON_INSTALL_DIR)/include/python3.9 -I ../input/interface \
	-I$(PYTHON_INSTALL_DIR)/lib/python3.9/site-packages/numpy/core/include \
	-I src/include/interface -I$(shell pwd)/../input/interface -I$(shell pwd)/interface

all: $(LIB_NETWORK)

$(LIB_NETWORK) : obj lib Neuron.o Relu.o SoftMax.o DenseLayer.o NeuralNetImpl.o \
	CategCrossEntropy.o SGD.o RMSProp.o Adam.o
	$(CC) $(CFLAGS) -shared -o lib/$(LIB_NETWORK) -L $(PYTHON_INSTALL_DIR)/lib -lpython3.9 \
	-L $(shell pwd)/../input/lib -ldata obj/*.o

obj:
	if [ ! -d "obj" ]; then mkdir obj; fi

lib:
	if [ ! -d "lib" ]; then mkdir lib; fi

Neuron.o : $(SRC)/Neuron.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Neuron.cpp -o obj/Neuron.o $(INCLUDE_DIR)

Relu.o : $(SRC)/Relu.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Relu.cpp -o obj/Relu.o $(INCLUDE_DIR)

SoftMax.o : $(SRC)/SoftMax.cpp
	$(CC) -c $(CFLAGS) $(SRC)/SoftMax.cpp -o obj/SoftMax.o $(INCLUDE_DIR)

DenseLayer.o : $(SRC)/DenseLayer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/DenseLayer.cpp -o obj/DenseLayer.o $(INCLUDE_DIR)

NeuralNetImpl.o : $(SRC)/NeuralNetImpl.cpp
	$(CC) -c $(CFLAGS) $(SRC)/NeuralNetImpl.cpp -o obj/NeuralNetImpl.o $(INCLUDE_DIR)

CategCrossEntropy.o: $(SRC)/CategCrossEntropy.cpp
	$(CC) -c $(CFLAGS) $(SRC)/CategCrossEntropy.cpp -o obj/CategCrossEntropy.o $(INCLUDE_DIR)

SGD.o: $(SRC)/SGD.cpp
	$(CC) -c $(CFLAGS) $(SRC)/SGD.cpp -o obj/SGD.o $(INCLUDE_DIR)

RMSProp.o: $(SRC)/RMSProp.cpp
	$(CC) -c $(CFLAGS) $(SRC)/RMSProp.cpp -o obj/RMSProp.o $(INCLUDE_DIR)

Adam.o: $(SRC)/Adam.cpp
	$(CC) -c $(CFLAGS) $(SRC)/Adam.cpp -o obj/Adam.o $(INCLUDE_DIR)

clean :
	rm -rf obj lib
