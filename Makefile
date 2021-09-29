CC=g++
CFLAGS := -std=c++11 -g
PROJ_ROOT := $(shell pwd)

INCLUDE_DIR := -I libraries/input/interface -I libraries/network/interface -I libraries/utils \
	-I $(PYTHON_INSTALL_DIR)/include/python3.9 -I libraries/network/interface \
	-I$(ML_PROJ_ROOT)/Input/include -I /usr/local/include/eigen3 \
	-I $(PYTHON_INSTALL_DIR)/lib/python3.9/site-packages/numpy/core/include

all: main

main : main.cpp
	$(CC) $(CFLAGS) $(PWD)/main.cpp -o NeuralNet $(INCLUDE_DIR) \
	-L $(PYTHON_INSTALL_DIR)/lib -lpython3.9 \
	-L $(PROJ_ROOT)/libraries/input/lib -ldata -L $(PROJ_ROOT)/libraries/network/lib -lnetwork

clean :
	rm -rf NeuralNet