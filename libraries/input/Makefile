CC=g++
INCLUDE_DIR := -I interface -I $(PYTHON_INSTALL_DIR)/include/python3.9 \
	-I $(PYTHON_INSTALL_DIR)/lib/python3.9/site-packages/numpy/core/include
SRC := src
CFLAGS := -std=c++11 -g
LIB_DATA := libdata.so

spiral: spiral_input
mnist: mnist_input

spiral_input : mkdir spiral_data_handler.o
	$(CC) $(CFLAGS) -shared -o lib/$(LIB_DATA) obj/data_handler.o

mnist_input : mkdir data.o mnist_data_handler.o
	$(CC) $(CFLAGS) -shared -o lib/$(LIB_DATA) obj/data_handler.o

mkdir :
	if [ ! -d "obj" ]; then mkdir obj; fi
	if [ ! -d "lib" ]; then mkdir lib; fi

spiral_data_handler.o: $(SRC)/spiral/SpiralDataHandler.cpp
	$(CC) -fPIC $(CFLAGS) -o obj/data_handler.o $(INCLUDE_DIR) \
	-c $(SRC)/spiral/SpiralDataHandler.cpp

mnist_data_handler.o: $(SRC)/mnist/DataHandler.cpp
	$(CC) -fPIC $(CFLAGS) -o obj/data_handler.o $(INCLUDE_DIR) \
	-c $(SRC)/mnist/DataHandler.cpp

clean:
	rm -r lib obj
