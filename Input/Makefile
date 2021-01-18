CC=g++
INCLUDE_DIR := -I$(ML_PROJ_ROOT)/Input/$(INPUT_TYPE)/include  -I$(ML_PROJ_ROOT)/Input \
	-I $(ML_PROJ_ROOT)/Network/include -I $(PYTHON_INSTALL_DIR)/include/python3.8 \
	-I $(PYTHON_INSTALL_DIR)/lib/python3.8/site-packages/numpy/core/include
SRC := $(ML_PROJ_ROOT)/Input
CFLAGS := -std=c++11 -g
LIB_DATA := libdata.so

all : $(LIB_DATA)

$(LIB_DATA) : libdir objdir obj/data.o obj/data_handler.o
	$(CC) $(CFLAGS) -shared -o $(ML_PROJ_ROOT)/Input/lib/$(LIB_DATA) obj/data.o obj/data_handler.o

libdir :
	mkdir -p $(ML_PROJ_ROOT)/Input/lib

objdir:
	mkdir -p $(ML_PROJ_ROOT)/Input/obj

obj/data.o: $(SRC)/Data.cpp
	$(CC) -fPIC $(CFLAGS)  -o obj/data.o $(INCLUDE_DIR) -c $(SRC)/Data.cpp

obj/data_handler.o: $(SRC)/$(INPUT_TYPE)/src/DataHandler.cpp
	$(CC) -fPIC $(CFLAGS) -o obj/data_handler.o $(INCLUDE_DIR) -c $(SRC)/$(INPUT_TYPE)/src/DataHandler.cpp

clean:
	rm -r $(ML_PROJ_ROOT)/Input/lib
	rm -r $(ML_PROJ_ROOT)/Input/obj
