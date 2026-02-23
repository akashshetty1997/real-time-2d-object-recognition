# Project 3: Real-time 2D Object Recognition
# Name: Akash Shridhar Shetty , Skandhan Madhusudhana
# Date: February 2025
# CS 5330 - Pattern Recognition & Computer Vision
#
# macOS M4 with OpenCV 4 via Homebrew
# Build system: Makefile with pkg-config

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
INCLUDES = -Iinclude $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

SRC_DIR = src
OBJ_DIR = obj
BIN = objrec

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

.PHONY: all clean dirs

all: dirs $(BIN)

dirs:
	@mkdir -p $(OBJ_DIR) data models data/task8_embeddings

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)