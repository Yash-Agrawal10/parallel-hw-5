CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -Wshadow -O3
LDFLAGS = -fopenmp

SRC_DIR = ./src
OUT_DIR = ./bin

.phony: all clean

all: sequential openmp

clean:
	rm -f $(OUT_DIR)/*

sequential: $(SRC_DIR)/sequential.cpp
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $<

openmp: $(SRC_DIR)/openmp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(OUT_DIR)/$@ $<