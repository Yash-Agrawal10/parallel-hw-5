CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -Wshadow -O3
LDFLAGS = -fopenmp

OUT_DIR = ./bin

.phony: all clean

all: sequential

clean:
	rm -f $(OUT_DIR)/*

sequential: sequential.cpp
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $<