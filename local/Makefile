CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -pthread
LDFLAGS = -lcryptopp -lsnappy -pthread

# Source files
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Target
TARGET = main

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
