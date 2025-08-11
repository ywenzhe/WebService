# Makefile for WebServiceBenchmark

# Compiler to use
CXX = g++

# Compiler flags
# -O3: Aggressive optimization for performance
# -DNDEBUG: Disables debugging macros (like assert()) for release builds
# -std=c++17: Specifies the C++ standard to use (C++17)
# -Wall -Wextra: Enables common and extra warning messages for better code quality
CXXFLAGS = -O3 -DNDEBUG -std=c++17 -Wall -Wextra

# Linker flags (e.g., for specifying library search paths if non-standard)
LDFLAGS =

# Libraries to link with
# -lsnappy: Links against the Snappy compression library
# -lcryptopp: Links against the Crypto++ library
# -lpthread: Links against the POSIX threads library (needed for std::thread)
LIBS = -lsnappy -lcryptopp -lpthread

# Source files (.cpp files)
SRCS = main.cpp

# Object files (.o files) generated from source files
OBJS = $(SRCS:.cpp=.o)

# Name of the final executable
TARGET = main

# .PHONY declares targets that are not real files, ensuring they always run
.PHONY: all clean

# Default target: builds the executable
all: $(TARGET)

# Rule to build the executable from object files
# $@ is the target name (main)
# $^ is all prerequisites (main.o)
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

# Rule to compile each .cpp file into a .o object file
# $< is the first prerequisite (e.g., main.cpp)
# $@ is the target name (e.g., main.o)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target: removes generated object files and the executable
clean:
	@echo "Cleaning up..."
	@rm -f $(OBJS) $(TARGET)
	@echo "Clean complete."

# Note about header files (zipf.hpp):
# zipf.hpp is a header-only file and is included directly by main.cpp.
# It does not need to be explicitly listed as a source file
# or compiled independently by the Makefile.
