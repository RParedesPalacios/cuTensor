
# Compiler options
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 
LIBS= -lcudart -lcuda 
NVCCFLAGS = 

# Directories
SRC_DIR = src
GPU_DIR = src/gpu
INCLUDE_DIR = include
INCLUDE_GPU_DIR = include/gpu

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
GPU_FILES = $(wildcard $(GPU_DIR)/*.cu)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(SRC_DIR)/%.o, $(SRC_FILES))
GPU_OBJ_FILES = $(patsubst $(GPU_DIR)/%.cu, $(GPU_DIR)/%.o, $(GPU_FILES))

# Include directories
INCLUDE_DIRS = -I$(INCLUDE_DIR) -I$(INCLUDE_GPU_DIR)

# Main program
MAIN = run.cpp
EXEC = run

# Targets
all: $(EXEC)

$(EXEC): $(MAIN) $(OBJ_FILES) $(GPU_OBJ_FILES)
	$(CXX) $(CXXFLAGS)  $(INCLUDE_DIRS) $(MAIN) $(OBJ_FILES) $(GPU_OBJ_FILES) -o $(EXEC) $(LIBS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(INCLUDE_DIRS) -c -o $@ $<

$(GPU_DIR)/%.o: $(GPU_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

clean:
	rm -f $(OBJ_FILES) $(GPU_OBJ_FILES)
