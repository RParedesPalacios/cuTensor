
# Compiler options
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wwrite-strings
LIBS= -lcudart -lcuda -lcublas
NVCCFLAGS = 

# Directories
SRC_DIR = src
GPU_DIR = src/gpu
INCLUDE_DIR = include
INCLUDE_GPU_DIR = include/gpu
BUILD_DIR = build

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
GPU_FILES = $(wildcard $(GPU_DIR)/*.cu)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
GPU_OBJ_FILES = $(patsubst $(GPU_DIR)/%.cu, $(BUILD_DIR)/%.o, $(GPU_FILES))

# Include directories
INCLUDE_DIRS = -I$(INCLUDE_DIR) -I$(INCLUDE_GPU_DIR)

# Main program all the cpp en examples dirs
MAIN= $(wildcard examples/*/*.cpp)
EXEC= $(basename $(MAIN))

# Targets
all: $(EXEC)

$(EXEC): $(MAIN) $(OBJ_FILES) $(GPU_OBJ_FILES)
	$(CXX) $(CXXFLAGS)  $(INCLUDE_DIRS) $(MAIN) $(OBJ_FILES) $(GPU_OBJ_FILES) -o $(EXEC) $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(GPU_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

clean:
	rm -f $(OBJ_FILES) $(GPU_OBJ_FILES)
