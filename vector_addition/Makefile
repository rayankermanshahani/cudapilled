# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_70

# Build directory
BUILD_DIR := build

# Source files
SRCS := $(wildcard *.cu)

# Object files
OBJS := $(SRCS:%.cu=$(BUILD_DIR)/%.o)

# Executable name
EXEC := $(BUILD_DIR)/cuda_program

# Default target
all: $(BUILD_DIR) $(EXEC)

# Rule to create build directory
$(BUILD_DIR):
	mkdir -p $@

# Rule to create executable
$(EXEC): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Rule to compile source files
$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
