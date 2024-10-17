# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_87

# Linker flags
LDFLAGS := -lm -lcublas

# Build directory
BUILD_DIR := build

# Source files
SRCS := $(wildcard *.cu)

# Executables (one for each source file)
EXECS := $(SRCS:%.cu=$(BUILD_DIR)/%)

# Default target
all: $(BUILD_DIR) $(EXECS)

# Rule to create build directory
$(BUILD_DIR):
	mkdir -p $@

# Rule to create executables
$(BUILD_DIR)/%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
