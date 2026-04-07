# Makefile for tinyexr test executable

# Compilers
CC ?= gcc
CXX ?= g++

# Flags
CFLAGS ?= -O2
CXXFLAGS ?= -O2 -std=c++11
CXXFLAGS_V3 ?= -O2 -std=c++17

# Default to use miniz for decompression
BUILD_NANOZLIB ?= 0

# Include paths and defines
INCLUDES = -I./deps/miniz
INCLUDES_V3 = -I. -I./deps/miniz
DEFINES =

# V3 specific defines (compression now built-in via C11 headers)
DEFINES_V3 =
CFLAGS_V3 ?= -O2 -std=c11

# Optional: Address sanitizer (debug builds)
# CFLAGS += -fsanitize=address -g -O0
# CXXFLAGS += -fsanitize=address -Werror -Wall -Wextra -g -O0 -DTINYEXR_USE_MINIZ=0 -DTINYEXR_USE_PIZ=0

# Optional: ZFP compression (experimental)
# DEFINES += -DTINYEXR_USE_ZFP=1
# INCLUDES += -I./deps/ZFP/include
# LDFLAGS += -L./deps/ZFP/lib -lzfp

# Source files
TEST_SRC = test_tinyexr.cc
TEST_V3_SRC = test/unit/tester-v3.cc
C_IMPL_SRC = tinyexr_c_impl.c
MINIZ_SRC = ./deps/miniz/miniz.c

# Object files
ifeq ($(BUILD_NANOZLIB),1)
  DEFINES += -DTINYEXR_USE_NANOZLIB=1 -DTINYEXR_USE_MINIZ=0 -I./deps/nanozlib
  DEPOBJS =
  DEPOBJS_V3 =
else
  DEPOBJS = miniz.o
  DEPOBJS_V3 = miniz-v3.o tinyexr_c_impl.o
endif

# Target executables
TARGET = test_tinyexr
TARGET_V3 = test_tinyexr_v3

# Default target
.PHONY: all v3 test test-v3 clean help

all: $(TARGET)

v3: $(TARGET_V3)

# Build standard test executable
$(TARGET): $(TEST_SRC) $(DEPOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -o $@ $< $(DEPOBJS) $(LDFLAGS)

# Build v3 test executable
$(TARGET_V3): $(TEST_V3_SRC) $(DEPOBJS_V3)
	$(CXX) $(CXXFLAGS_V3) $(INCLUDES_V3) $(DEFINES) $(DEFINES_V3) -o $@ $< $(DEPOBJS_V3) $(LDFLAGS)

# Build miniz object for standard build
miniz.o: $(MINIZ_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Build miniz object for v3 build
miniz-v3.o: $(MINIZ_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Build C implementation as pure C11 for v3
tinyexr_c_impl.o: $(C_IMPL_SRC)
	$(CC) $(CFLAGS_V3) $(INCLUDES_V3) $(DEFINES_V3) -c -o $@ $<

# Run tests
test: $(TARGET)
	./$(TARGET) asakusa.exr

# Run v3 tests
test-v3: $(TARGET_V3)
	./$(TARGET_V3)

# Clean build artifacts
clean:
	rm -rf $(TARGET) $(TARGET_V3) miniz.o miniz-v3.o tinyexr_c_impl.o

# Print help
help:
	@echo "tinyexr Makefile targets:"
	@echo "  make          - Build test_tinyexr executable"
	@echo "  make v3       - Build test_tinyexr_v3 executable (C++17)"
	@echo "  make test     - Run tests with asakusa.exr"
	@echo "  make test-v3  - Run v3 tests"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  BUILD_NANOZLIB=1  - Use nanozlib instead of miniz"
