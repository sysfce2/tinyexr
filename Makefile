# TinyEXR Makefile
#
#   make            - build the legacy v1 test executable (test_tinyexr)
#   make lib        - build the pure-C11 v3 library (build/libtinyexr3.a)
#   make test-c     - build + run the pure-C11 v3 reader unit test (ASan+UBSan)
#   make c11-gate   - compile every src/*.c as strict C11 -Werror (no C++)
#   make clean

CC  ?= gcc
CXX ?= g++

CFLAGS   ?= -O2
CXXFLAGS ?= -O2 -std=c++11

INCLUDES = -I./deps/miniz
MINIZ_SRC = ./deps/miniz/miniz.c

# ---- legacy v1 single-header test (unchanged) -----------------------------
TARGET = test_tinyexr

.PHONY: all test clean help lib test-c c11-gate

all: $(TARGET)

$(TARGET): test_tinyexr.cc miniz.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< miniz.o $(LDFLAGS)

miniz.o: $(MINIZ_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: $(TARGET)
	./$(TARGET) asakusa.exr

# ---- pure-C11 v3 library + tests ------------------------------------------
V3_INC   = -Iinclude -Isrc
V3_CSTD  = -std=c11
V3_WARN  = -Wall -Wextra -Werror
V3_SRC   = $(wildcard src/*.c)
V3_OBJ   = $(patsubst src/%.c,build/%.o,$(V3_SRC))
SAN      = -fsanitize=address,undefined

build:
	@mkdir -p build

build/%.o: src/%.c include/exr.h src/exr_internal.h | build
	$(CC) $(V3_CSTD) $(V3_WARN) $(V3_INC) -O2 -g -c $< -o $@

lib: $(V3_OBJ)
	$(AR) rcs build/libtinyexr3.a $(V3_OBJ)

# Strict pure-C11 gate: the rewrite must never require a C++ compiler.
c11-gate: | build
	@for f in $(V3_SRC); do \
	  echo "  C11  $$f"; \
	  $(CC) $(V3_CSTD) $(V3_WARN) $(V3_INC) -O1 -fsyntax-only $$f || exit 1; \
	done
	@echo "pure-C11 gate: OK"

test-c: $(V3_SRC) test/unit/test_exr_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) \
	  test/unit/test_exr_v3.c $(V3_SRC) -lm -o build/test_exr_v3
	./build/test_exr_v3

clean:
	rm -rf $(TARGET) miniz.o build

help:
	@echo "make        - legacy v1 test (test_tinyexr)"
	@echo "make lib    - pure-C11 v3 library (build/libtinyexr3.a)"
	@echo "make test-c - run pure-C11 v3 reader unit test (ASan+UBSan)"
	@echo "make c11-gate - strict C11 -Werror compile of all src/*.c"
