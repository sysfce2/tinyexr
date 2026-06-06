# TinyEXR Makefile
#
#   make            - build the legacy v1 test executable (test_tinyexr)
#   make lib        - build the pure-C11 v3 library (build/libtinyexr3.a)
#   make test-c     - build + run the pure-C11 v3 reader unit test (ASan+UBSan)
#   make c11-gate   - compile every src/*.c as strict C11 -Werror (no C++)
#   make fuzz-corpus - replay regression corpus under ASan+UBSan+LSan
#   make fuzz-corpus-asan - same replay with LSan disabled for ptrace sandboxes
#   make clean

CC  ?= gcc
CXX ?= g++

CFLAGS   ?= -O2
CXXFLAGS ?= -O2 -std=c++11

INCLUDES = -I./deps/miniz
MINIZ_SRC = ./deps/miniz/miniz.c

# ---- legacy v1 single-header test (unchanged) -----------------------------
TARGET = test_tinyexr

.PHONY: all test clean help lib test-c c11-gate fuzz-corpus fuzz-corpus-asan parse-test

all: $(TARGET)

$(TARGET): test_tinyexr.cc miniz.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< miniz.o $(LDFLAGS)

miniz.o: $(MINIZ_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test: $(TARGET)
	./$(TARGET) asakusa.exr

# ---- pure-C11 v3 library + tests ------------------------------------------
V3_INC   = -Iinclude -Isrc -Ideps/zstd
V3_CSTD  = -std=c11
V3_WARN  = -Wall -Wextra -Werror
V3_SRC   = $(wildcard src/*.c)
V3_OBJ   = $(patsubst src/%.c,build/%.o,$(V3_SRC))
ZSTD_SRC = deps/zstd/tinyexr_zstd.c
ZSTD_OBJ = build/tinyexr_zstd.o
V3_TEST_OBJ = $(patsubst src/%.c,build/test-%.o,$(V3_SRC))
SAN      = -fsanitize=address,undefined

build:
	@mkdir -p build

build/%.o: src/%.c include/exr.h src/exr_internal.h deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_WARN) $(V3_INC) -O2 -g -c $< -o $@

build/tinyexr_zstd.o: $(ZSTD_SRC) deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_INC) -O2 -g -w -c $< -o $@

lib: $(V3_OBJ) $(ZSTD_OBJ)
	$(AR) rcs build/libtinyexr3.a $(V3_OBJ) $(ZSTD_OBJ)

# Strict pure-C11 gate: the rewrite must never require a C++ compiler.
c11-gate: | build
	@for f in $(V3_SRC); do \
	  echo "  C11  $$f"; \
	  $(CC) $(V3_CSTD) $(V3_WARN) $(V3_INC) -O1 -fsyntax-only $$f || exit 1; \
	done
	@echo "pure-C11 gate: OK"

build/test-%.o: src/%.c include/exr.h src/exr_internal.h deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) -c $< -o $@

build/test-tinyexr_zstd.o: $(ZSTD_SRC) deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_INC) -O1 -g $(SAN) -w -c $< -o $@

test-c: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o test/unit/test_exr_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) \
	  test/unit/test_exr_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o -lm -o build/test_exr_v3
	ASAN_OPTIONS=detect_leaks=0 ./build/test_exr_v3

bench: $(V3_OBJ) $(ZSTD_OBJ) benchmark/bench.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O3 \
	  benchmark/bench.c $(V3_OBJ) $(ZSTD_OBJ) -lm -o build/bench
	./build/bench

# Coverage-guided fuzzer (clang+libFuzzer over the whole library).
#   ./build/fuzz_v3 -max_total_time=60 test/unit/regression
fuzz: test/fuzzer/fuzz_v3.c | build
	clang $(V3_CSTD) $(V3_INC) -O1 -g -w -fsanitize=fuzzer,address,undefined \
	  test/fuzzer/fuzz_v3.c $(V3_SRC) $(ZSTD_SRC) -lm -o build/fuzz_v3
	@echo "built build/fuzz_v3 - e.g. ./build/fuzz_v3 -max_total_time=60 test/unit/regression"

# Deterministic corpus replay under ASan+UBSan (no libFuzzer needed; CI gate).
fuzz-corpus: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o test/fuzzer/fuzz_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) -DEXR_FUZZ_STANDALONE \
	  test/fuzzer/fuzz_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o -lm \
	  -o build/fuzz_replay
	./build/fuzz_replay test/unit/regression/* asakusa.exr deepscanline.exr

# Some local sandboxes/debug wrappers use ptrace. LeakSanitizer cannot run
# under ptrace, so this target preserves ASan+UBSan corpus coverage there while
# keeping fuzz-corpus as the strict LSan gate for CI/native hosts.
fuzz-corpus-asan: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o test/fuzzer/fuzz_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) -DEXR_FUZZ_STANDALONE \
	  test/fuzzer/fuzz_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o -lm \
	  -o build/fuzz_replay
	ASAN_OPTIONS=detect_leaks=0 ./build/fuzz_replay test/unit/regression/* asakusa.exr deepscanline.exr

# Parse/load every *.exr under $(EXR_IMAGES) and classify PASS/XFAIL/FAIL.
# Override the corpus dir with: make parse-test EXR_IMAGES=/path/to/images
EXR_IMAGES ?= $(HOME)/work/openexr-images
PARSE_HARNESS = test/v3/parse_harness

parse-test: lib
	$(CC) $(V3_CSTD) -Wall -Wextra -Iinclude -O2 \
	  test/v3/parse_harness.c build/libtinyexr3.a -lm -o $(PARSE_HARNESS)
	python3 test/v3/parse-tester.py --harness $(PARSE_HARNESS) $(EXR_IMAGES)

clean:
	rm -rf $(TARGET) miniz.o build $(PARSE_HARNESS)

help:
	@echo "make        - legacy v1 test (test_tinyexr)"
	@echo "make lib    - pure-C11 v3 library (build/libtinyexr3.a)"
	@echo "make test-c - run pure-C11 v3 reader unit test (ASan+UBSan)"
	@echo "make c11-gate - strict C11 -Werror compile of all src/*.c"
	@echo "make bench  - codec/SIMD throughput benchmark"
	@echo "make fuzz   - build libFuzzer target (build/fuzz_v3)"
	@echo "make fuzz-corpus - replay regression corpus under ASan+UBSan+LSan"
	@echo "make fuzz-corpus-asan - replay corpus with LSan disabled for ptrace sandboxes"
	@echo "make parse-test - parse/load openexr-images corpus, classify PASS/XFAIL/FAIL"
