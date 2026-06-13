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
EMCC ?= emcc

CFLAGS   ?= -O2
CXXFLAGS ?= -O2 -std=c++11

INCLUDES = -I./deps/miniz
MINIZ_SRC = ./deps/miniz/miniz.c

# ---- legacy v1 single-header test (unchanged) -----------------------------
TARGET = test_tinyexr

.PHONY: all test clean help lib test-c test-c-threads test-c-tsan c11-gate fuzz-corpus fuzz-corpus-asan parse-test wasm freestanding-gate examples-c bench bench-compare arm-smoke host-smoke

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
V3_DEFS  =
V3_SRC   = $(wildcard src/*.c)
V3_OBJ   = $(patsubst src/%.c,build/%.o,$(V3_SRC))
# Freestanding core: everything except the optional stdio layer, the spectral
# helpers (hosted-only convenience), and the (freestanding-only) mem/str impls.
V3_CORE_SRC = $(filter-out src/exr_stdio.c src/exr_freestanding.c src/exr_spectral.c,$(V3_SRC))
ZSTD_SRC = deps/zstd/tinyexr_zstd.c
ZSTD_OBJ = build/tinyexr_zstd.o
V3_TEST_OBJ = $(patsubst src/%.c,build/test-%.o,$(V3_SRC))
SAN      = -fsanitize=address,undefined

# ---- optional libdeflate backend (default OFF; in-tree codec is the default)
# Build any target with LIBDEFLATE=1 to route ZIP/ZIPS/PXR24 deflate through the
# vendored libdeflate (deps/libdeflate, MIT - see deps/libdeflate/COPYING).
# NOTE: run `make clean` when toggling LIBDEFLATE (object flags are not tracked).
LIBDEFLATE ?= 0
LD_OBJ =
LD_TEST_OBJ =
ifeq ($(LIBDEFLATE),1)
  V3_DEFS += -DEXR_USE_LIBDEFLATE
  V3_INC  += -Ideps/libdeflate
  # Just the zlib (DEFLATE) path: no crc32/gzip. The x86/arm cpu_features files
  # self-guard by arch, so compiling both is safe everywhere.
  LD_SRC = deps/libdeflate/lib/adler32.c \
           deps/libdeflate/lib/deflate_compress.c \
           deps/libdeflate/lib/deflate_decompress.c \
           deps/libdeflate/lib/zlib_compress.c \
           deps/libdeflate/lib/zlib_decompress.c \
           deps/libdeflate/lib/utils.c \
           deps/libdeflate/lib/x86/cpu_features.c \
           deps/libdeflate/lib/arm/cpu_features.c
  LD_OBJ      = $(patsubst deps/libdeflate/%.c,build/libdeflate/%.o,$(LD_SRC))
  LD_TEST_OBJ = $(patsubst deps/libdeflate/%.c,build/test-libdeflate/%.o,$(LD_SRC))
endif

# ---- optional C11-threads multithreading (default OFF; serial is the default)
# Build any target with THREADS=1 to enable per-block parallel encode/decode
# (src/exr_thread.c, uses <threads.h>). Default and freestanding builds stay
# single-threaded. NOTE: run `make clean` when toggling THREADS.
THREADS ?= 0
THREAD_LIBS =
ifeq ($(THREADS),1)
  V3_DEFS    += -DEXR_USE_THREADS
  THREAD_LIBS = -pthread          # C11 threads need pthreads on glibc < 2.34
endif

build:
	@mkdir -p build

build/%.o: src/%.c include/exr.h src/exr_internal.h deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_WARN) $(V3_DEFS) $(V3_INC) -O2 -g -c $< -o $@

build/tinyexr_zstd.o: $(ZSTD_SRC) deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_INC) -O2 -g -w -c $< -o $@

# Vendored libdeflate (third-party: warnings off). Release + sanitized variants.
build/libdeflate/%.o: deps/libdeflate/%.c | build
	@mkdir -p $(dir $@)
	$(CC) -Ideps/libdeflate -O3 -g -w -c $< -o $@

build/test-libdeflate/%.o: deps/libdeflate/%.c | build
	@mkdir -p $(dir $@)
	$(CC) -Ideps/libdeflate -O1 -g $(SAN) -w -c $< -o $@

lib: $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ)
	$(AR) rcs build/libtinyexr3.a $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ)

# Strict pure-C11 gate: the rewrite must never require a C++ compiler.
c11-gate: | build
	@for f in $(V3_SRC); do \
	  echo "  C11  $$f"; \
	  $(CC) $(V3_CSTD) $(V3_WARN) $(V3_INC) -O1 -fsyntax-only $$f || exit 1; \
	done
	@echo "pure-C11 gate: OK"

build/test-%.o: src/%.c include/exr.h src/exr_internal.h deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_DEFS) $(V3_INC) -O1 -g $(SAN) -c $< -o $@

build/test-tinyexr_zstd.o: $(ZSTD_SRC) deps/zstd/tinyexr_zstd.h | build
	$(CC) $(V3_CSTD) $(V3_INC) -O1 -g $(SAN) -w -c $< -o $@

test-c: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) test/unit/test_exr_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_DEFS) $(V3_INC) -O1 -g $(SAN) \
	  test/unit/test_exr_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) $(THREAD_LIBS) -lm -o build/test_exr_v3
	ASAN_OPTIONS=detect_leaks=0 ./build/test_exr_v3

# Build + run the unit tests with multithreading enabled (parity + race checks).
test-c-threads:
	$(MAKE) test-c THREADS=1

# Thread sanitizer over the threaded build (no ASan; proves no data races).
# NOTE: requires a ThreadSanitizer that instruments C11 <threads.h>; some
# glibc/TSan combos only intercept pthread_* and crash on thrd_create/mtx_*.
# The threaded build also runs cleanly under ASan+UBSan via `make test-c-threads`.
test-c-tsan: | build
	$(CC) $(V3_CSTD) -Wall -Wextra -DEXR_USE_THREADS $(V3_INC) -O1 -g \
	  -fsanitize=thread test/unit/test_exr_v3.c $(V3_SRC) $(ZSTD_SRC) \
	  -pthread -lm -o build/test_exr_v3_tsan
	./build/test_exr_v3_tsan

bench: $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ) benchmark/bench.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_DEFS) $(V3_INC) -O3 \
	  benchmark/bench.c $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ) $(THREAD_LIBS) -lm -o build/bench
	./build/bench

# ---- tinyexr-vs-OpenEXR comparison (needs a built OpenEXR) -----------------
# Override OPENEXR_ROOT / OPENEXR_BUILD if your tree lives elsewhere. Extra
# files: make bench-compare ARGS="img1.exr img2.exr"
OPENEXR_ROOT  ?= $(HOME)/work/openexr
OPENEXR_BUILD ?= $(OPENEXR_ROOT)/_build
OPENEXR_INC    = -I$(OPENEXR_ROOT)/src/lib/OpenEXR -I$(OPENEXR_ROOT)/src/lib/OpenEXRCore \
                 -I$(OPENEXR_ROOT)/src/lib/Iex -I$(OPENEXR_ROOT)/src/lib/IlmThread \
                 -I$(OPENEXR_BUILD)/cmake $(shell pkg-config --cflags Imath 2>/dev/null)
OPENEXR_LIBDIR = $(OPENEXR_BUILD)/src/lib
OPENEXR_LIBS   = -L$(OPENEXR_LIBDIR)/OpenEXR -L$(OPENEXR_LIBDIR)/OpenEXRCore \
                 -L$(OPENEXR_LIBDIR)/Iex -L$(OPENEXR_LIBDIR)/IlmThread \
                 -lOpenEXR-4_0 -lOpenEXRCore-4_0 -lIex-4_0 -lIlmThread-4_0 -pthread
OPENEXR_LDPATH = $(OPENEXR_LIBDIR)/OpenEXR:$(OPENEXR_LIBDIR)/OpenEXRCore:$(OPENEXR_LIBDIR)/Iex:$(OPENEXR_LIBDIR)/IlmThread

# The tinyexr side (bench_tx.c) is compiled as C because exr.h and OpenEXR's
# C core declare the same global enum names and cannot share a translation unit.
build/bench_tx.o: benchmark/bench_tx.c benchmark/bench_tx.h include/exr.h | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O3 -c benchmark/bench_tx.c -o $@

bench-compare: $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ) build/bench_tx.o benchmark/bench_compare.cpp | build
	@test -d $(OPENEXR_LIBDIR)/OpenEXR || { \
	  echo "OpenEXR build not found at $(OPENEXR_BUILD)"; \
	  echo "build OpenEXR first, or set OPENEXR_ROOT=/path/to/openexr"; exit 1; }
	$(CXX) -std=c++14 -Wall -Ibenchmark $(OPENEXR_INC) -O3 \
	  benchmark/bench_compare.cpp build/bench_tx.o $(V3_OBJ) $(ZSTD_OBJ) $(LD_OBJ) \
	  $(OPENEXR_LIBS) $(THREAD_LIBS) -lm -o build/bench_compare
	LD_LIBRARY_PATH=$(OPENEXR_LDPATH) ./build/bench_compare $(ARGS)

# Coverage-guided fuzzer (clang+libFuzzer over the whole library).
#   ./build/fuzz_v3 -max_total_time=60 test/unit/regression
fuzz: test/fuzzer/fuzz_v3.c | build
	clang $(V3_CSTD) $(V3_INC) -O1 -g -w -fsanitize=fuzzer,address,undefined \
	  test/fuzzer/fuzz_v3.c $(V3_SRC) $(ZSTD_SRC) -lm -o build/fuzz_v3
	@echo "built build/fuzz_v3 - e.g. ./build/fuzz_v3 -max_total_time=60 test/unit/regression"

# HTJ2K (JPH) encode+decode+round-trip fuzzer.
#   ./build/fuzz_jph -max_total_time=600 test/fuzzer/corpus_jph
fuzz-jph: test/fuzzer/fuzz_jph.c | build
	clang $(V3_CSTD) $(V3_INC) -O1 -g -w -fsanitize=fuzzer,address,undefined \
	  test/fuzzer/fuzz_jph.c $(V3_SRC) $(ZSTD_SRC) -lm -o build/fuzz_jph
	@echo "built build/fuzz_jph"

# Deterministic corpus replay for fuzz_jph (no libFuzzer needed).
fuzz-jph-corpus: test/fuzzer/fuzz_jph.c | build
	clang $(V3_CSTD) -Wall $(V3_INC) -O1 -g $(SAN) -DEXR_JPH_FUZZ_STANDALONE \
	  test/fuzzer/fuzz_jph.c $(V3_SRC) $(ZSTD_SRC) -lm -o build/fuzz_jph_replay

# Deterministic corpus replay under ASan+UBSan (no libFuzzer needed; CI gate).
fuzz-corpus: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) test/fuzzer/fuzz_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) -DEXR_FUZZ_STANDALONE \
	  test/fuzzer/fuzz_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) -lm \
	  -o build/fuzz_replay
	./build/fuzz_replay test/unit/regression/* asakusa.exr deepscanline.exr

# Some local sandboxes/debug wrappers use ptrace. LeakSanitizer cannot run
# under ptrace, so this target preserves ASan+UBSan corpus coverage there while
# keeping fuzz-corpus as the strict LSan gate for CI/native hosts.
fuzz-corpus-asan: $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) test/fuzzer/fuzz_v3.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(V3_INC) -O1 -g $(SAN) -DEXR_FUZZ_STANDALONE \
	  test/fuzzer/fuzz_v3.c $(V3_TEST_OBJ) build/test-tinyexr_zstd.o $(LD_TEST_OBJ) -lm \
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

# ---- native example (uses the optional stdio layer) -----------------------
examples-c: lib
	$(CC) $(V3_CSTD) -Wall -Wextra -Iinclude -O2 \
	  examples/exrinfo/exrinfo.c build/libtinyexr3.a -o build/exrinfo
	@echo "built build/exrinfo - e.g. ./build/exrinfo asakusa.exr"

# ---- Emscripten WASM build of the v3 C API --------------------------------
# Pure-C exports, no filesystem. Produces an ES6 module + wasm. Needs emcc on
# PATH (override with EMCC=...). The freestanding core links cleanly with no FS.
WASM_EXPORTS = ['_exrw_decode_rgba','_exrw_encode_rgba','_exrw_free','_malloc','_free']
WASM_RUNTIME = ['HEAPU8','HEAPF32','HEAP32','HEAPU32']
wasm: | build
	$(EMCC) -O3 $(V3_INC) -w \
	  $(V3_CORE_SRC) $(ZSTD_SRC) examples/wasm/exr_wasm.c \
	  -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 \
	  -s EXPORT_ES6=1 -s ENVIRONMENT=web,node \
	  -s "EXPORTED_FUNCTIONS=$(WASM_EXPORTS)" \
	  -s "EXPORTED_RUNTIME_METHODS=$(WASM_RUNTIME)" \
	  -o build/exr_v3.mjs
	@echo "built build/exr_v3.mjs + build/exr_v3.wasm"

# ---- ARM (aarch64 NEON) cross build + emulated run ------------------------
# Cross-compile the SIMD smoke test for aarch64 and run it under an emulator to
# confirm the NEON path builds and is bit-identical to scalar. Static link so
# the emulator needs no guest sysroot. Override the toolchain/emulator as needed:
#   make arm-smoke ARM_CC=aarch64-linux-gnu-gcc ARM_QEMU=qemu-aarch64
ARM_CC   ?= aarch64-linux-gnu-gcc-13
ARM_QEMU ?= qemu-aarch64-static
ARM_SMOKE_SRC = $(filter-out src/exr_zstd.c,$(V3_SRC))

arm-smoke: test/v3/neon_smoke.c | build
	$(ARM_CC) -static -march=armv8-a $(V3_CSTD) -Wall -Wextra -DEXR_NO_ZSTD \
	  $(V3_INC) -O2 test/v3/neon_smoke.c $(ARM_SMOKE_SRC) -lm \
	  -o build/neon_smoke_arm
	@file build/neon_smoke_arm | sed 's/^/  /'
	$(ARM_QEMU) ./build/neon_smoke_arm

# Same smoke, built/run natively (sanity-checks the host SIMD tier).
host-smoke: test/v3/neon_smoke.c | build
	$(CC) $(V3_CSTD) -Wall -Wextra -DEXR_NO_ZSTD $(V3_INC) -O2 \
	  test/v3/neon_smoke.c $(ARM_SMOKE_SRC) -lm -o build/neon_smoke_host
	./build/neon_smoke_host

# ---- freestanding gate ----------------------------------------------------
# Compile the core with only stdint/stddef/limits, prove there are no forbidden
# libc dependencies (nm scan), and run a functional memory round-trip.
FS_FLAGS = -DEXR_FREESTANDING -DEXR_NO_ZSTD -ffreestanding -fno-builtin \
           -fno-stack-protector $(V3_CSTD) $(V3_WARN) $(V3_INC) -O2 -g
# Freestanding excludes the zstd glue (EXR_NO_ZSTD); the codec dispatch returns
# UNSUPPORTED, so the vendored allocator/amalgamation is never pulled in.
FS_CORE_SRC = $(filter-out src/exr_zstd.c,$(V3_CORE_SRC))
FS_OBJ = $(patsubst src/%.c,build/fs-%.o,$(FS_CORE_SRC)) build/fs-exr_freestanding.o
FS_FORBIDDEN = fopen|fread|fwrite|fseek|ftell|fclose|fprintf|printf|snprintf|malloc|calloc|realloc|free|abort|exit|qsort|exp|log|pow

build/fs-%.o: src/%.c include/exr.h src/exr_internal.h | build
	$(CC) $(FS_FLAGS) -c $< -o $@

freestanding-gate: $(FS_OBJ) test/v3/freestanding_smoke.c | build
	@echo "  scan: only exr_stdio.c may include <stdio.h>"
	@bad=`grep -rl '<stdio.h>' src/ | grep -v 'src/exr_stdio.c' || true`; \
	  if [ -n "$$bad" ]; then echo "  FAIL: stdio leaked into: $$bad"; exit 1; fi
	@echo "  scan: no forbidden libc symbols referenced by the freestanding core"
	@for o in $(FS_OBJ); do \
	  hit=`nm -u $$o 2>/dev/null | awk '{print $$NF}' | grep -wE '$(FS_FORBIDDEN)' || true`; \
	  if [ -n "$$hit" ]; then echo "  FAIL: $$o references:" $$hit; exit 1; fi; \
	done
	@echo "  run: freestanding-compiled core + custom-allocator round-trip"
	$(CC) $(V3_CSTD) -Wall -Wextra -Iinclude -Isrc -O2 \
	  test/v3/freestanding_smoke.c $(FS_OBJ) -o build/fs_smoke
	./build/fs_smoke
	@echo "freestanding gate: OK"

# ---- tocio (sandbox: tiny OpenColorIO config engine + codegen) ------------
# Pure-C11, freestanding, no external deps. Lives outside src/ so it has its own
# build/gate targets. Mirrors the v3 freestanding discipline.
TOC_INC      = -Isandbox/tocio/include -Isandbox/tocio/src
TOC_SRC      = $(wildcard sandbox/tocio/src/*.c)
TOC_OBJ      = $(patsubst sandbox/tocio/src/%.c,build/toc-%.o,$(TOC_SRC))
TOC_HDRS     = sandbox/tocio/include/tocio.h sandbox/tocio/src/toc_internal.h
# Freestanding core: everything except the optional hosted stdio loader and the
# hosted JIT (needs OS executable memory).
TOC_CORE_SRC = $(filter-out sandbox/tocio/src/toc_stdio.c sandbox/tocio/src/toc_jit.c,$(TOC_SRC))
TOC_FS_OBJ   = $(patsubst sandbox/tocio/src/%.c,build/toc-fs-%.o,$(TOC_CORE_SRC))
TOC_FS_FORBIDDEN = $(FS_FORBIDDEN)

.PHONY: tocio-lib tocio-c11-gate tocio-freestanding-gate tocio-test

build/toc-%.o: sandbox/tocio/src/%.c $(TOC_HDRS) | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O2 -g $(SAN) -c $< -o $@

tocio-lib: $(TOC_OBJ)
	$(AR) rcs build/libtocio.a $(TOC_OBJ)
	@echo "built build/libtocio.a"

tocio-c11-gate: | build
	@for f in $(TOC_SRC); do \
	  echo "  C11  $$f"; \
	  $(CC) $(V3_CSTD) $(V3_WARN) $(TOC_INC) -O1 -fsyntax-only $$f || exit 1; \
	done
	@echo "tocio pure-C11 gate: OK"

build/toc-fs-%.o: sandbox/tocio/src/%.c $(TOC_HDRS) | build
	$(CC) -DTOC_FREESTANDING -ffreestanding -fno-builtin -fno-stack-protector \
	  $(V3_CSTD) $(V3_WARN) $(TOC_INC) -O2 -g -c $< -o $@

tocio-freestanding-gate: $(TOC_FS_OBJ) sandbox/tocio/tests/toc_fs_smoke.c | build
	@echo "  scan: only toc_stdio.c may include <stdio.h>"
	@bad=`grep -rl '<stdio.h>' sandbox/tocio/src/ | grep -v 'toc_stdio.c' || true`; \
	  if [ -n "$$bad" ]; then echo "  FAIL: stdio leaked into: $$bad"; exit 1; fi
	@echo "  scan: no forbidden libc symbols in the freestanding core"
	@for o in $(TOC_FS_OBJ); do \
	  hit=`nm -u $$o 2>/dev/null | awk '{print $$NF}' | grep -wE '$(TOC_FS_FORBIDDEN)' || true`; \
	  if [ -n "$$hit" ]; then echo "  FAIL: $$o references:" $$hit; exit 1; fi; \
	done
	@echo "  run: freestanding-compiled core + custom-allocator round-trip"
	$(CC) $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O2 \
	  sandbox/tocio/tests/toc_fs_smoke.c $(TOC_FS_OBJ) -o build/toc_fs_smoke
	./build/toc_fs_smoke
	@echo "tocio freestanding gate: OK"

tocio-test: | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O1 -g $(SAN) \
	  sandbox/tocio/tests/toc_test.c $(TOC_SRC) -lm -ldl -o build/toc_test
	ASAN_OPTIONS=detect_leaks=0 ./build/toc_test

# Validate tocio against the REAL AcademySoftwareFoundation ACES OCIO configs:
# parse each config, build every transform, and compare results to golden values
# captured from PyOpenColorIO (the C++ reference engine). The configs live under
# sandbox/tocio/ref (fetch them first); the golden TSV is committed.
#   make tocio-fetch-ref    # download configs + reference repos into ref/
#   make tocio-gen-golden   # regenerate the golden TSV (needs PyOpenColorIO)
TOC_REFDIR    = sandbox/tocio/ref/configs
TOC_GOLDEN    = sandbox/tocio/tests/golden/aces_golden.tsv
.PHONY: tocio-validate tocio-fetch-ref tocio-gen-golden
tocio-validate: | build
	@if [ ! -d "$(TOC_REFDIR)" ]; then \
	  echo "ref configs missing - run 'make tocio-fetch-ref' first"; exit 2; fi
	$(CC) $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O1 -g $(SAN) \
	  sandbox/tocio/tests/toc_validate.c $(TOC_SRC) -lm -ldl -o build/toc_validate
	ASAN_OPTIONS=detect_leaks=0 ./build/toc_validate $(TOC_GOLDEN) $(TOC_REFDIR)

tocio-fetch-ref:
	bash sandbox/tocio/scripts/fetch_ocio_ref.sh

tocio-gen-golden:
	@sp=$$(echo sandbox/tocio/ref/.pyoracle/lib/python*/site-packages); \
	  PYTHONPATH="$$sp" python3 sandbox/tocio/scripts/gen_golden.py

# Interpreter throughput benchmark (scalar vs SIMD per op). -O2, no sanitizers.
.PHONY: tocio-bench
tocio-bench: | build
	$(CC) $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O2 \
	  sandbox/tocio/tests/toc_bench.c $(TOC_SRC) -lm -ldl -o build/toc_bench
	./build/toc_bench

# ---- tocio ARM64 (aarch64 NEON) cross build + qemu test ---------------------
# Cross-compile the full tocio test suite and run under qemu-aarch64.
# toc_jit.c has a native AArch64/NEON backend (emits A64 code to executable
# memory); under qemu-user the JIT may report TOC_ERROR_UNSUPPORTED and skip.
.PHONY: tocio-arm-test
tocio-arm-test: | build
	$(ARM_CC) -static -march=armv8-a $(V3_CSTD) -Wall -Wextra $(TOC_INC) -O2 \
	  sandbox/tocio/tests/toc_test.c $(TOC_SRC) -lm -o build/toc_test_arm
	@file build/toc_test_arm | sed 's/^/  /'
	$(ARM_QEMU) ./build/toc_test_arm

# ---- tocio WASM (Emscripten ES6 module for the web viewer) -----------------
TOCW_EXPORTS = ['_tocw_parse','_tocw_free_config','_tocw_processor','_tocw_processor_view','_tocw_free_ops','_tocw_apply','_tocw_emit_glsl','_tocw_emit_metal','_tocw_emit_c','_tocw_free_str','_tocw_num_colorspaces','_tocw_colorspace_name','_malloc','_free']
TOCW_RUNTIME = ['HEAPU8','HEAPF32','HEAP32','UTF8ToString','stringToUTF8','lengthBytesUTF8']
.PHONY: wasm-tocio wasm-tocio-test
wasm-tocio: | build
	$(EMCC) -O3 $(TOC_INC) -w \
	  $(TOC_CORE_SRC) sandbox/tocio/wasm/toc_wasm.c \
	  -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 \
	  -s EXPORT_ES6=1 -s ENVIRONMENT=web,node \
	  -s "EXPORTED_FUNCTIONS=$(TOCW_EXPORTS)" \
	  -s "EXPORTED_RUNTIME_METHODS=$(TOCW_RUNTIME)" \
	  -o build/tocio.mjs
	@echo "built build/tocio.mjs + build/tocio.wasm"

wasm-tocio-test: wasm-tocio
	node sandbox/tocio/wasm/test.mjs

clean:
	rm -rf $(TARGET) miniz.o build $(PARSE_HARNESS)

help:
	@echo "make        - legacy v1 test (test_tinyexr)"
	@echo "make lib    - pure-C11 v3 library (build/libtinyexr3.a)"
	@echo "make test-c - run pure-C11 v3 reader unit test (ASan+UBSan)"
	@echo "make test-c-threads - unit tests with multithreading (THREADS=1)"
	@echo "make test-c-tsan - threaded unit tests under ThreadSanitizer"
	@echo "make c11-gate - strict C11 -Werror compile of all src/*.c"
	@echo "make bench  - codec/SIMD throughput benchmark (incl. HTJ2K SIMD tiers)"
	@echo "make bench-compare - tinyexr-vs-OpenEXR codec comparison (needs OpenEXR build)"
	@echo "make fuzz   - build libFuzzer target (build/fuzz_v3)"
	@echo "make fuzz-corpus - replay regression corpus under ASan+UBSan+LSan"
	@echo "make fuzz-corpus-asan - replay corpus with LSan disabled for ptrace sandboxes"
	@echo "make parse-test - parse/load openexr-images corpus, classify PASS/XFAIL/FAIL"
	@echo "make examples-c - build native v3 example (build/exrinfo)"
	@echo "make wasm    - Emscripten WASM build of the v3 C API (build/exr_v3.mjs)"
	@echo "make freestanding-gate - prove the core builds with no libc (stdint/stddef only)"
	@echo "make arm-smoke - cross-build (aarch64) + run NEON SIMD smoke under qemu"
	@echo "make host-smoke - build + run the SIMD smoke test natively"
	@echo ""
	@echo "Add LIBDEFLATE=1 to any target to use the optional vendored libdeflate"
	@echo "  backend for ZIP/ZIPS/PXR24 (default: in-tree codec). Run 'make clean'"
	@echo "  when toggling. e.g. make bench-compare LIBDEFLATE=1"
	@echo "Add THREADS=1 to any target for C11-threads parallel encode/decode"
	@echo "  (default: single-threaded). Set count via exr_set_num_threads()."
