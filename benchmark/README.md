# TinyEXR v3 benchmark

Standalone micro/throughput benchmark for the pure-C11 EXR codec.

```sh
make bench                 # build + run on asakusa.exr
./build/bench foo.exr bar.exr   # run on your own files
```

It reports:

1. **Codec throughput** — for each codec the writer supports
   (NONE/RLE/ZIPS/ZIP/PIZ/HTJ2K32/HTJ2K256), the encode and decode rate in
   megapixels/second and the compressed size. Decode covers all codecs in the
   file; the auto-generated set only exercises the writable ones (pass
   PXR24/B44/DWA files on the command line to time those decoders too).

2. **ZIP decode by forced SIMD tier** — the same ZIP decode run with the
   predictor + byte de-interleave kernels forced to scalar / SSE2 / AVX2 (via
   `exr_simd_force`), showing the end-to-end effect of that dispatch.

3. **HTJ2K256 by forced SIMD tier** — end-to-end HTJ2K encode *and* decode at
   scalar / SSE4.1 / AVX2. The JPH (HTJ2K) codec dispatches its SIMD kernels
   (NLT type-3, half pack, 5/3 inverse) directly off the cached CPU caps, so
   this section uses `exr_cpu_caps_force` (a benchmark-only override, mirroring
   `exr_simd_force`) to exercise the whole HTJ2K pipeline per tier.

4. **SIMD kernels** — direct throughput (GB/s) and speedup for the byte
   de-interleave (scalar / SSE2 / AVX2) and half→float (scalar / F16C)
   primitives, plus the JPH `nlt3` / `pack` micro-kernels (M elem/s per tier).

## OpenEXR comparison

```sh
make bench-compare                          # asakusa.exr, vs a built OpenEXR
make bench-compare ARGS="a.exr b.exr"       # your own files
```

Prints a side-by-side table of tinyexr vs OpenEXR encode/decode throughput
(MP/s) and compressed size for every codec both support, **including HTJ2K256
and HTJ2K32**. Each library independently loads the same source file into its
own native pixel representation, then encodes/decodes fully in memory; both run
single-threaded (`Imf::setGlobalThreadCount(0)`) for an apples-to-apples
comparison.

Needs a built OpenEXR (4.x, with HTJ2K/OpenJPH). Point the target at your tree
if it is not at `~/work/openexr`:

```sh
make bench-compare OPENEXR_ROOT=/path/to/openexr OPENEXR_BUILD=/path/to/openexr/_build
```

The tinyexr side lives in `bench_tx.c` (compiled as C): tinyexr's `exr.h` and
OpenEXR's C core declare the same global enum names, so they cannot share a
translation unit — `bench_tx.h` exposes a clean wrapper instead.

## SIMD tiers

The kernels are runtime-dispatched from CPUID (`exr_simd_info()` prints the
selected path). Tiers measured:

| tier   | de-interleave | half↔float | predictor (delta) |
|--------|---------------|------------|-------------------|
| scalar | ✅            | ✅         | ✅                |
| SSE2   | ✅            | —          | ✅                |
| AVX2   | ✅            | F16C       | (uses SSE2)       |
| NEON   | ✅            | (scalar)   | ✅ (aarch64)      |

**ARM / NEON.** The byte de-interleave and the predictor (prefix-sum decode /
delta encode) have NEON kernels (`src/exr_simd_neon.c`), selected automatically
on aarch64. To confirm the NEON path builds and is bit-identical to scalar under
an emulator:

```sh
make arm-smoke    # aarch64-linux-gnu-gcc + qemu-aarch64, static, small image
```

It prints `active tier: neon` and round-trips a small image plus a
scalar-vs-NEON predictor equality check. `make host-smoke` runs the same test
natively. clang also cross-compiles it, e.g.
`clang --target=aarch64-linux-gnu --gcc-toolchain=/usr ... | qemu-aarch64 -L /usr/aarch64-linux-gnu`.

**PSHUFB Huffman-emit:** the `DEFLATE encode on EXR bytes` section compares
three encoders on real predictor+split EXR bytes: the default fpng LZ77 encoder,
and the fpnge-derived literal encoder with its PSHUFB per-byte Huffman-table
lookup (scalar vs SSE4.1). The PSHUFB lookup is the SSE4.1 (`pshufb`) tier. Note
that fpnge's literal-only, PNG-tuned table tends to *expand* EXR data, so it is
an opt-in/benchmark encoder — the default ZIP codec keeps the fpng LZ77 path.

## Measured results

Machine: **AMD Ryzen 9 3950X (Zen2)**, idle. SIMD reported as `avx2+f16c`
(caps `0x7` = SSE2 | SSE4.1 | AVX2). Source: `asakusa.exr` (660×440, 4 ch).
Numbers vary run-to-run by a few percent.

> The full **tinyexr vs OpenEXR** comparison (single-thread + multi-thread,
> in-tree + libdeflate, with charts) lives in
> [`doc/performance-vs-openexr.md`](../doc/performance-vs-openexr.md). This
> section covers the bench's own SIMD-tier / micro-kernel numbers.

### SIMD tiers (`make bench`)

End-to-end decode by forced tier (the ZIP tier covers the SIMD predictor +
de-interleave; HTJ2K256 the JPH dispatch):

| path                  | scalar | SSE2/SSE4.1 | AVX2 |
|-----------------------|-------:|------------:|-----:|
| ZIP decode (MP/s)     | 36.6   | **47.5** (SSE2) | 47.6 |
| HTJ2K256 encode (MP/s)| 7.5    | 7.5 (SSE4.1)| 7.6  |
| HTJ2K256 decode (MP/s)| 13.7   | 14.3 (SSE4.1)| **19.6** |

SIMD micro-kernels (throughput, speedup vs scalar):

| kernel                  | scalar    | SSE2/SSE4.1     | AVX2            |
|-------------------------|----------:|----------------:|----------------:|
| byte de-interleave      | 3.04 GB/s | 7.93 GB/s (2.61×, SSE2) | 6.81 GB/s (2.24×) |
| half→float              | 3.21 GB/s | —               | 14.70 GB/s (4.57×, F16C) |
| fpnge PSHUFB lookup     | 0.50 GB/s | 2.71 GB/s (5.44×, SSE4.1) | 3.21 GB/s (6.44×) |
| JPH `nlt3` (NLT type-3) | 305.6 M/s | 1204.3 M/s (3.94×, SSE2) | 1190.4 M/s (3.89×) |
| JPH `pack` (i32→half)   | 1617.6 M/s | 2197.6 M/s (1.36×, SSE4.1) | 2415.8 M/s (1.49×) |
| predictor decode (delta)| 1.35 GB/s | **4.78 GB/s** (3.54×, SSE2) | (SSE2) |

The predictor (delta) decode is a serial byte prefix-sum; the SSE2 version
builds the prefix sum with log₂(16) lane shifts and carries the running total
across chunks (~3.5× over scalar). It runs on every ZIP/ZIPS/PXR24/RLE decode.

De-interleave and `pack` are memory-bandwidth-bound, so AVX2 does not beat SSE
much. The compute-bound kernels — F16C half→float, PSHUFB Huffman lookup, JPH
NLT — get the biggest wins, and AVX2 lifts HTJ2K256 decode by ~1.4×.

### tinyexr vs OpenEXR

The codec-by-codec comparison (encode/decode throughput and size), the optional
`LIBDEFLATE=1` backend results, and the multi-threaded scaling + 16-thread
comparison — all on a clean idle machine, with charts — are written up in
[`doc/performance-vs-openexr.md`](../doc/performance-vs-openexr.md).

Quick reproduce:

```sh
make bench-compare                         # single-thread, in-tree codec
make bench-compare LIBDEFLATE=1            # + vendored libdeflate backend
make bench-compare THREADS=1               # build with C11-threads support
EXR_THREADS=16 ./build/bench_compare       # 16 threads, both libraries
```

Headlines (asakusa.exr, single thread): tinyexr decodes uncompressed ~3.4× and
RLE ~2.5× faster than OpenEXR; OpenEXR leads the deflate family / PIZ / HTJ2K
(its libdeflate / tuned PIZ / OpenJPH backends). With `LIBDEFLATE=1` tinyexr
matches or beats OpenEXR on the deflate family (ZIP decode ~1.4×). With
`THREADS=1` tinyexr's per-block parallel path scales ~5–9× to 16 threads and, at
16 threads, out-decodes OpenEXR on RLE/ZIP/ZIPS/B44 (and the whole deflate family
when libdeflate is also enabled).
