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
   byte de-interleave kernel forced to scalar / SSE2 / AVX2 (via
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

Machine: **AMD Ryzen 9 3950X (Zen2)**, single-threaded. SIMD reported as
`avx2+f16c` (caps `0x7` = SSE2 | SSE4.1 | AVX2). Source: `asakusa.exr`
(660×440, 4 ch). Numbers vary run-to-run by a few percent.

### SIMD tiers (`make bench`)

End-to-end decode by forced de-interleave tier, and HTJ2K256 (JPH) by forced
tier:

| path                  | scalar | SSE2/SSE4.1 | AVX2 |
|-----------------------|-------:|------------:|-----:|
| ZIP decode (MP/s)     | 29.2   | **37.0** (SSE2) | 36.2 |
| HTJ2K256 encode (MP/s)| 5.2    | 5.5 (SSE4.1)| 5.4  |
| HTJ2K256 decode (MP/s)| 10.1   | 10.2 (SSE4.1)| **13.3** |

(ZIP decode tier now covers both the SIMD predictor and the de-interleave;
SSE2 lifts it ~1.27×.)

SIMD micro-kernels (throughput, speedup vs scalar):

| kernel                  | scalar    | SSE2/SSE4.1     | AVX2            |
|-------------------------|----------:|----------------:|----------------:|
| byte de-interleave      | 2.03 GB/s | 4.02 GB/s (1.98×, SSE2) | 3.52 GB/s (1.74×) |
| half→float              | 2.31 GB/s | —               | 6.60 GB/s (2.85×, F16C) |
| fpnge PSHUFB lookup     | 0.36 GB/s | 1.53 GB/s (4.28×, SSE4.1) | 1.64 GB/s (4.58×) |
| JPH `nlt3` (NLT type-3) | 238.7 M/s | 563.1 M/s (2.36×, SSE2) | 601.9 M/s (2.52×) |
| JPH `pack` (i32→half)   | 914.9 M/s | 1105.5 M/s (1.21×, SSE4.1) | 1148.7 M/s (1.26×) |
| predictor decode (delta)| 0.95 GB/s | **3.3 GB/s** (3.4×, SSE2)  | (SSE2)          |

The predictor (delta) decode is a serial byte prefix-sum; the SSE2 version
builds the prefix sum with log₂(16) lane shifts and carries the running total
across chunks, ~3.4× over scalar. It runs on every ZIP/ZIPS/PXR24/RLE decode.

De-interleave and `pack` are memory-bandwidth-bound, so AVX2 does not beat SSE
much (and de-interleave's SSE2 path is actually fastest here). The compute-bound
kernels — F16C half→float, PSHUFB Huffman lookup, JPH NLT — get the biggest
wins, and AVX2 lifts HTJ2K256 decode by ~1.35×.

### tinyexr vs OpenEXR (`make bench-compare`)

OpenEXR 4.0-dev. **Both single-threaded** — the harness calls
`Imf::setGlobalThreadCount(0)` (calling thread only) by default; set
`EXR_THREADS=N` to give OpenEXR a worker pool (see "Threading" below). Both
fully in-memory.

| codec    | tx enc | exr enc | tx dec | exr dec | tx KB | exr KB |
|----------|-------:|--------:|-------:|--------:|------:|-------:|
| none     | 57.5   | 47.3    | **1892** | 309  | 2276 | 2276 |
| rle      | 38.0   | 32.9    | **120.3** | 58.4 | 1644 | 1644 |
| zips     | 8.1    | 10.9    | 17.2   | 34.9    | 1205  | 1212   |
| zip      | 7.4    | 10.8    | 36.2   | 44.4    | 1155  | **1070** |
| piz      | 16.5   | 17.9    | 18.7   | 46.1    | 742   | 742    |
| pxr24    | 7.1    | 12.2    | 36.1   | 58.5    | 1158  | 1163   |
| b44      | 24.1   | 23.9    | 81.9   | 108.9   | 993   | 993    |
| htj2k256 | 5.0    | 21.7    | 13.0   | 34.5    | 1132  | **1016** |
| htj2k32  | 6.2    | 22.1    | 17.9   | 36.6    | 1160  | **1042** |

(enc/dec in MP/s; KB = compressed size. Numbers vary a few % run-to-run.)

**Short summary.** Compressed-output sizes are essentially identical for
NONE/RLE/PIZ/PXR24/B44/ZIPS (codecs interoperate — a tinyexr-written ZIP decodes
correctly in OpenEXR and vice-versa); the only notable size gaps are ZIP/HTJ2K,
from different deflate/JPH encoder tuning. On throughput, single-threaded:

- **tinyexr wins** on uncompressed decode (~**6×**, 1892 vs 309 MP/s — no thread
  setup or framebuffer copy) and on **RLE decode** (~2×, 120 vs 58), and ties on
  RLE encode, PIZ, and B44.
- **ZIP/PXR24 are now close:** decode is within **~1.2×** (zip 36 vs 44, pxr24 36
  vs 59) and encode within **~1.5×** (was ~3.6×). The residual gap is the
  backend — OpenEXR uses **libdeflate** (level 4), the fastest deflate in
  existence; tinyexr keeps an in-tree, dependency-free fpng-derived LZ77 encoder
  + compact inflate. HTJ2K encode (~3–4× behind) is the separate JPH encoder.
  *For full deflate parity, build with `LIBDEFLATE=1` — see the next section.*

> **In-tree codec tuning (this branch), no new dependencies.**
> - **Deflate encoder ~2.3×** (ZIP encode 3.1 → 7.4 MP/s, PXR24 3.2 → 7.1, ZIPS
>   5.3 → 8.1) with *no* ratio loss: word-at-a-time match scan, `nice_length`
>   early-out, shorter hash chain (128 → 16), and a 4-byte multiplicative hash
>   with a block-sized table (no 128 KB memset per tiny ZIPS block).
> - **SIMD predictor (SSE2)** — the serial delta/prefix-sum is now ~3.4× faster
>   (0.95 → 3.3 GB/s), lifting **ZIP decode 31 → 36** and **RLE decode 73 → 120**
>   MP/s. Bit-identical to the scalar reference (verified) and fuzz-clean.
> - **Inflate** `copy_match` advances 8 bytes at a time when `distance ≥ 8`.
>
> Decode time for a ZIP block is now ~95 % inflate (predictor went from 28 % of
> the cost to ~8 %); closing the last bit would mean a libdeflate-class inflate.

### Optional libdeflate backend (`LIBDEFLATE=1`)

OpenEXR's deflate speed comes from **libdeflate**. TinyEXR vendors libdeflate
1.25 (MIT, `deps/libdeflate/COPYING`) as an **optional, off-by-default** backend
for ZIP/ZIPS/PXR24 — the in-tree pure-C codec stays the default and remains the
only path for freestanding builds. Enable it on any target:

```sh
make bench-compare LIBDEFLATE=1     # run 'make clean' when toggling the flag
```

The flag defines `EXR_USE_LIBDEFLATE`; ZIP/PXR24 then route their zlib
compress/decompress through libdeflate (level 4, matching OpenEXR; override with
`-DEXR_LIBDEFLATE_LEVEL=N`). Only the zlib path is compiled (no crc32/gzip).
With it on, the same machine, single-threaded:

| codec | tx enc | exr enc | tx dec | exr dec | tx KB | exr KB |
|-------|-------:|--------:|-------:|--------:|------:|-------:|
| zip   | **11.6** | 10.6  | **60.9** | 43.5 | 1070 | 1070 |
| zips  | **12.6** | 11.6  | **45.0** | 35.4 | 1212 | 1212 |
| pxr24 | **12.8** | 10.1  | 59.7   | 60.8    | 1163 | 1163 |

Sizes become identical (same libdeflate level), and tinyexr **matches or beats**
OpenEXR: both use libdeflate's inflate, but tinyexr's decode adds the fast SSE2
predictor with less per-block framework overhead, so **ZIP decode runs ~1.4×
OpenEXR** (61 vs 44 MP/s). This is the route to true deflate parity when a small
extra dependency is acceptable; otherwise the in-tree codec (above) stays within
~1.2–1.5× with zero dependencies.

### Threading (answering "is OpenEXR single-threaded?")

The table above is single-threaded on both sides. OpenEXR's scanline API
parallelizes per-block, so with a worker pool it pulls far ahead — which is why
the fair comparison pins it to one thread. For contrast, `EXR_THREADS=16` on the
same machine:

| codec | exr enc 1T → 16T | exr dec 1T → 16T |
|-------|-----------------:|-----------------:|
| zip   | 11.0 → **27.7**  | 44.1 → **107.0** |
| zips  | 11.2 → 24.8      | 35.2 → 81.6      |
| pxr24 | 12.0 → 36.3      | 59.5 → 114.4     |
| piz   | 17.7 → 34.6      | 45.7 → 89.5      |
| rle   | 32.4 → 73.2      | 56.5 → 76.4      |

tinyexr is single-threaded everywhere, so for a like-for-like core-vs-core
comparison use the default (`EXR_THREADS=0`).
