# tir — tiny image resize

A small, fast, content-aware image resizer in pure C11. Standalone (no
dependency on the TinyEXR core — only the CLI and benchmark link
`libtinyexr3.a` for EXR file I/O), one allocation per resize, SIMD-optimized
for SSE2 / SSE4.1 / AVX2(+FMA,F16C) with runtime dispatch, NEON at compile
time, and SVE 1.0 behind a `HWCAP_SVE` runtime gate.

```c
#include "tir.h"

tir_image_view src = {pixels, w, h, 4, TIR_F32, 0};
tir_image_view dst = {out, w/2, h/2, 4, TIR_F32, 0};
tir_options o;
tir_options_init(&o);
o.filter_x = o.filter_y = TIR_FILTER_LANCZOS3;
o.antiring  = 1.0f;   /* confine ringing to the local source range */
o.clamp_min = 0.0f;   /* hard guarantee: no negative pixels */
tir_resize(NULL, &src, &dst, &o);
```

## Why another resizer

- **HDR-aware.** Values are never clamped by default. Negative-lobe filters
  (Mitchell ±3.6%, Catmull-Rom ±7.4%, Lanczos3 ±11.8% step overshoot) ring
  around HDR highlights and produce negative pixels; `antiring` blends the
  result toward the min/max of a reduced source footprint — applied **once,
  after both passes** (per-pass clamping is non-separable and artifact-prone;
  same design as madVR/libplacebo), so `antiring = 1` plus `clamp_min = 0`
  gives a hard no-negative-pixels guarantee without flattening gradients.
  `hicomp` filters under the OIIO/SPI highlight-compression curve instead.
- **Normal-map-aware.** `TIR_MODE_NORMAL_MAP` decodes (`snorm`, `unorm`,
  NVTT center-127, or 2-channel RG with `z = sqrt(1-x²-y²)` reconstructed
  *before* filtering), filters the 3 components, and renormalizes once after
  both passes ((0,0,1) fallback for cancelled footprints); the
  pre-normalization |N| is available as a side output for Toksvig/roughness
  baking.
- **Displacement/height-map-aware.** `TIR_MODE_HEIGHTMAP` defaults to the
  mean-preserving area average on downscale, auto-enables `antiring = 1` for
  ringing filters (geometry spikes are worse than softness), and supports
  `TIR_REG_GRID_VERTEX` registration: texels as grid points, endpoints map
  exactly (the 2ⁿ+1 terrain convention), avoiding the half-texel-shift bug
  class.
- **Correct RGBA.** Premultiplied filtering by default (straight-alpha
  filtering bleeds the RGB of transparent texels); NaN/Inf scrub policies
  (`zero`, 3×3 `repair`, `error`) run before filtering, where one NaN would
  otherwise poison its whole footprint.

## Performance

Separable two-pass resampling over f32 with precomputed, edge-folded,
zero-padded coefficient tables (branchless inner loops), a streaming ring
buffer of `filter_taps` rows (O(support) memory, one arena allocation), FMA
kernels with independent accumulator chains, and pass-order selection by a
multiply-count cost model (vertical-first when the geometry favors it).

`make resize-bench`, output MP/s, f32, AMD/Intel AVX2 vs the TinyEXR core
resizer (`exr_resize_float`):

| shape, filter | scalar | SSE2 | AVX2 | exr_v3 |
|---|---|---|---|---|
| 2× up, mitchell, 4ch | 137 | 241 | 285 | 122 |
| 2× down, mitchell, 4ch | 38 | 55 | 67 | 22 |
| 7.3× down, mitchell, 1ch | 11.1 | 24.2 | 26.8 | 7.2 |
| 7.3× down, lanczos3, 1ch | 7.6 | 17.6 | 19.6 | — |

Pixel I/O types: `TIR_F32`, `TIR_F16` (IEEE half, F16C/NEON converters),
`TIR_U8`/`TIR_U16` unorm. Filtering is always f32; SIMD kernels may
reassociate sums within a small ULP budget — `deterministic = 1` pins the
scalar reference kernels for bit-identical output across platforms, SIMD
levels, and thread counts.

## API shape

- `tir_resize(alloc, &src, &dst, &opt)` — one-shot.
- `tir_sampler_create/_run/_destroy` — prebuilt sampler, reuse across frames.
- `tir_sampler_push_row/_pull_row/_reset` — streaming rows, O(support)
  memory (`pull` returns `TIR_WOULD_BLOCK` until enough rows are pushed;
  `push` returns `TIR_WOULD_BLOCK` when the caller must drain pulls first).
- `tir_simd_available/_force/_info` — runtime kernel control (bench/tests).
- Optional band-parallel whole-image runs with C11 threads:
  build with `-DTIR_ENABLE_THREADS` and set `opt.num_threads`; output is
  byte-identical to serial for any thread count.

Errors are a signed `tir_result` (`TIR_OK(r)`); all size arithmetic is
overflow-checked; hostile dimensions are rejected. `EDGE_WRAP` in y buffers
the whole image (documented cost); everything else streams.

## Building / testing

Targets live in the repository root Makefile:

```
make resize-lib            # build/libtir.a
make resize-c11-gate       # strict C11 -Werror syntax gate + stdio scan
make resize-test           # unit tests under ASan+UBSan (vs a naive
                           #   double-precision reference resampler)
make resize-test-threads   # same, with -DTIR_ENABLE_THREADS
make resize-test-tsan      # ThreadSanitizer build
make resize-bench          # MP/s vs exr_resize_float; STB=1 adds a
                           #   stb_image_resize2 column (drop the header
                           #   into tools/resize/tests/ first, not vendored)
make resize-cli            # build/tir_resize (EXR in/out)
make resize-arm-test       # aarch64 NEON cross-build, runs under qemu
make resize-sve-test       # +SVE unit, qemu -cpu max,sve=on (HWCAP gate)
```

Only `tools/resize/src/tir_kernels_sve.c` is compiled with `+sve`
(`TIR_SVE=1` on native aarch64); everything else stays at the baseline ISA so
the runtime gate is sound. There is no non-streaming SVE on Apple Silicon —
NEON is the path there.

## CLI

```
make resize-cli
./build/tir_resize in.exr -o out.exr --scale 0.5 --filter lanczos3 \
    --antiring 1 --clamp-min 0 --stats
./build/tir_resize height.exr -o small.exr --mode height --reg vertex
./build/tir_resize normal.exr -o mip1.exr --mode normal
```

`--stats` prints per-channel min/max/mean and the negative-pixel count of
the output (the antiring/clamp guarantees are directly observable).
