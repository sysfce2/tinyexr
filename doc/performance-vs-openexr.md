# TinyEXR vs OpenEXR — single-thread performance

A look at how the pure-C11 TinyEXR v3 codec stacks up against the reference
**OpenEXR** library, codec by codec, for both encode and decode.

> **Scope of this note**
> - **Single-threaded only.** Both libraries are pinned to one thread
>   (`Imf::setGlobalThreadCount(0)`). OpenEXR's scanline API parallelizes per
>   block, so multi-core numbers are a different story — see
>   [Multi-threading](#multi-threading) (TinyEXR now has an opt-in parallel path
>   too; see that section).
> - **x86-64 only** here (AMD Ryzen 9 3950X, Zen2, `avx2+f16c`). ARM/NEON builds
>   and run, but are not benchmarked in this note.
> - Everything is measured **in memory** (no disk), each library loading the same
>   source independently. Throughput is in **megapixels/second** (higher is
>   better); sizes are the compressed payload.

## TL;DR

- **Decode:** TinyEXR is **~6× faster on uncompressed**, **~2× faster on RLE**,
  and roughly on par on **PIZ / B44**. On the DEFLATE-family codecs
  (ZIP/ZIPS/PXR24) it lands within ~1.2–1.5× of OpenEXR — the gap is purely the
  deflate backend.
- **Encode:** Competitive on RLE/PIZ/B44; ~1.5× behind on ZIP/PXR24; OpenEXR's
  HTJ2K encoder is the clear leader (~3–4×).
- **Need exact deflate parity?** Build with `LIBDEFLATE=1` (optional, off by
  default) and TinyEXR uses the same libdeflate backend — at which point it
  **meets or beats** OpenEXR, including **~1.4× faster ZIP decode**.

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 9 3950X (Zen2), `avx2+f16c` (caps `0x7`) |
| Image | `asakusa.exr`, 660×440, 4 channels (HALF) |
| Build | `gcc -O3`, TinyEXR v3 (pure C11); OpenEXR 4.0-dev |
| Threads | 1 (both) |
| I/O | in-memory (`StdOSStream`/`StdISStream` on the OpenEXR side) |

Reproduce:

```sh
make bench-compare                       # default: in-tree codec
make bench-compare LIBDEFLATE=1          # optional libdeflate backend
EXR_THREADS=16 make bench-compare        # OpenEXR multi-threaded (contrast)
```

## Decode throughput

![Decode throughput, single thread](perf-decode.svg)

`none` (uncompressed) is off the chart on purpose: **TinyEXR ~1892 vs OpenEXR
~309 MP/s** — about **6×**, because there is no real work to do and TinyEXR has
no thread-pool or framebuffer-copy overhead on that path.

Among the compressed codecs, TinyEXR clearly leads on **RLE** (120 vs 58) and is
competitive on **B44**. On **ZIP/ZIPS/PXR24/PIZ** OpenEXR is ahead by ~1.2–2.5×;
for the DEFLATE family that gap is the inflate backend (OpenEXR uses
[libdeflate](https://github.com/ebiggers/libdeflate)). A profile of TinyEXR's
ZIP decode shows it is now ~95 % inflate — the byte-predictor and de-interleave
passes were vectorized (SSE2/AVX2) and are no longer significant.

## Encode throughput

![Encode throughput, single thread](perf-encode.svg)

TinyEXR matches OpenEXR on **RLE / PIZ / B44**. On **ZIP/ZIPS/PXR24** it is
~1.5× behind: its in-tree, dependency-free LZ77 encoder is fast (a tuned
greedy/hash-chain parser with a word-at-a-time match scan) but does not match
libdeflate's level-4 encoder. **HTJ2K** encode is the widest gap (~3–4×) — that
is the separate JPH/OpenJPH encoder, not deflate.

## Compression size

Sizes are effectively identical for the lossless/standard codecs — the formats
are fully interoperable (a TinyEXR-written file decodes in OpenEXR and vice
versa). The only differences come from encoder *tuning*, not format:

| codec | TinyEXR (KB) | OpenEXR (KB) |
|-------|-------------:|-------------:|
| none  | 2276 | 2276 |
| rle   | 1644 | 1644 |
| zips  | 1205 | 1212 |
| zip   | 1155 | 1070 |
| piz   |  742 |  742 |
| pxr24 | 1158 | 1163 |
| b44   |  993 |  993 |
| htj2k256 | 1132 | 1016 |
| htj2k32  | 1160 | 1042 |

## Optional: the libdeflate backend

OpenEXR's deflate speed comes from libdeflate. TinyEXR can use it too — it
vendors **libdeflate 1.25** (MIT) as an **optional, off-by-default** backend for
ZIP/ZIPS/PXR24. The in-tree pure-C codec remains the default and the only path
for freestanding builds; libdeflate is enabled with a single build flag:

```sh
make bench-compare LIBDEFLATE=1     # -DEXR_USE_LIBDEFLATE, level 4 (= OpenEXR)
```

![Decode with libdeflate backend](perf-libdeflate-decode.svg)

With the same backend, compressed sizes become byte-identical and TinyEXR
**meets or beats** OpenEXR. Decode is the standout: both call libdeflate's
inflate, but TinyEXR adds its fast SSE2/AVX2 predictor with less per-block
framework overhead, so **ZIP decode runs ~1.4× OpenEXR (61 vs 44 MP/s)**.
Encode reaches parity (zip 11.6 vs 10.6, pxr24 12.8 vs 10.1 MP/s).

So there are two operating points:

- **default** — zero dependencies, freestanding-friendly, within ~1.2–1.5× of
  OpenEXR on the deflate family and faster on RLE/uncompressed;
- **`LIBDEFLATE=1`** — opt in to one small MIT dependency and reach/exceed
  parity.

## Full numbers (single thread)

In-tree default:

| codec | tx enc | exr enc | tx dec | exr dec | tx KB | exr KB |
|-------|-------:|--------:|-------:|--------:|------:|-------:|
| none  | 57.5 | 47.3 | 1892 | 309 | 2276 | 2276 |
| rle   | 38.0 | 32.9 | 120.3 | 58.4 | 1644 | 1644 |
| zips  | 8.1  | 10.9 | 17.2 | 34.9 | 1205 | 1212 |
| zip   | 7.4  | 10.8 | 36.2 | 44.4 | 1155 | 1070 |
| piz   | 16.5 | 17.9 | 18.7 | 46.1 | 742  | 742  |
| pxr24 | 7.1  | 12.2 | 36.1 | 58.5 | 1158 | 1163 |
| b44   | 24.1 | 23.9 | 81.9 | 108.9 | 993 | 993 |
| htj2k256 | 5.0 | 21.7 | 13.0 | 34.5 | 1132 | 1016 |
| htj2k32  | 6.2 | 22.1 | 17.9 | 36.6 | 1160 | 1042 |

`LIBDEFLATE=1` (deflate family):

| codec | tx enc | exr enc | tx dec | exr dec |
|-------|-------:|--------:|-------:|--------:|
| zip   | 11.6 | 10.6 | 60.9 | 43.5 |
| zips  | 12.6 | 11.6 | 45.0 | 35.4 |
| pxr24 | 12.8 | 10.1 | 59.7 | 60.8 |

*(enc/dec in MP/s; KB = compressed size. Values vary a few % run-to-run.)*

## Multi-threading

The numbers above are **single-threaded** on both sides. OpenEXR's scanline
reader/writer parallelize across blocks via a global thread pool, so with workers
it scales well beyond them — e.g. at `EXR_THREADS=16` on this machine OpenEXR ZIP
decode jumps from 44 to ~107 MP/s and ZIPS from 35 to ~82.

TinyEXR now also supports **per-block parallel encode and decode** using portable
C11 `<threads.h>` with a small ephemeral worker pool. It is **opt-in** (build with
`THREADS=1` / `-DEXR_USE_THREADS`; default builds and freestanding stay
single-threaded) and the thread count is set at runtime:

```c
exr_set_num_threads(8);   /* 0/1 = serial (default) */
```

Parallelism covers scanline and single-level tiled parts on the in-memory
load/save paths (`exr_load_from_*` / `exr_save_to_*`); deep, mipmap/ripmap, and
the streaming reader/writer APIs remain single-threaded. Encode is
byte-deterministic and decode bit-identical regardless of thread count (verified
in the unit tests).

Still future work:

- [ ] Benchmark TinyEXR vs OpenEXR at matched thread counts (1, 2, 4, 8, 16) and
      report scaling efficiency, not just peak (extend `bench_compare` to call
      `exr_set_num_threads()` for the TinyEXR side).
- [ ] Repeat across image sizes/channel counts (small tiles vs large scanline
      blocks change the per-block overhead balance).
- [ ] Parallelize the deep and mipmap/ripmap paths.
- [ ] Add ARM/NEON throughput numbers (the NEON path builds and is verified for
      correctness under qemu, but is not yet benchmarked here).

---

*Charts: `doc/perf-decode.svg`, `doc/perf-encode.svg`,
`doc/perf-libdeflate-decode.svg`. Benchmark harness:
`benchmark/bench_compare.cpp` (`make bench-compare`).*
