# Closing the astcenc speed gap: port vs. continue

Status 2026-07-02 (branch texcomp-astc-quality, commit 12c2ae5), single
thread, 1024x1024 photo, 6x6, pure coding time:

| tier   | ours              | astcenc (-j 1)          | gap  |
|--------|-------------------|-------------------------|------|
| fast   | 0.176 s, 47.30 dB | -fastest 0.045 s, 47.78 | 3.9x |
| medium | 0.650 s, 48.43 dB | -medium  0.095 s, 48.63 | 6.8x |

## Option A: wholesale C11 port of astcenc

License: astcenc is **Apache-2.0** (not BSD). Permissive and compatible
with shipping inside a BSD-3-Clause tree, but ported files must retain
the Arm copyright + Apache-2.0 notice and the top-level NOTICE must say
so. tools/texcomp already mixes licenses knowingly (see NOTICE.md).

Scope: ~20k lines of C++14 built around a mandatory SIMD abstraction
(vfloat4/vint4 with SSE/NEON/SVE backends), float-based pipeline,
context/config machinery, and large generated tables. A faithful C11
port is a multi-week effort and would replace, not reuse, the current
integer encoder — including the reference-decoder test harness whose
bit-exact integer contract does not hold for a float pipeline.

## Option B (recommended): port astcenc's *algorithms*, keep our C11 core

The measured 4-7x comes from identifiable techniques, portable one at a
time into the existing encoder with the PSNR/parity harness as a gate:

1. **Block-mode percentile tables** (astcenc_block_sizes.cpp +
   percentile data): empirically ranked mode usage per footprint;
   -fastest keeps only the top few percent. Replaces our heuristic
   weight_bits+weight_count ranking; biggest search-space lever left.
2. **8-wide kernels** (astcenc_vecmathlib): widen our SSE2 kernels to
   AVX2 and add the LSQ-sum + infill kernels (33% + 15% of the fast
   profile are still scalar).
3. **Iterative ideal-endpoints-and-weights refinement**
   (astcenc_ideal_endpoints_and_weights.cpp): alternate weight/endpoint
   solves instead of our single LSQ pass — quality headroom that buys
   back search-count cuts.
4. **Bounded trial counts** (tune_candidate_limit etc.): astcenc's
   preset knobs map cleanly onto our scan_cap/selected_limit constants.

Measured dead end (do not revisit): winner-only LSQ — 10-20% speed for
0.2-1 dB; the current inline-LSQ config is on the frontier.

Threading remains the biggest single lever (astcenc numbers above are
already single-thread; ours parallelizes trivially per block row now
that the encode context is heap-allocated).
