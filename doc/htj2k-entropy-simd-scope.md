# Scope: SIMD the HTJ2K cleanup-pass entropy decoder (decode-side)

Status: **scoping / not yet implemented.** Target: `src/exr_jph.c`,
`src/exr_jph_simd.c`. Reference: OpenJPH `ojph_block_decoder_avx2.cpp` /
`_ssse3.cpp` in `~/work/openexr/external/OpenJPH/src/core/coding/`.

## ⚠ Empirical finding (2026-06-09): Stage 0 regressed — premise needs revisiting

Stage 0 (the scalar per-quad 128-bit-window refactor) was implemented and
**measured ~4% SLOWER** on all-HALF htj2k256 decode (clean interleaved A/B,
min-of-6: 5.09s → 5.30s), and was reverted. Root cause: the assumption that
windowing "removes 3 of 4 fetches" is wrong for real data. HALF detail bands are
**sparse** — most quads have 0–1 significant samples — and the current
per-sample path does *zero* MagSgn work for insignificant samples. A per-quad
window pays a fixed read + offset-bookkeeping cost on every quad (even fully
insignificant ones; guarding the read by significance did not recover it).

Implication for the SIMD stages: per-quad/2-quad vectorization pays the same
fixed per-quad cost, so a net win requires enough significant samples per quad to
amortize it. That holds for **dense, low-frequency (LL / coarse-resolution)**
subbands but not the sparse high-frequency detail that dominates a typical HALF
image. OpenJPH's measured step-2 speedups likely come from dense/high-bit-depth
blocks plus its fully-SIMD surrounding pipeline, not this step on sparse data.

**Revised recommendation:** treat decode entropy SIMD as **low priority / uncertain
ROI** for the common all-HALF case. If pursued, gate it to dense quads (e.g.
significant-sample count ≥ threshold, or only the coarse resolutions) and
benchmark per-subband before committing. The wavelet/extraction SIMD already
landed are the higher-confidence wins. The analysis below stands as the design
if/when a dense-block path is wanted.

## Why

After the wavelet/extraction/unstuff SIMD work, both decode profiles are
dominated by the per-sample cleanup-pass entropy coder. Decode (all-HALF
htj2k256, asakusa, gprof): `jph_decode_codeblock` ~37% + `jph_magsgn_fetch64`
~9%. The bulk of that is **Step 2 (MagSgn magnitude/sign reconstruction)**, which
today runs one sample at a time with a per-sample bitstream fetch+advance
(`jph_decode_magsgn_sample64` / the inline loops; 260M `jph_magsgn_fetch64`
calls in the profile). OpenJPH vectorizes exactly this step and leaves the rest
scalar.

## What OpenJPH actually vectorizes (and what it does not)

- **Step 1 — MEL + VLC + UVLC quad decode: 100% scalar** in OpenJPH's AVX2 path
  too (variable-length, reverse-parsed, context-dependent). **tinyexr already
  matches this** (table-driven scalar, `vlc_tbl0/1`, `uvlc_tbl0/1`). *No work,
  no opportunity here.*
- **Step 2 — MagSgn reconstruction: heavily vectorized.** Per group of quads:
  compute each significant sample's bit count `m_n = U_q - e_k`, inclusive
  **prefix-sum** the `m_n` to get per-sample bit offsets, fetch 128 bits once,
  `vpshufb`-gather the two bytes straddling each sample's offset, variable-shift
  (`mullo_epi16`/`srlv`) to right-align `m_n` bits, then assemble
  `v_n = bits | (e_1<<m_n) | 1` and `val = sign | (v_n+2)<<(p-1)`. AVX2 does **2
  quads (8 samples)** at once (`decode_two_quad32_avx2`,
  avx2.cpp:787-887), or **4 quads (16 samples)** in the ≤16-bit path
  (`decode_four_quad16`, avx2.cpp:901-1036). SSSE3 does 1 quad
  (`decode_one_quad32`, ssse3.cpp:788-885).
- **MagSgn bit reader (`frwd_*`)**: OpenJPH unstuffs the stream **on the fly**
  with SSE byte-removal (avx2.cpp:587-775) — its single most intricate SIMD
  block.
- **SigProp / MRP**: mostly scalar; only small SSE "bit-spread" helpers
  (spread a 16-bit mask to bytes via shuffle + a popcount prefix-sum). Low value.

## Why this is *more* tractable in tinyexr than in OpenJPH

1. **Scratch format already matches.** tinyexr's `inf` word (`scratch[2k]`) packs
   `rho` (significance) at bits 4-7, `e_1` at 8-11, `e_k` at 12-15 — identical to
   OpenJPH's VLC-table entry. `U_q` is `scratch[2k+1]`. The per-sample inputs the
   SIMD needs are already laid out (see `jph_decode_block` step 2, exr_jph.c
   ~3469-3559).
2. **The MagSgn buffer is already unstuffed** (`JphMagSgn.buf`, a contiguous
   little-endian bit array; `jph_unstuff_bits` ran once up front). This **removes
   OpenJPH's hardest piece**: there is no on-the-fly `frwd_read` unstuffing to
   vectorize. A 128-bit window at any bit offset is a plain unaligned load from
   `buf` + a shift. The prefix-sum/`pshufb` gather reads straight from `buf`.
3. The scalar reference already exists and is bit-exact (the SIMD becomes a
   drop-in alternative, gated by a parity test — same discipline as the wavelet
   kernels).

## Bounding the scope: int32 path only

- The **int32 path** (`jph_decode_block`, all-HALF / `missing_msbs<30 && kmax<=30`)
  has `mmsbp2 = missing_msbs+2 <= 31`, so `m_n <= 31` and a quad's 4 samples
  consume `<= 124` bits — a **single 128-bit fetch per quad** suffices. This maps
  one-to-one onto OpenJPH's `decode_two_quad32` / `decode_four_quad16`. This is
  the common, high-value case.
- The **int64 path** (`jph_decode_block64_cleanup`, float/32-bit, `mmsbp2` up to
  62) has samples up to 62 bits → a quad can exceed 128 bits. **OpenJPH itself
  keeps the 64-bit decoder scalar** (`ojph_block_decoder64.cpp`). So we **leave
  the int64 MagSgn scalar** — matches the reference, and the int64 wavelet is
  already SIMD (commit f905c6d).

## The one real complication: U_q dependency in non-initial rows

- **Initial row** (exr_jph.c:3469-3559): `U_q = scratch[2k+1]` directly, **no
  cross-quad dependency** → two horizontally-adjacent quads (8 samples) are
  independent. **Clean first target.**
- **Non-initial rows** (exr_jph.c:3561-3680): `U_q = u_q + kappa`, where
  `kappa = gamma ? emax : 1` and `emax = 31 - clz(vp[0]|vp[1]|2)` reads the
  **v_n of the left neighbour quad** (`v_n_scratch`). That is a left-to-right
  serial dependency: quad x's `U_q` needs quad x-2's decoded `v_n`. So 2-quad
  horizontal batching is not directly available; options:
  - **(a)** keep the `U_q`/`kappa` recurrence scalar per quad, but vectorize the
    **4 samples within each quad** (still removes the 4 per-sample fetches → 1
    fetch + parallel extract). Lower speedup, lowest risk.
  - **(b)** mirror OpenJPH's formulation that derives `emax` from the step-1 EMB
    exponents rather than the full `v_n`, breaking the dependency to allow
    2-quad batching. Higher speedup, needs careful equivalence proof.
- The vectorized path must still write `v_n_scratch` (`vp[]`) so the kappa
  recurrence and the SigProp/MRP passes downstream stay correct.

## Staged plan

**Stage 0 — scaffolding + per-quad fetch (no SIMD yet).** Refactor step 2 of
`jph_decode_block` so a quad is decoded by: gather the 4 samples' `m_n`,
prefix-sum, one 128-bit window read from `magsgn.buf`, scalar per-sample extract.
This already removes 3 of every 4 `jph_magsgn_fetch64` calls and is a pure
scalar refactor — easy to verify byte-identical, and it establishes the data
layout the SIMD kernel will consume. Add a `jph_magsgn_fetch128`(offset) helper
on the unstuffed buffer.

**Stage 1 — AVX2 MagSgn for the initial row.** Port `decode_two_quad32` to a
`jph_decode_magsgn_quadpair_i32_avx2(inf0,Uq0,inf1,Uq1, buf, bitpos, p) -> 8
vals + 8 v_n`. Reuse the prefix-sum + `_mm_shuffle_epi8` gather +
`_mm_mullo_epi16` shift sequence (quoted in the agent notes). Bit-identical to
the scalar quad decode; gate behind `exr_cpu_caps() & EXR_SIMD_AVX2`; add a
parity unit test (`jph_simd_check` in test_exr_v3.c) over random
`(inf,U_q,p,buffer)` incl. all-insignificant, full-width `m_n`, and boundary
`p`. Measure.

**Stage 2 — AVX2 MagSgn for non-initial rows, option (a).** Keep the per-quad
`U_q`/`kappa` recurrence scalar; vectorize the 4-sample extract per quad
(single-quad SSSE3-style, `decode_one_quad32`). Still writes `v_n_scratch`.
Parity test + measure.

**Stage 3 (optional, higher risk) — non-initial 2-quad batching, option (b).**
Only if Stage 2 leaves meaningful headroom: adopt OpenJPH's EMB-exponent `emax`
to break the v_n dependency and decode two non-initial quads at once. Requires
proving the `emax` reformulation is identical to the current `vp[]`-based one.

**Out of scope (low value):** SigProp/MRP SSE bit-spread helpers; the int64
MagSgn path (scalar, matches OpenJPH); Step 1.

## Expected payoff & risk

- MagSgn reconstruction is the largest vectorizable slice of decode (a good chunk
  of the ~37% `jph_decode_codeblock` plus the ~9% fetch). OpenJPH gets ~2-4× on
  step 2; realistically this is a **~10-20% all-HALF decode** gain, concentrated
  in Stages 0-1 (which also delete most of the 260M per-sample fetch calls).
- **Risk: medium.** The arithmetic is intricate (variable-width bit gather, sign
  packing) but (i) the scalar reference is bit-exact and unit-tested, (ii) the
  unstuffed buffer removes the worst SIMD hazard, (iii) every stage is gated +
  parity-tested + verified byte-identical end-to-end on the corpus, exactly like
  the wavelet kernels. Stages are independently shippable.

## Key references

- tinyexr scalar step 2: `src/exr_jph.c` initial row 3469-3559, non-initial
  3561-3680 (int32); int64 mirror 2727-2848. Per-sample core: the
  `ms_val/m_n/v_n/val` block (3486-3496). MagSgn reader: `JphMagSgn` +
  `jph_magsgn_fetch64`/`_advance` ~2429-2490; unstuff `jph_unstuff_bits` ~2451.
- OpenJPH: `decode_two_quad32_avx2` avx2.cpp:787-887; `decode_four_quad16`
  901-1036; `decode_one_quad32` ssse3.cpp:788-885; `frwd_*` (NOT needed here)
  avx2.cpp:587-775.
- Parity-test pattern: `jph_simd_check` in `test/unit/test_exr_v3.c`.
