/*
 * tocio - ARM64 NEON SIMD batch kernels for the interpreter (matrix, range).
 * Each reproduces its scalar reference using the same mul/add order (no FMA),
 * so the parity test matches.  NEON is mandatory on aarch64 so this file only
 * needs the preprocessor guard; no runtime CPUID required.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

#if defined(__aarch64__)

#include <arm_neon.h>

/* Match the scalar reference bit-for-bit: forbid the compiler from fusing the
 * separate vmulq/vaddq pairs below into vfmla (see toc_interp.c parity note). */
#if defined(__GNUC__) || defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

/* matrix: same column-major layout as SSE2 and scalar.
 * Process one RGBA pixel per iteration using 128-bit NEON vectors. */
void toc_matrix_batch_neon(const float *m, const float *off, float *rgba,
                           size_t npix, int ch) {
    float32x4_t c0, c1, c2, c3, vo;
    size_t i;
    if (ch != 4) { toc_matrix_batch_scalar(m, off, rgba, npix, ch); return; }
    c0 = vld1q_f32(m + 0);
    c1 = vld1q_f32(m + 4);
    c2 = vld1q_f32(m + 8);
    c3 = vld1q_f32(m + 12);
    vo = vld1q_f32(off);
    for (i = 0; i < npix; ++i) {
        float *px = rgba + i * 4;
        float32x4_t r = vdupq_n_f32(px[0]);
        float32x4_t g = vdupq_n_f32(px[1]);
        float32x4_t b = vdupq_n_f32(px[2]);
        float32x4_t a = vdupq_n_f32(px[3]);
        float32x4_t v = vaddq_f32(vmulq_f32(c0, r), vmulq_f32(c1, g));
        v = vaddq_f32(v, vmulq_f32(c2, b));
        v = vaddq_f32(v, vmulq_f32(c3, a));
        v = vaddq_f32(v, vo);
        vst1q_f32(px, v);
    }
}

/* range: scale + offset, optional lo/hi clamp.
 * Process one RGBA pixel per iteration. */
void toc_range_batch_neon(const float *scale, const float *offset,
                          const float *vmin, const float *vmax, int clamp_lo,
                          int clamp_hi, float *rgba, size_t npix, int ch) {
    float32x4_t vs, vof, lo_v, hi_v;
    size_t i;
    if (ch != 4) {
        toc_range_batch_scalar(scale, offset, vmin, vmax, clamp_lo, clamp_hi,
                               rgba, npix, ch);
        return;
    }
    vs = vld1q_f32(scale);
    vof = vld1q_f32(offset);
    lo_v = vld1q_f32(vmin);
    hi_v = vld1q_f32(vmax);
    for (i = 0; i < npix; ++i) {
        float *px = rgba + i * 4;
        float32x4_t v = vaddq_f32(vmulq_f32(vld1q_f32(px), vs), vof);
        if (clamp_lo) v = vmaxq_f32(v, lo_v);
        if (clamp_hi) v = vminq_f32(v, hi_v);
        vst1q_f32(px, v);
    }
}

#endif /* __aarch64__ */
