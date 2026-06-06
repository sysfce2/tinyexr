/*
 * TinyEXR - SIMD kernels for the JPH (HTJ2K) codec.
 *
 * These are runtime-dispatched (via exr_cpu_caps()) accelerated variants of the
 * vectorizable JPH transform/pixel stages. The scalar implementations in
 * exr_jph.c are the source of truth; every kernel here must be bit-identical to
 * its `_scalar` counterpart (verified in test/unit/test_exr_v3.c). Everything
 * compiles at the C11 baseline; per-function `__attribute__((target(...)))`
 * upgrades the ISA so the baseline never requires SSE/AVX.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

#if defined(EXR_X86)
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define EXR_TARGET(x) __attribute__((target(x)))
#else
#define EXR_TARGET(x)
#endif

/* ----------------------------------------------------------------------------
 * NLT type-3 (float non-linearity), the OpenEXR HTJ2K float transform. It is an
 * involution applied to int64 coefficient planes:  if (v < 0) v = -v - bias;
 * Fully element-independent -> a masked negate. Used by both decode (inverse)
 * and encode (forward), which are identical.
 * ------------------------------------------------------------------------- */

EXR_TARGET("sse2")
void jph_nlt_type3_i64_sse2(int64_t *data, size_t count, int64_t bias) {
    size_t i = 0;
    __m128i vbias = _mm_set1_epi64x(bias);
    __m128i zero = _mm_setzero_si128();
    for (; i + 2 <= count; i += 2) {
        __m128i v = _mm_loadu_si128((const __m128i *)(data + i));
        /* neg = (0 - v) - bias */
        __m128i neg = _mm_sub_epi64(_mm_sub_epi64(zero, v), vbias);
        /* mask = (v < 0) per 64-bit lane: broadcast the high-dword sign bit.
         * SSE2 has no 64-bit compare, so replicate each lane's high dword and
         * arithmetic-shift it right by 31 to fill the lane with its sign. */
        __m128i hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
        __m128i mask = _mm_srai_epi32(hi, 31);
        /* result = (mask ? neg : v) */
        __m128i r = _mm_or_si128(_mm_and_si128(mask, neg),
                                 _mm_andnot_si128(mask, v));
        _mm_storeu_si128((__m128i *)(data + i), r);
    }
    for (; i < count; ++i)
        if (data[i] < 0) data[i] = -data[i] - bias;
}

EXR_TARGET("avx2")
void jph_nlt_type3_i64_avx2(int64_t *data, size_t count, int64_t bias) {
    size_t i = 0;
    __m256i vbias = _mm256_set1_epi64x(bias);
    __m256i zero = _mm256_setzero_si256();
    for (; i + 4 <= count; i += 4) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i neg = _mm256_sub_epi64(_mm256_sub_epi64(zero, v), vbias);
        __m256i mask = _mm256_cmpgt_epi64(zero, v); /* v < 0 */
        __m256i r = _mm256_blendv_epi8(v, neg, mask);
        _mm256_storeu_si256((__m256i *)(data + i), r);
    }
    for (; i < count; ++i)
        if (data[i] < 0) data[i] = -data[i] - bias;
}

#endif /* EXR_X86 */
