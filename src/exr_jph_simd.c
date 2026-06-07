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

/* int32 NLT type-3 (bit_depth<=31): if v<0, v = ~v - biasm1 (== -v-bias, which
 * always fits int32 there). Pure int32, masked - no overflow, native compare. */

EXR_TARGET("sse2")
void jph_nlt_type3_i32_sse2(int32_t *data, size_t count, int32_t biasm1) {
    size_t i = 0;
    __m128i vb = _mm_set1_epi32(biasm1);
    __m128i ones = _mm_set1_epi32(-1);
    __m128i zero = _mm_setzero_si128();
    for (; i + 4 <= count; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(data + i));
        __m128i neg = _mm_sub_epi32(_mm_xor_si128(v, ones), vb); /* ~v - biasm1 */
        __m128i mask = _mm_cmplt_epi32(v, zero);
        __m128i r = _mm_or_si128(_mm_and_si128(mask, neg),
                                 _mm_andnot_si128(mask, v));
        _mm_storeu_si128((__m128i *)(data + i), r);
    }
    for (; i < count; ++i)
        if (data[i] < 0) data[i] = ~data[i] - biasm1;
}

EXR_TARGET("avx2")
void jph_nlt_type3_i32_avx2(int32_t *data, size_t count, int32_t biasm1) {
    size_t i = 0;
    __m256i vb = _mm256_set1_epi32(biasm1);
    __m256i ones = _mm256_set1_epi32(-1);
    __m256i zero = _mm256_setzero_si256();
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i neg = _mm256_sub_epi32(_mm256_xor_si256(v, ones), vb);
        __m256i mask = _mm256_cmpgt_epi32(zero, v); /* v < 0 */
        __m256i r = _mm256_blendv_epi8(v, neg, mask);
        _mm256_storeu_si256((__m256i *)(data + i), r);
    }
    for (; i < count; ++i)
        if (data[i] < 0) data[i] = ~data[i] - biasm1;
}

/* ----------------------------------------------------------------------------
 * Pixel pack: int32 plane sample -> little-endian uint16, by truncation (low 16
 * bits), the all-HALF decode store. A byte shuffle (vpshufb) gathers the low two
 * bytes of each int32 lane; truncation (not saturation) keeps it bit-identical
 * to the scalar (uint16_t)v store even for out-of-range/corrupt coefficients.
 * ------------------------------------------------------------------------- */

EXR_TARGET("sse4.1")
void jph_pack_i32_to_half_sse41(uint8_t *dst, const int32_t *src, size_t n) {
    size_t i = 0;
    const __m128i sh = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                     -1, -1, -1, -1, -1, -1, -1, -1);
    for (; i + 4 <= n; i += 4) {
        __m128i v = _mm_loadu_si128((const __m128i *)(src + i));
        __m128i s = _mm_shuffle_epi8(v, sh);
        _mm_storel_epi64((__m128i *)(dst + 2u * i), s);
    }
    for (; i < n; ++i) {
        uint16_t b = (uint16_t)src[i];
        dst[2u * i] = (uint8_t)b;
        dst[2u * i + 1u] = (uint8_t)(b >> 8);
    }
}

EXR_TARGET("avx2")
void jph_pack_i32_to_half_avx2(uint8_t *dst, const int32_t *src, size_t n) {
    size_t i = 0;
    const __m256i sh = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i s = _mm256_shuffle_epi8(v, sh); /* per-128-lane gather */
        _mm_storel_epi64((__m128i *)(dst + 2u * i),
                         _mm256_castsi256_si128(s));
        _mm_storel_epi64((__m128i *)(dst + 2u * i + 8u),
                         _mm256_extracti128_si256(s, 1));
    }
    for (; i < n; ++i) {
        uint16_t b = (uint16_t)src[i];
        dst[2u * i] = (uint8_t)b;
        dst[2u * i + 1u] = (uint8_t)(b >> 8);
    }
}

/* ----------------------------------------------------------------------------
 * Inverse reversible 5/3 1D lifting (int32 fast path). Mirrors the scalar
 * exr_jph_inverse_53_i32 exactly: int64 intermediates, floor division by 2^s
 * (== arithmetic right shift), and EXR_ERROR_CORRUPT iff any reconstructed
 * sample leaves int32. Computes the even/odd sub-sequences into caller scratch
 * (ev/od) then narrows+interleaves into out, so there is no scatter.
 * ------------------------------------------------------------------------- */

/* floor(v / 2^s), identical to exr_jph.c's jph_floor_div_pow2 (scalar edges). */
static int64_t jph_sra64_ref(int64_t v, unsigned s) {
    int64_t d = (int64_t)1 << s;
    if (v >= 0) return v / d;
    return -(((-v) + d - 1) / d);
}

EXR_TARGET("avx2")
static __m256i jph_sra64x4(__m256i x, int s) {
    /* arithmetic >> s on 4x int64 (s in 1..2); == floor div by 2^s */
    __m256i sign = _mm256_srai_epi32(_mm256_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1)), 31);
    __m256i lo = _mm256_srli_epi64(x, s);
    __m256i hi = _mm256_slli_epi64(sign, 64 - s);
    return _mm256_or_si256(lo, hi);
}

/* widen 4x int32 (low 128) -> 4x int64 */
EXR_TARGET("avx2")
static __m256i jph_widen_i32x4(const int32_t *p) {
    return _mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i *)p));
}

/* narrow 4x int64 (in int32 range) -> 4x int32 (__m128i), by truncation */
EXR_TARGET("avx2")
static __m128i jph_narrow_i64x4(__m256i v) {
    const __m256i sh = _mm256_setr_epi8(
        0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 2, 3, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i s = _mm256_shuffle_epi8(v, sh);
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(s),
                              _mm256_extracti128_si256(s, 1));
}

EXR_TARGET("avx2")
exr_result jph_inverse_53_i32_avx2(const int32_t *low, size_t low_count,
                                   const int32_t *high, size_t high_count,
                                   int32_t *out, size_t out_count,
                                   int64_t *ev, int64_t *od) {
    size_t i;
    size_t expected_low = (out_count + 1u) / 2u;
    size_t expected_high = out_count / 2u;
    __m256i two = _mm256_set1_epi64x(2);
    __m256i imin = _mm256_set1_epi64x(INT32_MIN);
    __m256i imax = _mm256_set1_epi64x(INT32_MAX);
    __m256i ovf = _mm256_setzero_si256();

    if (low_count != expected_low || high_count != expected_high)
        return EXR_ERROR_INVALID_ARGUMENT;
    if (out_count == 0) return EXR_SUCCESS;
    if (high_count == 0) { out[0] = low[0]; return EXR_SUCCESS; }

    /* ---- predict: ev[i] = low[i] - floor((dl+dr+2)/2) ---- */
    ev[0] = (int64_t)low[0] -
            jph_sra64_ref((int64_t)high[0] + (int64_t)high[0] + 2, 2);
    i = 1;
    for (; i + 4 <= high_count; i += 4) {
        __m256i dl = jph_widen_i32x4(high + i - 1);
        __m256i dr = jph_widen_i32x4(high + i);
        __m256i lo = jph_widen_i32x4(low + i);
        __m256i q = jph_sra64x4(_mm256_add_epi64(_mm256_add_epi64(dl, dr), two), 2);
        _mm256_storeu_si256((__m256i *)(ev + i), _mm256_sub_epi64(lo, q));
    }
    for (; i < low_count; ++i) {
        int64_t dl = high[i - 1];
        int64_t dr = (i < high_count) ? high[i] : high[high_count - 1];
        ev[i] = (int64_t)low[i] - jph_sra64_ref(dl + dr + 2, 2);
    }

    /* ---- update: od[i] = high[i] + floor((ev[i]+ev[i+1])/2) ---- */
    i = 0;
    for (; i + 4 <= high_count && i + 4 < low_count; i += 4) {
        __m256i e0 = _mm256_loadu_si256((const __m256i *)(ev + i));
        __m256i e1 = _mm256_loadu_si256((const __m256i *)(ev + i + 1));
        __m256i hi = jph_widen_i32x4(high + i);
        __m256i q = jph_sra64x4(_mm256_add_epi64(e0, e1), 1);
        _mm256_storeu_si256((__m256i *)(od + i), _mm256_add_epi64(hi, q));
    }
    for (; i < high_count; ++i) {
        int64_t e0 = ev[i];
        int64_t e1 = (i + 1u < low_count) ? ev[i + 1] : e0;
        od[i] = (int64_t)high[i] + jph_sra64_ref(e0 + e1, 1);
    }

    /* ---- narrow + interleave with int32-range check ---- */
    i = 0;
    for (; i + 4 <= high_count; i += 4) {
        __m256i e = _mm256_loadu_si256((const __m256i *)(ev + i));
        __m256i o = _mm256_loadu_si256((const __m256i *)(od + i));
        __m128i e32, o32;
        ovf = _mm256_or_si256(ovf, _mm256_cmpgt_epi64(e, imax));
        ovf = _mm256_or_si256(ovf, _mm256_cmpgt_epi64(imin, e));
        ovf = _mm256_or_si256(ovf, _mm256_cmpgt_epi64(o, imax));
        ovf = _mm256_or_si256(ovf, _mm256_cmpgt_epi64(imin, o));
        e32 = jph_narrow_i64x4(e);
        o32 = jph_narrow_i64x4(o);
        _mm_storeu_si128((__m128i *)(out + 2u * i), _mm_unpacklo_epi32(e32, o32));
        _mm_storeu_si128((__m128i *)(out + 2u * i + 4u),
                         _mm_unpackhi_epi32(e32, o32));
    }
    if (!_mm256_testz_si256(ovf, ovf)) return EXR_ERROR_CORRUPT;
    for (; i < high_count; ++i) {
        if (ev[i] < INT32_MIN || ev[i] > INT32_MAX) return EXR_ERROR_CORRUPT;
        if (od[i] < INT32_MIN || od[i] > INT32_MAX) return EXR_ERROR_CORRUPT;
        out[2u * i] = (int32_t)ev[i];
        out[2u * i + 1u] = (int32_t)od[i];
    }
    if (low_count > high_count) { /* trailing even (odd out_count) */
        if (ev[high_count] < INT32_MIN || ev[high_count] > INT32_MAX)
            return EXR_ERROR_CORRUPT;
        out[2u * high_count] = (int32_t)ev[high_count];
    }
    return EXR_SUCCESS;
}

#endif /* EXR_X86 */
