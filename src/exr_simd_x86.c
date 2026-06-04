/*
 * TinyEXR - x86 SIMD kernels (SSE2 byte interleave, F16C half<->float).
 *
 * These use GCC/Clang function target attributes so the whole file compiles at
 * baseline ISA; the runtime dispatcher in exr_cpu.c only calls a kernel when
 * CPUID reports the matching feature. On MSVC the intrinsics compile directly.
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

/* De-split: out[2i]=t1[i], out[2i+1]=t2[i] (inverse of the EXR byte split). */
EXR_TARGET("sse2")
void exr_interleave_sse2(const uint8_t *src, uint8_t *dst, size_t n) {
    size_t half = (n + 1) / 2, n2 = n / 2, i = 0;
    const uint8_t *t1 = src, *t2 = src + half;
    for (; i + 16 <= n2; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(t1 + i));
        __m128i b = _mm_loadu_si128((const __m128i *)(t2 + i));
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm_unpacklo_epi8(a, b));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 16), _mm_unpackhi_epi8(a, b));
    }
    for (; i < n2; ++i) {
        dst[2 * i] = t1[i];
        dst[2 * i + 1] = t2[i];
    }
    if (n & 1) dst[n - 1] = t1[n2];
}

EXR_TARGET("avx2")
void exr_interleave_avx2(const uint8_t *src, uint8_t *dst, size_t n) {
    size_t half = (n + 1) / 2, n2 = n / 2, i = 0;
    const uint8_t *t1 = src, *t2 = src + half;
    for (; i + 32 <= n2; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(t1 + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(t2 + i));
        __m256i lo = _mm256_unpacklo_epi8(a, b);
        __m256i hi = _mm256_unpackhi_epi8(a, b);
        _mm256_storeu_si256((__m256i *)(dst + 2 * i),
                            _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_storeu_si256((__m256i *)(dst + 2 * i + 32),
                            _mm256_permute2x128_si256(lo, hi, 0x31));
    }
    for (; i < n2; ++i) {
        dst[2 * i] = t1[i];
        dst[2 * i + 1] = t2[i];
    }
    if (n & 1) dst[n - 1] = t1[n2];
}

EXR_TARGET("avx2,f16c")
void exr_half_to_float_f16c(const uint16_t *src, float *dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
    for (; i < count; ++i) {
        __m128i h = _mm_cvtsi32_si128(src[i]);
        dst[i] = _mm_cvtss_f32(_mm_cvtph_ps(h));
    }
}

#define EXR_F16_RND (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
EXR_TARGET("avx2,f16c")
void exr_float_to_half_f16c(const float *src, uint16_t *dst, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(f, EXR_F16_RND));
    }
    for (; i < count; ++i) {
        __m128 f = _mm_set_ss(src[i]);
        __m128i h = _mm_cvtps_ph(f, EXR_F16_RND);
        dst[i] = (uint16_t)_mm_extract_epi16(h, 0);
    }
}

#endif /* EXR_X86 */
