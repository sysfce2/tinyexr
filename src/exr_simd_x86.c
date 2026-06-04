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

/* fpnge-style PSHUFB Huffman-table lookup: for each byte, gather (nbits, code)
 * where the 16-bit code is split into blo (low 8) + bhi (high). Symbols 0-15
 * and 240-255 are looked up by low nibble; 16-239 share length 12 (mid_nbits),
 * so their code is mid_lowbits[hi_nibble] | (revnib[lo_nibble] << 8). */
EXR_TARGET("sse4.1")
void exr_fpnge_lookup_sse41(const exr_fpnge_table *t, const uint8_t *src,
                            size_t count, uint8_t *nb, uint8_t *blo,
                            uint8_t *bhi) {
    static const uint8_t kRevNib[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
    size_t i = 0;
    if (t->mid_nbits == 12) { /* always true for this construction */
        const __m128i f16n = _mm_loadu_si128((const __m128i *)t->first16_nbits);
        const __m128i f16b = _mm_loadu_si128((const __m128i *)t->first16_bits);
        const __m128i l16n = _mm_loadu_si128((const __m128i *)t->last16_nbits);
        const __m128i l16b = _mm_loadu_si128((const __m128i *)t->last16_bits);
        const __m128i midl = _mm_loadu_si128((const __m128i *)t->mid_lowbits);
        const __m128i rev = _mm_loadu_si128((const __m128i *)kRevNib);
        const __m128i nib = _mm_set1_epi8(0x0F);
        const __m128i midn = _mm_set1_epi8((char)t->mid_nbits);
        const __m128i zero = _mm_setzero_si128();
        const __m128i ones = _mm_set1_epi8((char)0xFF);
        const __m128i c15 = _mm_set1_epi8(15);
        const __m128i c239 = _mm_set1_epi8((char)239);
        for (; i + 16 <= count; i += 16) {
            __m128i b = _mm_loadu_si128((const __m128i *)(src + i));
            __m128i lo = _mm_and_si128(b, nib);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), nib);
            __m128i fnb = _mm_shuffle_epi8(f16n, lo);
            __m128i fb = _mm_shuffle_epi8(f16b, lo);
            __m128i lnb = _mm_shuffle_epi8(l16n, lo);
            __m128i lb = _mm_shuffle_epi8(l16b, lo);
            __m128i mlo = _mm_shuffle_epi8(midl, hi);
            __m128i mhi = _mm_shuffle_epi8(rev, lo);
            __m128i is_first = _mm_cmpeq_epi8(_mm_subs_epu8(b, c15), zero);
            __m128i is_last =
                _mm_xor_si128(_mm_cmpeq_epi8(_mm_subs_epu8(b, c239), zero), ones);
            __m128i fl = _mm_or_si128(is_first, is_last);
            __m128i rnb = _mm_blendv_epi8(midn, fnb, is_first);
            __m128i rlo = _mm_blendv_epi8(mlo, fb, is_first);
            rnb = _mm_blendv_epi8(rnb, lnb, is_last);
            rlo = _mm_blendv_epi8(rlo, lb, is_last);
            _mm_storeu_si128((__m128i *)(nb + i), rnb);
            _mm_storeu_si128((__m128i *)(blo + i), rlo);
            _mm_storeu_si128((__m128i *)(bhi + i), _mm_andnot_si128(fl, mhi));
        }
    }
    if (i < count)
        exr_fpnge_lookup_scalar(t, src + i, count - i, nb + i, blo + i, bhi + i);
}

/* AVX2 variant: 32 bytes/iter via _mm256_shuffle_epi8 (per-128-bit-lane PSHUFB
 * with the 16-entry tables broadcast to both lanes). */
EXR_TARGET("avx2")
void exr_fpnge_lookup_avx2(const exr_fpnge_table *t, const uint8_t *src,
                           size_t count, uint8_t *nb, uint8_t *blo,
                           uint8_t *bhi) {
    static const uint8_t kRevNib[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
    size_t i = 0;
    if (t->mid_nbits == 12) {
        const __m256i f16n = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)t->first16_nbits));
        const __m256i f16b = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)t->first16_bits));
        const __m256i l16n = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)t->last16_nbits));
        const __m256i l16b = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)t->last16_bits));
        const __m256i midl = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)t->mid_lowbits));
        const __m256i rev = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((const __m128i *)kRevNib));
        const __m256i nib = _mm256_set1_epi8(0x0F);
        const __m256i midn = _mm256_set1_epi8((char)t->mid_nbits);
        const __m256i zero = _mm256_setzero_si256();
        const __m256i ones = _mm256_set1_epi8((char)0xFF);
        const __m256i c15 = _mm256_set1_epi8(15);
        const __m256i c239 = _mm256_set1_epi8((char)239);
        for (; i + 32 <= count; i += 32) {
            __m256i b = _mm256_loadu_si256((const __m256i *)(src + i));
            __m256i lo = _mm256_and_si256(b, nib);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), nib);
            __m256i fnb = _mm256_shuffle_epi8(f16n, lo);
            __m256i fb = _mm256_shuffle_epi8(f16b, lo);
            __m256i lnb = _mm256_shuffle_epi8(l16n, lo);
            __m256i lb = _mm256_shuffle_epi8(l16b, lo);
            __m256i mlo = _mm256_shuffle_epi8(midl, hi);
            __m256i mhi = _mm256_shuffle_epi8(rev, lo);
            __m256i is_first = _mm256_cmpeq_epi8(_mm256_subs_epu8(b, c15), zero);
            __m256i is_last = _mm256_xor_si256(
                _mm256_cmpeq_epi8(_mm256_subs_epu8(b, c239), zero), ones);
            __m256i fl = _mm256_or_si256(is_first, is_last);
            __m256i rnb = _mm256_blendv_epi8(midn, fnb, is_first);
            __m256i rlo = _mm256_blendv_epi8(mlo, fb, is_first);
            rnb = _mm256_blendv_epi8(rnb, lnb, is_last);
            rlo = _mm256_blendv_epi8(rlo, lb, is_last);
            _mm256_storeu_si256((__m256i *)(nb + i), rnb);
            _mm256_storeu_si256((__m256i *)(blo + i), rlo);
            _mm256_storeu_si256((__m256i *)(bhi + i), _mm256_andnot_si256(fl, mhi));
        }
    }
    if (i < count)
        exr_fpnge_lookup_scalar(t, src + i, count - i, nb + i, blo + i, bhi + i);
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
