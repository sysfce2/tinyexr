/*
 * TinyEXR texcomp - shared internal declarations across the per-codec
 * translation units (texcomp.c + texcomp_<codec>.c).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef TINYEXR_TEXCOMP_INTERNAL_H_
#define TINYEXR_TEXCOMP_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

/* --- architecture / SIMD feature detection (mirrors texcomp.c) ---------- */
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define TC_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TC_TARGET(x) __attribute__((target(x)))
#else
#define TC_TARGET(x)
#endif

#define TC_CPU_SSE2 1u
#define TC_CPU_SSE41 2u
#define TC_CPU_AVX2 4u
#define TC_CPU_NEON 8u
#define TC_CPU_SVE 16u

/* --- helpers shared across translation units ---------------------------- */
/* Defined in texcomp.c (common core). */
uint32_t tc_cpu_caps(void);
void tc_set_bits(uint8_t *dst, uint32_t *bitpos, uint32_t val, uint32_t nbits);
uint32_t tc_luma_u8(const uint8_t *p);
int32_t tc_clamp_i32(int32_t v, int32_t lo, int32_t hi);
uint8_t tc_clamp_u8_i32(int32_t v);
void tc_wr_u64(uint8_t *p, uint64_t v);
void tc_wr_u24(uint8_t *p, uint32_t v);
uint32_t tc_bswap32(uint32_t v);
uint64_t tc_bswap64(uint64_t v);
int tc_astc_valid_block(uint32_t bx, uint32_t by);
int tc_mul_ovf_size(size_t a, size_t b, size_t *out);

/* Shared 4-bit interpolation weights (defined in texcomp_bc7.c). */
extern const uint32_t tc_bc7_weights4[16];

/* Defined in texcomp_bc5.c / texcomp_bc1.c, reused by texcomp_bc3.c. */
void tc_encode_bc4_block(const uint8_t v[16], uint8_t out[8]);
void tc_encode_bc1_color_block(const uint8_t px[16][4], int dxt1, uint8_t out[8]);

/* EAC alpha block (defined in texcomp_eac.c), reused by texcomp_etc2.c. */
uint64_t tc_encode_eac_alpha(const uint8_t alpha[16]);

#endif /* TINYEXR_TEXCOMP_INTERNAL_H_ */
