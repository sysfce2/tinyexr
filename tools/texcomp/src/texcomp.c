/*
 * TinyEXR texcomp - BC7/DDS implementation.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "texcomp.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

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

static void tc_set_bits(uint8_t *dst, uint32_t *bitpos, uint32_t val,
                        uint32_t nbits);

static const uint32_t tc_bc7_weights4[16] = {0,  4,  9,  13, 17, 21, 26, 30,
                                             34, 38, 43, 47, 51, 55, 60, 64};
static const uint32_t tc_bc7_weights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
static const uint32_t tc_bc7_weights2[4] = {0, 21, 43, 64};

static const uint8_t tc_bc7_num_subsets[8] = {3, 2, 3, 2, 1, 1, 1, 2};
static const uint8_t tc_bc7_partition_bits[8] = {4, 6, 6, 6, 0, 0, 0, 6};
static const uint8_t tc_bc7_color_index_bits[8] = {3, 3, 2, 2, 2, 2, 4, 2};
static const uint8_t tc_bc7_alpha_index_bits[8] = {0, 0, 0, 0, 3, 2, 4, 2};
static const uint8_t tc_bc7_color_precision[8] = {4, 6, 5, 7, 5, 7, 7, 5};
static const uint8_t tc_bc7_alpha_precision[8] = {0, 0, 0, 0, 6, 8, 7, 5};
static const uint8_t tc_bc7_has_pbits[8] = {1, 1, 0, 1, 0, 0, 1, 1};
static const uint8_t tc_bc7_shared_pbits[8] = {0, 1, 0, 0, 0, 0, 0, 0};
static const uint8_t tc_bc7_sep_alpha[8] = {0, 0, 0, 0, 1, 1, 0, 0};

static const uint8_t tc_bc7_partition1[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t tc_bc7_partition2[64 * 16] = {
    0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,    0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,    0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,    0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,    0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,    0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,    0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,    0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
    0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,    0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,    0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,    0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,    0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,    0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,    0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
    0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,    0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,    0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,    0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,    0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,    0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,    0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,    0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,    0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,    0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,    0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,    0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,    0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,    0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
    0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,    0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,    0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,    0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,    0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,    0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,    0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
    0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,    0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,    0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,    0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,    0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,    0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,    0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
    0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,    0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,    0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,    0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,    0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,    0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,    0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,    0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};
static const uint8_t tc_bc7_partition3_0[16] = {0, 0, 1, 1, 0, 0, 1, 1,
                                                0, 2, 2, 1, 2, 2, 2, 2};
static const uint8_t tc_bc7_anchor2[64] = {
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,15,15,15,15,15, 2, 2,15
};
static const uint8_t tc_bc7_anchor3_0[2] = {3, 15};

static const uint8_t tc_bc7_part_lut[8][21] = {
    {5, 9, 12, 27, 28, 47, 49, 51, 53},
    {34, 38, 39, 44},
    {2, 26, 32, 37, 46, 63},
    {16, 21, 25, 31, 43, 55, 59},
    {14, 29, 33, 36, 45, 60},
    {17, 19, 23, 30, 40, 54, 57},
    {48, 52, 56, 58},
    {0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 15, 18, 20, 22, 24, 35, 41, 42, 50, 61, 62}
};
static const uint8_t tc_bc7_alpha_part_lut[8][12] = {
    {4, 7, 8, 11, 52, 60, 63},
    {27, 28, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43},
    {0, 1, 2, 3, 23, 32, 39, 45, 58, 59},
    {18, 21, 22, 25, 54, 62},
    {10, 13, 14, 15, 16, 33, 38, 46, 56, 57},
    {17, 19, 20, 24, 55, 61},
    {5, 6, 9, 12, 53},
    {26, 29, 44, 47, 48, 49, 50, 51}
};

static const int32_t tc_eac_alpha[16][8] = {
    {-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12},
    {-2, -5, -8, -13, 1, 4, 7, 12}, {-2, -4, -6, -13, 1, 3, 5, 12},
    {-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10},
    {-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10},
    {-2, -6, -8, -10, 1, 5, 7, 9},  {-2, -5, -8, -10, 1, 4, 7, 9},
    {-2, -4, -8, -10, 1, 3, 7, 9},  {-2, -5, -7, -10, 1, 4, 6, 9},
    {-3, -4, -7, -10, 2, 3, 6, 9},  {-1, -2, -3, -10, 0, 1, 2, 9},
    {-4, -6, -8, -9, 3, 5, 7, 8},   {-3, -5, -7, -9, 2, 4, 6, 8}};

static const int32_t tc_etc1_mod[8][4] = {
    {2, 8, -2, -8},     {5, 17, -5, -17},    {9, 29, -9, -29},
    {13, 42, -13, -42}, {18, 60, -18, -60},  {24, 80, -24, -80},
    {33, 106, -33, -106}, {47, 183, -47, -183}};

static const int32_t tc_eac_alpha_range[16] = {
    0x100FF / (1 + 14 - -15), 0x100FF / (1 + 12 - -13),
    0x100FF / (1 + 12 - -13), 0x100FF / (1 + 12 - -13),
    0x100FF / (1 + 11 - -12), 0x100FF / (1 + 10 - -11),
    0x100FF / (1 + 10 - -11), 0x100FF / (1 + 10 - -11),
    0x100FF / (1 + 9 - -10),  0x100FF / (1 + 9 - -10),
    0x100FF / (1 + 9 - -10),  0x100FF / (1 + 9 - -10),
    0x100FF / (1 + 9 - -10),  0x100FF / (1 + 9 - -10),
    0x100FF / (1 + 8 - -9),   0x100FF / (1 + 8 - -9)};

static const uint32_t tc_etc2_planar_flags[64] = {
    0x80800402u, 0x80800402u, 0x80800402u, 0x80800402u,
    0x80800402u, 0x80800402u, 0x80800402u, 0x8080e002u,
    0x80800402u, 0x80800402u, 0x8080e002u, 0x8080e002u,
    0x80800402u, 0x8080e002u, 0x8080e002u, 0x8080e002u,
    0x80000402u, 0x80000402u, 0x80000402u, 0x80000402u,
    0x80000402u, 0x80000402u, 0x80000402u, 0x8000e002u,
    0x80000402u, 0x80000402u, 0x8000e002u, 0x8000e002u,
    0x80000402u, 0x8000e002u, 0x8000e002u, 0x8000e002u,
    0x00800402u, 0x00800402u, 0x00800402u, 0x00800402u,
    0x00800402u, 0x00800402u, 0x00800402u, 0x0080e002u,
    0x00800402u, 0x00800402u, 0x0080e002u, 0x0080e002u,
    0x00800402u, 0x0080e002u, 0x0080e002u, 0x0080e002u,
    0x00000402u, 0x00000402u, 0x00000402u, 0x00000402u,
    0x00000402u, 0x00000402u, 0x00000402u, 0x0000e002u,
    0x00000402u, 0x00000402u, 0x0000e002u, 0x0000e002u,
    0x00000402u, 0x0000e002u, 0x0000e002u, 0x0000e002u};

const char *tc_result_string(tc_result r) {
    switch (r) {
        case TC_SUCCESS: return "success";
        case TC_ERROR_INVALID_ARGUMENT: return "invalid argument";
        case TC_ERROR_OUT_OF_MEMORY: return "out of memory";
        case TC_ERROR_IO: return "I/O error";
        case TC_ERROR_UNSUPPORTED: return "unsupported";
        case TC_ERROR_CORRUPT: return "corrupt data";
        default: return "unknown error";
    }
}

#if defined(TC_X86)
#if defined(_MSC_VER)
static void tc_cpuidex(int out[4], int leaf, int subleaf) {
    __cpuidex(out, leaf, subleaf);
}
static unsigned long long tc_xgetbv0(void) { return _xgetbv(0); }
#else
static void tc_cpuidex(int out[4], int leaf, int subleaf) {
    unsigned int a, b, c, d;
    if (!__get_cpuid_count((unsigned int)leaf, (unsigned int)subleaf, &a, &b, &c, &d)) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    out[0] = (int)a;
    out[1] = (int)b;
    out[2] = (int)c;
    out[3] = (int)d;
}
static unsigned long long tc_xgetbv0(void) {
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#endif

static uint32_t tc_detect_cpu_caps(void) {
    int r[4];
    int max_leaf, os_ymm = 0;
    uint32_t caps = 0;
    tc_cpuidex(r, 0, 0);
    max_leaf = r[0];
    if (max_leaf < 1) return 0;
    tc_cpuidex(r, 1, 0);
    if (r[3] & (1 << 26)) caps |= TC_CPU_SSE2;
    if (r[2] & (1 << 19)) caps |= TC_CPU_SSE41;
    if ((r[2] & (1 << 27)) && (r[2] & (1 << 28)))
        os_ymm = (tc_xgetbv0() & 0x6) == 0x6;
    if (os_ymm && max_leaf >= 7) {
        tc_cpuidex(r, 7, 0);
        if (r[1] & (1 << 5)) caps |= TC_CPU_AVX2;
    }
    return caps;
}
#else
static uint32_t tc_detect_cpu_caps(void) {
    uint32_t caps = 0;
#if defined(__ARM_FEATURE_SVE)
    caps |= TC_CPU_SVE;
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    caps |= TC_CPU_NEON;
#endif
    return caps;
}
#endif

static uint32_t tc_backend_force = TC_BACKEND_ALL;

uint32_t tc_backend_available_mask(void) {
    static int ready = 0;
    static uint32_t caps = 0;
    if (!ready) {
        caps = tc_detect_cpu_caps();
        ready = 1;
    }
    return caps;
}

void tc_backend_force_mask(uint32_t mask) { tc_backend_force = mask; }

static uint32_t tc_cpu_caps(void) { return tc_backend_available_mask() & tc_backend_force; }

const char *tc_backend_name(void) {
    uint32_t caps = tc_cpu_caps();
    if (caps & TC_CPU_AVX2) return "avx2";
    if (caps & TC_CPU_SSE41) return "sse4.1";
    if (caps & TC_CPU_SSE2) return "sse2";
    if (caps & TC_CPU_SVE) return "sve";
    if (caps & TC_CPU_NEON) return "neon";
    return "scalar";
}

void tc_bc7_options_init(tc_bc7_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
    opt->quality = TC_BC7_QUALITY_QUICKBC7;
    opt->perceptual = 1;
    opt->quick = 1;
    opt->threads = 1;
    opt->mode_mask = 0xffu;
}

void tc_bc5_options_init(tc_bc5_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
}

void tc_bc6h_options_init(tc_bc6h_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
}

void tc_etc2_options_init(tc_etc2_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
    opt->alpha = 1;
}

void tc_astc_options_init(tc_astc_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
    opt->block_x = 6;
    opt->block_y = 6;
    opt->quality = 2;
}

static int tc_mul_ovf_size(size_t a, size_t b, size_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > SIZE_MAX / a) return 1;
    *out = a * b;
    return 0;
#endif
}

size_t tc_bc7_compressed_size(uint32_t width, uint32_t height) {
    size_t bx, by, blocks;
    if (!width || !height) return 0;
    bx = ((size_t)width + 3u) >> 2;
    by = ((size_t)height + 3u) >> 2;
    if (tc_mul_ovf_size(bx, by, &blocks)) return 0;
    if (blocks > SIZE_MAX / 16u) return 0;
    return blocks * 16u;
}

size_t tc_bc5_compressed_size(uint32_t width, uint32_t height) {
    return tc_bc7_compressed_size(width, height);
}

size_t tc_bc6h_compressed_size(uint32_t width, uint32_t height) {
    return tc_bc7_compressed_size(width, height);
}

size_t tc_etc2_rgb_compressed_size(uint32_t width, uint32_t height) {
    size_t bc = tc_bc7_compressed_size(width, height);
    return bc ? bc / 2u : 0u;
}

size_t tc_etc2_rgba_compressed_size(uint32_t width, uint32_t height) {
    return tc_bc7_compressed_size(width, height);
}

size_t tc_eac_r11_compressed_size(uint32_t width, uint32_t height) {
    return tc_etc2_rgb_compressed_size(width, height);
}

size_t tc_eac_rg11_compressed_size(uint32_t width, uint32_t height) {
    return tc_etc2_rgba_compressed_size(width, height);
}

static int tc_astc_valid_block(uint32_t bx, uint32_t by) {
    return (bx == 4u && by == 4u) || (bx == 5u && by == 4u) ||
           (bx == 5u && by == 5u) || (bx == 6u && by == 5u) ||
           (bx == 6u && by == 6u) || (bx == 8u && by == 5u) ||
           (bx == 8u && by == 6u) || (bx == 8u && by == 8u) ||
           (bx == 10u && by == 5u) || (bx == 10u && by == 6u) ||
           (bx == 10u && by == 8u) || (bx == 10u && by == 10u) ||
           (bx == 12u && by == 10u) || (bx == 12u && by == 12u);
}

size_t tc_astc_compressed_size(uint32_t width, uint32_t height,
                               const tc_astc_options *opt) {
    tc_astc_options defopt;
    size_t bx, by, blocks;
    if (!width || !height) return 0;
    if (!opt) {
        tc_astc_options_init(&defopt);
        opt = &defopt;
    }
    if (!tc_astc_valid_block(opt->block_x, opt->block_y)) return 0;
    bx = ((size_t)width + opt->block_x - 1u) / opt->block_x;
    by = ((size_t)height + opt->block_y - 1u) / opt->block_y;
    if (tc_mul_ovf_size(bx, by, &blocks)) return 0;
    if (blocks > SIZE_MAX / 16u) return 0;
    return blocks * 16u;
}

unsigned int tc_astc_ise_sequence_bitcount(unsigned int value_count,
                                           unsigned int quant_level) {
    static const uint8_t scale[21] = {1, 8, 2, 7, 13, 3, 10, 18, 4, 13, 23,
                                      5, 16, 28, 6, 19, 33, 7, 22, 38, 8};
    static const uint8_t div_code[21] = {0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                         0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    unsigned int divisor;
    if (quant_level >= 21u) return 1024u;
    divisor = (unsigned int)div_code[quant_level] * 2u + 1u;
    return ((unsigned int)scale[quant_level] * value_count + divisor - 1u) /
           divisor;
}

static const uint8_t tc_astc_integer_of_quints[5][5][5] = {
    {{0, 1, 2, 3, 4},
     {8, 9, 10, 11, 12},
     {16, 17, 18, 19, 20},
     {24, 25, 26, 27, 28},
     {5, 13, 21, 29, 6}},
    {{32, 33, 34, 35, 36},
     {40, 41, 42, 43, 44},
     {48, 49, 50, 51, 52},
     {56, 57, 58, 59, 60},
     {37, 45, 53, 61, 14}},
    {{64, 65, 66, 67, 68},
     {72, 73, 74, 75, 76},
     {80, 81, 82, 83, 84},
     {88, 89, 90, 91, 92},
     {69, 77, 85, 93, 22}},
    {{96, 97, 98, 99, 100},
     {104, 105, 106, 107, 108},
     {112, 113, 114, 115, 116},
     {120, 121, 122, 123, 124},
     {101, 109, 117, 125, 30}},
    {{102, 103, 70, 71, 38},
     {110, 111, 78, 79, 46},
     {118, 119, 86, 87, 54},
     {126, 127, 94, 95, 62},
     {39, 47, 55, 63, 31}}};

static const uint8_t tc_astc_integer_of_trits[3][3][3][3][3] = {
    {{{{0, 1, 2}, {4, 5, 6}, {8, 9, 10}},
      {{16, 17, 18}, {20, 21, 22}, {24, 25, 26}},
      {{3, 7, 15}, {19, 23, 27}, {12, 13, 14}}},
     {{{32, 33, 34}, {36, 37, 38}, {40, 41, 42}},
      {{48, 49, 50}, {52, 53, 54}, {56, 57, 58}},
      {{35, 39, 47}, {51, 55, 59}, {44, 45, 46}}},
     {{{64, 65, 66}, {68, 69, 70}, {72, 73, 74}},
      {{80, 81, 82}, {84, 85, 86}, {88, 89, 90}},
      {{67, 71, 79}, {83, 87, 91}, {76, 77, 78}}}},
    {{{{128, 129, 130}, {132, 133, 134}, {136, 137, 138}},
      {{144, 145, 146}, {148, 149, 150}, {152, 153, 154}},
      {{131, 135, 143}, {147, 151, 155}, {140, 141, 142}}},
     {{{160, 161, 162}, {164, 165, 166}, {168, 169, 170}},
      {{176, 177, 178}, {180, 181, 182}, {184, 185, 186}},
      {{163, 167, 175}, {179, 183, 187}, {172, 173, 174}}},
     {{{192, 193, 194}, {196, 197, 198}, {200, 201, 202}},
      {{208, 209, 210}, {212, 213, 214}, {216, 217, 218}},
      {{195, 199, 207}, {211, 215, 219}, {204, 205, 206}}}},
    {{{{96, 97, 98}, {100, 101, 102}, {104, 105, 106}},
      {{112, 113, 114}, {116, 117, 118}, {120, 121, 122}},
      {{99, 103, 111}, {115, 119, 123}, {108, 109, 110}}},
     {{{224, 225, 226}, {228, 229, 230}, {232, 233, 234}},
      {{240, 241, 242}, {244, 245, 246}, {248, 249, 250}},
      {{227, 231, 239}, {243, 247, 251}, {236, 237, 238}}},
     {{{28, 29, 30}, {60, 61, 62}, {92, 93, 94}},
      {{156, 157, 158}, {188, 189, 190}, {220, 221, 222}},
      {{31, 63, 127}, {159, 191, 255}, {252, 253, 254}}}}};

tc_result tc_astc_ise_encode_bits(unsigned int quant_level, unsigned int value_count,
                                  const uint8_t *values, uint8_t *out,
                                  size_t out_size, unsigned int bit_offset) {
    static const uint8_t bits[21] = {1, 0, 2, 0, 1, 3, 1, 2, 4, 2, 3,
                                     5, 3, 4, 6, 4, 5, 7, 5, 6, 8};
    static const uint8_t has_trit[21] = {
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
    static const uint8_t has_quint[21] = {
        0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    unsigned int i, nbits, total_bits;
    if (!values || !out || !value_count || quant_level >= 21u)
        return TC_ERROR_INVALID_ARGUMENT;
    nbits = bits[quant_level];
    total_bits = tc_astc_ise_sequence_bitcount(value_count, quant_level);
    if (((size_t)bit_offset + total_bits + 7u) / 8u > out_size)
        return TC_ERROR_INVALID_ARGUMENT;
    if (has_trit[quant_level]) {
        unsigned int mask = (1u << nbits) - 1u;
        i = 0;
        while (i + 4u < value_count) {
            uint8_t t = tc_astc_integer_of_trits[values[i + 4u] >> nbits]
                                                 [values[i + 3u] >> nbits]
                                                 [values[i + 2u] >> nbits]
                                                 [values[i + 1u] >> nbits]
                                                 [values[i] >> nbits];
            uint32_t pack = (values[i++] & mask) | (((uint32_t)t & 0x3u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 2u);
            pack = (values[i++] & mask) | ((((uint32_t)t >> 2) & 0x3u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 2u);
            pack = (values[i++] & mask) | ((((uint32_t)t >> 4) & 0x1u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 1u);
            pack = (values[i++] & mask) | ((((uint32_t)t >> 5) & 0x3u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 2u);
            pack = (values[i++] & mask) | ((((uint32_t)t >> 7) & 0x1u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 1u);
        }
        if (i != value_count) {
            unsigned int i4 = 0;
            unsigned int i3 = i + 3u >= value_count ? 0 : values[i + 3u] >> nbits;
            unsigned int i2 = i + 2u >= value_count ? 0 : values[i + 2u] >> nbits;
            unsigned int i1 = i + 1u >= value_count ? 0 : values[i + 1u] >> nbits;
            unsigned int i0 = values[i] >> nbits;
            uint8_t t = tc_astc_integer_of_trits[i4][i3][i2][i1][i0];
            unsigned int j = 0;
            static const uint8_t tbits[4] = {2, 2, 1, 2};
            static const uint8_t tshift[4] = {0, 2, 4, 5};
            while (i < value_count) {
                uint32_t pack = (values[i++] & mask) |
                                ((((uint32_t)t >> tshift[j]) &
                                  ((1u << tbits[j]) - 1u))
                                 << nbits);
                tc_set_bits(out, &bit_offset, pack, nbits + tbits[j]);
                ++j;
            }
        }
        return TC_SUCCESS;
    }
    if (has_quint[quant_level]) {
        unsigned int mask = (1u << nbits) - 1u;
        i = 0;
        while (i + 2u < value_count) {
            uint8_t q = tc_astc_integer_of_quints[values[i + 2u] >> nbits]
                                                  [values[i + 1u] >> nbits]
                                                  [values[i] >> nbits];
            uint32_t pack = (values[i++] & mask) | (((uint32_t)q & 0x7u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 3u);
            pack = (values[i++] & mask) | ((((uint32_t)q >> 3) & 0x3u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 2u);
            pack = (values[i++] & mask) | ((((uint32_t)q >> 5) & 0x3u) << nbits);
            tc_set_bits(out, &bit_offset, pack, nbits + 2u);
        }
        if (i != value_count) {
            unsigned int i2 = 0;
            unsigned int i1 = i + 1u >= value_count ? 0 : values[i + 1u] >> nbits;
            unsigned int i0 = values[i] >> nbits;
            uint8_t q = tc_astc_integer_of_quints[i2][i1][i0];
            unsigned int j = 0;
            static const uint8_t tbits[2] = {3, 2};
            static const uint8_t tshift[2] = {0, 3};
            while (i < value_count) {
                uint32_t pack = (values[i++] & mask) |
                                ((((uint32_t)q >> tshift[j]) &
                                  ((1u << tbits[j]) - 1u))
                                 << nbits);
                tc_set_bits(out, &bit_offset, pack, nbits + tbits[j]);
                ++j;
            }
        }
        return TC_SUCCESS;
    }
    for (i = 0; i < value_count; ++i) {
        tc_set_bits(out, &bit_offset, values[i], nbits);
    }
    return TC_SUCCESS;
}

static void tc_wr_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void tc_wr_u64(uint8_t *p, uint64_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

static void tc_wr_u24(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
}

static uint32_t tc_bswap32(uint32_t v) {
    return ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
           ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
}

static uint64_t tc_bswap64(uint64_t v) {
    return ((v & 0x00000000000000ffULL) << 56) |
           ((v & 0x000000000000ff00ULL) << 40) |
           ((v & 0x0000000000ff0000ULL) << 24) |
           ((v & 0x00000000ff000000ULL) << 8) |
           ((v & 0x000000ff00000000ULL) >> 8) |
           ((v & 0x0000ff0000000000ULL) >> 24) |
           ((v & 0x00ff000000000000ULL) >> 40) |
           ((v & 0xff00000000000000ULL) >> 56);
}

static uint64_t tc_etc2_fix_byte_order(uint64_t d) {
    return (d & 0x00000000ffffffffULL) |
           ((d & 0xff00000000000000ULL) >> 24) |
           ((d & 0x000000ff00000000ULL) << 24) |
           ((d & 0x00ff000000000000ULL) >> 8) |
           ((d & 0x0000ff0000000000ULL) << 8);
}

static int32_t tc_clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static uint8_t tc_clamp_u8_i32(int32_t v) {
    return (uint8_t)tc_clamp_i32(v, 0, 255);
}

static void tc_set_bits(uint8_t *dst, uint32_t *bitpos, uint32_t val,
                        uint32_t nbits) {
    uint32_t n;
    while (nbits) {
        n = 8u - (*bitpos & 7u);
        if (n > nbits) n = nbits;
        dst[*bitpos >> 3] |= (uint8_t)((val & ((1u << n) - 1u)) << (*bitpos & 7u));
        val >>= n;
        *bitpos += n;
        nbits -= n;
    }
}

static uint8_t tc_unquant(uint32_t q, uint32_t bits) {
    uint32_t maxv;
    if (bits >= 8u) return (uint8_t)q;
    maxv = (1u << bits) - 1u;
    return (uint8_t)((q * 255u + maxv / 2u) / maxv);
}

static uint32_t tc_quant(uint32_t v, uint32_t bits) {
    uint32_t maxv = (1u << bits) - 1u;
    return (v * maxv + 127u) / 255u;
}

static uint32_t tc_luma_u8(const uint8_t *p) {
    return (uint32_t)p[0] * 38u + (uint32_t)p[1] * 76u + (uint32_t)p[2] * 14u;
}

static void tc_encode_bc4_block(const uint8_t v[16], uint8_t out[8]) {
    uint8_t minv = 255, maxv = 0, palette[8];
    uint64_t bits = 0;
    uint32_t i;
    for (i = 0; i < 16u; ++i) {
        if (v[i] < minv) minv = v[i];
        if (v[i] > maxv) maxv = v[i];
    }
    out[0] = maxv;
    out[1] = minv;
    palette[0] = maxv;
    palette[1] = minv;
    for (i = 1; i <= 6u; ++i)
        palette[i + 1u] = (uint8_t)(((7u - i) * maxv + i * minv + 3u) / 7u);
    for (i = 0; i < 16u; ++i) {
        uint32_t j, best = 0, best_err = UINT_MAX;
        for (j = 0; j < 8u; ++j) {
            int d = (int)v[i] - (int)palette[j];
            uint32_t e = (uint32_t)(d * d);
            if (e < best_err) {
                best_err = e;
                best = j;
            }
        }
        bits |= (uint64_t)best << (3u * i);
    }
    for (i = 0; i < 6u; ++i) out[2u + i] = (uint8_t)(bits >> (8u * i));
}

static void tc_encode_bc5_block(const uint8_t block[16][2], uint8_t out[16]) {
    uint8_t r[16], g[16];
    uint32_t i;
    for (i = 0; i < 16u; ++i) {
        r[i] = block[i][0];
        g[i] = block[i][1];
    }
    tc_encode_bc4_block(r, out);
    tc_encode_bc4_block(g, out + 8);
}

static uint16_t tc_float_to_half_bits(float fv) {
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign, mant, exp;
    v.f = fv;
    sign = (v.u >> 16) & 0x8000u;
    exp = (v.u >> 23) & 0xffu;
    mant = v.u & 0x7fffffu;
    if (exp == 255u) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    if (exp > 142u) return (uint16_t)(sign | 0x7c00u);
    if (exp < 113u) {
        uint32_t m;
        if (exp < 103u) return (uint16_t)sign;
        m = mant | 0x800000u;
        m >>= 125u - exp;
        m = (m + 0x1000u) >> 13;
        return (uint16_t)(sign | m);
    }
    exp = exp - 112u;
    mant = (mant + 0x1000u) >> 13;
    if (mant & 0x400u) {
        mant = 0;
        ++exp;
    }
    if (exp >= 31u) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3ffu));
}

static uint32_t tc_bc6h_quant_uf16(float f) {
    uint16_t h;
    uint32_t mag;
    if (!(f > 0.0f)) return 0;
    h = tc_float_to_half_bits(f);
    mag = h & 0x7fffu;
    if (mag > 0x7bffu) mag = 0x7bffu;
    return (mag * 1023u + 15871u) / 31743u;
}

static int32_t tc_bc6h_quant_sf16(float f) {
    uint16_t h;
    uint32_t mag;
    int32_t q;
    if (f == 0.0f) return 0;
    h = tc_float_to_half_bits(f);
    mag = h & 0x7fffu;
    if (mag > 0x7bffu) mag = 0x7bffu;
    q = (int32_t)((mag * 511u + 15871u) / 31743u);
    if (q > 511) q = 511;
    return (h & 0x8000u) ? -q : q;
}

static uint32_t tc_bc6h_unquant_uf16_to_mag(uint32_t q) {
    uint32_t unq;
    if (q == 0u) unq = 0u;
    else if (q >= 1023u) unq = 0xffffu;
    else unq = ((q << 16) + 0x8000u) >> 10;
    return (unq * 31u) >> 6;
}

static int32_t tc_bc6h_unquant_sf16_to_smag(int32_t q) {
    int32_t sign = 0, unq;
    if (q < 0) {
        sign = 1;
        q = -q;
    }
    if (q == 0) unq = 0;
    else if (q >= 511) unq = 0x7fff;
    else unq = (int32_t)(((uint32_t)q << 15) + 0x4000u) >> 9;
    if (sign) unq = -unq;
    if (unq < 0) return -(((-unq) * 31) >> 5);
    return (unq * 31) >> 5;
}

static uint32_t tc_bc6h_err3_mag(const uint32_t a[3], uint32_t r, uint32_t g,
                                 uint32_t b) {
    int32_t dr = (int32_t)a[0] - (int32_t)r;
    int32_t dg = (int32_t)a[1] - (int32_t)g;
    int32_t db = (int32_t)a[2] - (int32_t)b;
    return (uint32_t)(dr * dr + dg * dg + db * db);
}

static uint32_t tc_bc6h_err3_smag(const int32_t a[3], int32_t r, int32_t g,
                                  int32_t b) {
    int64_t dr = (int64_t)a[0] - (int64_t)r;
    int64_t dg = (int64_t)a[1] - (int64_t)g;
    int64_t db = (int64_t)a[2] - (int64_t)b;
    uint64_t e = (uint64_t)(dr * dr + dg * dg + db * db);
    return e > UINT_MAX ? UINT_MAX : (uint32_t)e;
}

static uint32_t tc_bc6h_pack_signed10(int32_t q) {
    if (q < -512) q = -512;
    if (q > 511) q = 511;
    return (uint32_t)q & 1023u;
}

static uint64_t tc_bc6h_choose_selectors_uf16(const uint32_t target[16][3],
                                              const uint32_t lo[3],
                                              const uint32_t hi[3],
                                              uint8_t sel[16]) {
    uint32_t pal[16][3];
    uint64_t err = 0;
    uint32_t i, c, s;
    for (s = 0; s < 16u; ++s) {
        uint32_t w = tc_bc7_weights4[s];
        for (c = 0; c < 3u; ++c) {
            uint32_t qv = ((64u - w) * lo[c] + w * hi[c] + 32u) >> 6;
            pal[s][c] = tc_bc6h_unquant_uf16_to_mag(qv);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t best = 0, best_err = UINT_MAX;
        for (s = 0; s < 16u; ++s) {
            uint32_t e = tc_bc6h_err3_mag(target[i], pal[s][0], pal[s][1], pal[s][2]);
            if (e < best_err) {
                best_err = e;
                best = s;
            }
        }
        sel[i] = (uint8_t)best;
        err += best_err;
    }
    return err;
}

static uint64_t tc_bc6h_choose_selectors_sf16(const int32_t target[16][3],
                                              const int32_t lo[3],
                                              const int32_t hi[3],
                                              uint8_t sel[16]) {
    int32_t pal[16][3];
    uint64_t err = 0;
    uint32_t i, c, s;
    for (s = 0; s < 16u; ++s) {
        uint32_t w = tc_bc7_weights4[s];
        for (c = 0; c < 3u; ++c) {
            int32_t qv = (int32_t)(((int64_t)(64u - w) * lo[c] +
                                    (int64_t)w * hi[c] + 32) >>
                                   6);
            pal[s][c] = tc_bc6h_unquant_sf16_to_smag(qv);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t best = 0, best_err = UINT_MAX;
        for (s = 0; s < 16u; ++s) {
            uint32_t e = tc_bc6h_err3_smag(target[i], pal[s][0], pal[s][1], pal[s][2]);
            if (e < best_err) {
                best_err = e;
                best = s;
            }
        }
        sel[i] = (uint8_t)best;
        err += best_err;
    }
    return err;
}

static void tc_encode_bc6h_block_uf16(const float pix[16][3], uint8_t out[16]) {
    uint32_t q[16][3], target[16][3], lo[3], hi[3], luma_lo[3], luma_hi[3];
    uint32_t i, c, bitpos = 0, min_l = UINT_MAX, max_l = 0, min_i = 0, max_i = 0;
    uint8_t sel[16], box_sel[16], luma_sel[16];
    memset(out, 0, 16);
    for (c = 0; c < 3u; ++c) {
        lo[c] = UINT_MAX;
        hi[c] = 0;
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t l;
        for (c = 0; c < 3u; ++c) {
            q[i][c] = tc_bc6h_quant_uf16(pix[i][c]);
            target[i][c] = tc_bc6h_unquant_uf16_to_mag(q[i][c]);
            if (q[i][c] < lo[c]) lo[c] = q[i][c];
            if (q[i][c] > hi[c]) hi[c] = q[i][c];
        }
        l = target[i][0] * 38u + target[i][1] * 76u + target[i][2] * 14u;
        if (l < min_l) {
            min_l = l;
            min_i = i;
        }
        if (l >= max_l) {
            max_l = l;
            max_i = i;
        }
    }
    for (c = 0; c < 3u; ++c) {
        luma_lo[c] = q[min_i][c];
        luma_hi[c] = q[max_i][c];
    }
    if (tc_bc6h_choose_selectors_uf16(target, luma_lo, luma_hi, luma_sel) <
        tc_bc6h_choose_selectors_uf16(target, lo, hi, box_sel)) {
        memcpy(lo, luma_lo, sizeof(lo));
        memcpy(hi, luma_hi, sizeof(hi));
        memcpy(sel, luma_sel, sizeof(sel));
    } else {
        memcpy(sel, box_sel, sizeof(sel));
    }
    if (sel[0] & 8u) {
        uint32_t t;
        for (c = 0; c < 3u; ++c) {
            t = lo[c];
            lo[c] = hi[c];
            hi[c] = t;
        }
        for (i = 0; i < 16u; ++i) sel[i] = (uint8_t)(15u - sel[i]);
    }

    tc_set_bits(out, &bitpos, 0x03u, 5); /* BC6H mode 11: bit pattern 00011 */
    tc_set_bits(out, &bitpos, lo[0], 10);
    tc_set_bits(out, &bitpos, lo[1], 10);
    tc_set_bits(out, &bitpos, lo[2], 10);
    tc_set_bits(out, &bitpos, hi[0], 10);
    tc_set_bits(out, &bitpos, hi[1], 10);
    tc_set_bits(out, &bitpos, hi[2], 10);
    tc_set_bits(out, &bitpos, sel[0], 3);
    for (i = 1; i < 16u; ++i) tc_set_bits(out, &bitpos, sel[i], 4);
}

static void tc_encode_bc6h_block_sf16(const float pix[16][3], uint8_t out[16]) {
    int32_t q[16][3], target[16][3], lo[3], hi[3], luma_lo[3], luma_hi[3];
    uint32_t i, c, bitpos = 0, min_i = 0, max_i = 0;
    int32_t min_l = INT_MAX, max_l = INT_MIN;
    uint8_t sel[16], box_sel[16], luma_sel[16];
    memset(out, 0, 16);
    for (c = 0; c < 3u; ++c) {
        lo[c] = INT_MAX;
        hi[c] = INT_MIN;
    }
    for (i = 0; i < 16u; ++i) {
        int32_t l;
        for (c = 0; c < 3u; ++c) {
            q[i][c] = tc_bc6h_quant_sf16(pix[i][c]);
            target[i][c] = tc_bc6h_unquant_sf16_to_smag(q[i][c]);
            if (q[i][c] < lo[c]) lo[c] = q[i][c];
            if (q[i][c] > hi[c]) hi[c] = q[i][c];
        }
        l = target[i][0] * 38 + target[i][1] * 76 + target[i][2] * 14;
        if (l < min_l) {
            min_l = l;
            min_i = i;
        }
        if (l >= max_l) {
            max_l = l;
            max_i = i;
        }
    }
    for (c = 0; c < 3u; ++c) {
        luma_lo[c] = q[min_i][c];
        luma_hi[c] = q[max_i][c];
    }
    if (tc_bc6h_choose_selectors_sf16(target, luma_lo, luma_hi, luma_sel) <
        tc_bc6h_choose_selectors_sf16(target, lo, hi, box_sel)) {
        memcpy(lo, luma_lo, sizeof(lo));
        memcpy(hi, luma_hi, sizeof(hi));
        memcpy(sel, luma_sel, sizeof(sel));
    } else {
        memcpy(sel, box_sel, sizeof(sel));
    }
    if (sel[0] & 8u) {
        int32_t t;
        for (c = 0; c < 3u; ++c) {
            t = lo[c];
            lo[c] = hi[c];
            hi[c] = t;
        }
        for (i = 0; i < 16u; ++i) sel[i] = (uint8_t)(15u - sel[i]);
    }

    tc_set_bits(out, &bitpos, 0x03u, 5);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[0]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[1]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[2]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[0]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[1]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[2]), 10);
    tc_set_bits(out, &bitpos, sel[0], 3);
    for (i = 1; i < 16u; ++i) tc_set_bits(out, &bitpos, sel[i], 4);
}

static uint32_t tc_block_quick_mask(const uint8_t pix[16][4], uint32_t user_mask) {
    uint32_t i;
    uint32_t lmin = UINT_MAX, lmax = 0, amin = UINT_MAX, amax = 0;
    uint32_t has_alpha = 0, lstate, astate, mask = 0;
    for (i = 0; i < 16u; ++i) {
        uint32_t y = tc_luma_u8(pix[i]) >> 7;
        uint32_t a = pix[i][3];
        if (y < lmin) lmin = y;
        if (y > lmax) lmax = y;
        if (a < amin) amin = a;
        if (a > amax) amax = a;
        if (a < 255u) has_alpha = 1;
    }

    lstate = (lmax - lmin <= 19u) ? 0u : ((lmax - lmin <= 48u) ? 1u : 2u);
    if (!has_alpha) {
        if (lstate == 0u) mask = 1u << 6;
        else if (lstate == 1u) mask = (1u << 1) | (1u << 6);
        else mask = 1u << 1;
    } else {
        uint32_t adiff = amax - amin;
        astate = (adiff < 10u) ? 0u : ((adiff <= 21u) ? 1u : 2u);
        if (lstate <= 1u && astate <= 1u) mask |= 1u << 6;
        if (lstate <= 1u && astate >= 1u) mask |= 1u << 5;
        if (lstate >= 1u) mask |= 1u << 7;
        if (!mask) mask = 1u << 6;
    }

    mask &= user_mask;
    return mask ? mask : user_mask;
}

static uint32_t tc_err4(const uint8_t *a, uint8_t r, uint8_t g, uint8_t b,
                        uint8_t al, int has_alpha) {
    int dr = (int)a[0] - (int)r;
    int dg = (int)a[1] - (int)g;
    int db = (int)a[2] - (int)b;
    int da = has_alpha ? ((int)a[3] - (int)al) : 0;
    return (uint32_t)(dr * dr + dg * dg + db * db + da * da);
}

static uint32_t tc_err3(const uint8_t *a, uint8_t r, uint8_t g, uint8_t b) {
    int dr = (int)a[0] - (int)r;
    int dg = (int)a[1] - (int)g;
    int db = (int)a[2] - (int)b;
    return (uint32_t)(dr * dr + dg * dg + db * db);
}

static uint32_t tc_err1(uint8_t a, uint8_t b) {
    int d = (int)a - (int)b;
    return (uint32_t)(d * d);
}

typedef struct tc_bc7_candidate {
    uint8_t mode;
    uint8_t partition;
    uint8_t index_selector;
    uint8_t rotation;
    uint8_t lo[3][4];
    uint8_t hi[3][4];
    uint8_t pbits[3][2];
    uint8_t selectors[16];
    uint8_t alpha_selectors[16];
} tc_bc7_candidate;

static const uint8_t *tc_partition_for(uint32_t mode, uint32_t partition) {
    if (tc_bc7_num_subsets[mode] == 1u) return tc_bc7_partition1;
    if (tc_bc7_num_subsets[mode] == 2u)
        return &tc_bc7_partition2[(partition & 63u) * 16u];
    return tc_bc7_partition3_0;
}

static uint8_t tc_anchor_for_subset(uint32_t mode, uint32_t partition,
                                    uint32_t subset) {
    if (subset == 0u) return 0;
    if (tc_bc7_num_subsets[mode] == 3u) return tc_bc7_anchor3_0[subset - 1u];
    return tc_bc7_anchor2[partition & 63u];
}

static const uint32_t *tc_weights_for_bits(uint32_t bits) {
    if (bits == 2u) return tc_bc7_weights2;
    if (bits == 3u) return tc_bc7_weights3;
    return tc_bc7_weights4;
}

static uint8_t tc_decode_endpoint(uint8_t q, uint8_t p, uint32_t precision,
                                  uint32_t has_pbit) {
    if (has_pbit) return tc_unquant(((uint32_t)q << 1u) | p, precision + 1u);
    return tc_unquant(q, precision);
}

static void tc_quant_endpoint(uint8_t v, uint32_t precision, uint32_t has_pbit,
                              uint32_t shared_pbit, uint8_t shared_p,
                              uint8_t *q, uint8_t *p) {
    if (has_pbit) {
        uint32_t total_bits = precision + 1u;
        uint32_t u = tc_quant(v, total_bits);
        if (shared_pbit) {
            *p = shared_p;
            *q = (uint8_t)(u >> 1u);
        } else {
            *p = (uint8_t)(u & 1u);
            *q = (uint8_t)(u >> 1u);
        }
    } else {
        *p = 0;
        *q = (uint8_t)tc_quant(v, precision);
    }
}

static void tc_recon_color(const tc_bc7_candidate *cand, uint32_t mode,
                           uint32_t subset, uint32_t sel, uint32_t asel,
                           uint8_t out[4]) {
    uint32_t c;
    uint32_t cbits = tc_bc7_color_index_bits[mode] + cand->index_selector;
    uint32_t abits = tc_bc7_alpha_index_bits[mode] - cand->index_selector;
    const uint32_t *cw = tc_weights_for_bits(cbits);
    const uint32_t *aw = tc_weights_for_bits(abits ? abits : cbits);
    uint32_t w = cw[sel];
    uint32_t awv = aw[asel];

    for (c = 0; c < 3u; ++c) {
        uint32_t a = tc_decode_endpoint(cand->lo[subset][c], cand->pbits[subset][0],
                                        tc_bc7_color_precision[mode],
                                        tc_bc7_has_pbits[mode]);
        uint32_t b = tc_decode_endpoint(cand->hi[subset][c], cand->pbits[subset][1],
                                        tc_bc7_color_precision[mode],
                                        tc_bc7_has_pbits[mode]);
        out[c] = (uint8_t)(((64u - w) * a + w * b + 32u) >> 6);
    }
    if (mode < 4u) {
        out[3] = 255u;
    } else {
        uint32_t ap = tc_bc7_has_pbits[mode] && !tc_bc7_sep_alpha[mode];
        uint32_t a = tc_decode_endpoint(cand->lo[subset][3], cand->pbits[subset][0],
                                        tc_bc7_alpha_precision[mode], ap);
        uint32_t b = tc_decode_endpoint(cand->hi[subset][3], cand->pbits[subset][1],
                                        tc_bc7_alpha_precision[mode], ap);
        if (!tc_bc7_sep_alpha[mode]) awv = w;
        out[3] = (uint8_t)(((64u - awv) * a + awv * b + 32u) >> 6);
    }
}

static void tc_fill_palette(const tc_bc7_candidate *cand, uint32_t mode,
                            uint8_t pal[3][16][4]) {
    uint32_t subset, sel, max_sel = 1u << (tc_bc7_color_index_bits[mode] + cand->index_selector);
    uint32_t max_asel = tc_bc7_alpha_index_bits[mode] > cand->index_selector
                            ? (1u << (tc_bc7_alpha_index_bits[mode] - cand->index_selector))
                            : max_sel;
    uint32_t n = max_sel > max_asel ? max_sel : max_asel;
    for (subset = 0; subset < tc_bc7_num_subsets[mode]; ++subset) {
        for (sel = 0; sel < n; ++sel) {
            uint32_t cs = sel < max_sel ? sel : max_sel - 1u;
            uint32_t as = sel < max_asel ? sel : max_asel - 1u;
            tc_recon_color(cand, mode, subset, cs, as, pal[subset][sel]);
        }
    }
}

static void tc_pack_candidate(const tc_bc7_candidate *src, uint8_t out[16]) {
    tc_bc7_candidate cand = *src;
    const uint8_t *part = tc_partition_for(cand.mode, cand.partition);
    uint32_t subsets = tc_bc7_num_subsets[cand.mode];
    uint32_t bitpos = 0, subset, comp, i;
    uint8_t anchor[3] = {0, 0, 0};

    for (subset = 0; subset < subsets; ++subset) {
        uint32_t anchor_index = tc_anchor_for_subset(cand.mode, cand.partition, subset);
        uint32_t cidx_bits = tc_bc7_color_index_bits[cand.mode] + cand.index_selector;
        uint32_t ncolor = 1u << cidx_bits;
        anchor[subset] = (uint8_t)anchor_index;
        if (cand.selectors[anchor_index] & (ncolor >> 1u)) {
            uint8_t tq[4], tp;
            for (i = 0; i < 16u; ++i)
                if (part[i] == subset) cand.selectors[i] = (uint8_t)((ncolor - 1u) - cand.selectors[i]);
            for (comp = 0; comp < (tc_bc7_sep_alpha[cand.mode] ? 3u : 4u); ++comp) {
                tq[comp] = cand.lo[subset][comp];
                cand.lo[subset][comp] = cand.hi[subset][comp];
                cand.hi[subset][comp] = tq[comp];
            }
            if (!tc_bc7_shared_pbits[cand.mode]) {
                tp = cand.pbits[subset][0];
                cand.pbits[subset][0] = cand.pbits[subset][1];
                cand.pbits[subset][1] = tp;
            }
        }
        if (tc_bc7_sep_alpha[cand.mode]) {
            uint32_t aidx_bits = tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector;
            uint32_t nalpha = 1u << aidx_bits;
            if (cand.alpha_selectors[anchor_index] & (nalpha >> 1u)) {
                uint8_t tq = cand.lo[subset][3];
                for (i = 0; i < 16u; ++i)
                    if (part[i] == subset)
                        cand.alpha_selectors[i] = (uint8_t)((nalpha - 1u) - cand.alpha_selectors[i]);
                cand.lo[subset][3] = cand.hi[subset][3];
                cand.hi[subset][3] = tq;
            }
        }
    }

    memset(out, 0, 16);
    tc_set_bits(out, &bitpos, 1u << cand.mode, cand.mode + 1u);
    if (cand.mode == 4u || cand.mode == 5u) tc_set_bits(out, &bitpos, cand.rotation, 2);
    if (cand.mode == 4u) tc_set_bits(out, &bitpos, cand.index_selector, 1);
    if (tc_bc7_partition_bits[cand.mode])
        tc_set_bits(out, &bitpos, cand.partition, tc_bc7_partition_bits[cand.mode]);

    for (comp = 0; comp < (cand.mode >= 4u ? 4u : 3u); ++comp) {
        uint32_t prec = comp == 3u ? tc_bc7_alpha_precision[cand.mode]
                                   : tc_bc7_color_precision[cand.mode];
        for (subset = 0; subset < subsets; ++subset) {
            tc_set_bits(out, &bitpos, cand.lo[subset][comp], prec);
            tc_set_bits(out, &bitpos, cand.hi[subset][comp], prec);
        }
    }
    if (tc_bc7_has_pbits[cand.mode]) {
        for (subset = 0; subset < subsets; ++subset) {
            tc_set_bits(out, &bitpos, cand.pbits[subset][0], 1);
            if (!tc_bc7_shared_pbits[cand.mode]) tc_set_bits(out, &bitpos, cand.pbits[subset][1], 1);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t n = cand.index_selector ? (tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector)
                                         : (tc_bc7_color_index_bits[cand.mode] + cand.index_selector);
        if (i == anchor[0] || i == anchor[1] || i == anchor[2]) n--;
        tc_set_bits(out, &bitpos,
                    cand.index_selector ? cand.alpha_selectors[i] : cand.selectors[i], n);
    }
    if (tc_bc7_sep_alpha[cand.mode]) {
        for (i = 0; i < 16u; ++i) {
            uint32_t n = cand.index_selector ? (tc_bc7_color_index_bits[cand.mode] + cand.index_selector)
                                             : (tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector);
            if (i == anchor[0] || i == anchor[1] || i == anchor[2]) n--;
            tc_set_bits(out, &bitpos,
                        cand.index_selector ? cand.selectors[i] : cand.alpha_selectors[i], n);
        }
    }
}

static uint64_t tc_build_candidate(uint32_t mode, uint32_t partition,
                                   const uint8_t pix[16][4],
                                   tc_bc7_candidate *cand) {
    const uint8_t *part = tc_partition_for(mode, partition);
    uint32_t subsets = tc_bc7_num_subsets[mode];
    uint32_t subset, i, c;
    uint64_t total_err = 0;
    memset(cand, 0, sizeof(*cand));
    cand->mode = (uint8_t)mode;
    cand->partition = (uint8_t)partition;
    cand->index_selector = 0;
    cand->rotation = 0;

    for (subset = 0; subset < subsets; ++subset) {
        uint32_t min_l = UINT_MAX, max_l = 0, min_i = 0, max_i = 0;
        for (i = 0; i < 16u; ++i) {
            if (part[i] == subset) {
                uint32_t y = tc_luma_u8(pix[i]);
                if (y < min_l) {
                    min_l = y;
                    min_i = i;
                }
                if (y >= max_l) {
                    max_l = y;
                    max_i = i;
                }
            }
        }
        for (c = 0; c < 4u; ++c) {
            uint8_t qp0, qp1;
            uint32_t prec = c == 3u && mode >= 4u ? tc_bc7_alpha_precision[mode]
                                                   : tc_bc7_color_precision[mode];
            uint32_t has_p = tc_bc7_has_pbits[mode] && (c < 3u || !tc_bc7_sep_alpha[mode]);
            uint8_t shared = (uint8_t)(((pix[min_i][c] + pix[max_i][c]) >> 1u) & 1u);
            if (mode < 4u && c == 3u) {
                cand->lo[subset][c] = 0;
                cand->hi[subset][c] = 0;
                continue;
            }
            tc_quant_endpoint(pix[min_i][c], prec, has_p, tc_bc7_shared_pbits[mode],
                              shared, &cand->lo[subset][c], &qp0);
            tc_quant_endpoint(pix[max_i][c], prec, has_p, tc_bc7_shared_pbits[mode],
                              shared, &cand->hi[subset][c], &qp1);
            if (tc_bc7_shared_pbits[mode]) {
                cand->pbits[subset][0] = shared;
                cand->pbits[subset][1] = shared;
            } else if (has_p) {
                cand->pbits[subset][0] = qp0;
                cand->pbits[subset][1] = qp1;
            }
        }
    }

    {
        uint8_t pal[3][16][4];
        uint32_t cbits = tc_bc7_color_index_bits[mode] + cand->index_selector;
        uint32_t abits = tc_bc7_alpha_index_bits[mode] - cand->index_selector;
        uint32_t nc = 1u << cbits;
        uint32_t na = abits ? (1u << abits) : nc;
        tc_fill_palette(cand, mode, pal);
        for (i = 0; i < 16u; ++i) {
            uint32_t subset = part[i], s, best_s = 0, best_a = 0;
            uint32_t best = UINT_MAX;
            uint32_t best_color_err = 0;
            uint32_t best_alpha_err = 0;
            for (s = 0; s < nc; ++s) {
                const uint8_t *r = pal[subset][s];
                uint32_t e;
                if (tc_bc7_sep_alpha[mode] || mode < 4u)
                    e = tc_err3(pix[i], r[0], r[1], r[2]);
                else
                    e = tc_err4(pix[i], r[0], r[1], r[2], r[3], 1);
                if (e < best) {
                    best = e;
                    best_s = s;
                }
            }
            best_color_err = best;
            if (tc_bc7_sep_alpha[mode]) {
                best = UINT_MAX;
                for (s = 0; s < na; ++s) {
                    const uint8_t *r = pal[subset][s];
                    uint32_t e = tc_err1(pix[i][3], r[3]);
                    if (e < best) {
                        best = e;
                        best_a = s;
                    }
                }
                best_alpha_err = best;
            } else {
                best_a = best_s;
                best_alpha_err = 0;
            }
            cand->selectors[i] = (uint8_t)best_s;
            cand->alpha_selectors[i] = (uint8_t)best_a;
            total_err += (uint64_t)best_color_err + (uint64_t)best_alpha_err;
        }
    }
    return total_err;
}

static uint8_t tc_lookup_index_from_mask(uint32_t mask) {
    switch (mask) {
        case 11u: return 0;
        case 12u: return 1;
        case 18u: return 2;
        case 21u: return 3;
        case 33u: return 4;
        case 38u: return 5;
        case 56u: return 6;
        case 63u: return 7;
        default: return 0;
    }
}

static uint8_t tc_dominant_rgb_channel(const uint8_t *p) {
    if (p[0] > p[1]) return p[0] > p[2] ? 0u : 2u;
    return p[1] > p[2] ? 1u : 2u;
}

static uint8_t tc_rgb_partition_lut_index(const uint8_t pix[16][4]) {
    const uint8_t ids[4] = {0, 1, 4, 5};
    uint8_t ch[4];
    uint32_t i, j, bit = 0, mask = 0;
    for (i = 0; i < 4u; ++i) ch[i] = tc_dominant_rgb_channel(pix[ids[i]]);
    for (i = 0; i < 4u; ++i) {
        for (j = i + 1u; j < 4u; ++j) {
            if (ch[i] == ch[j]) mask |= 1u << bit;
            ++bit;
        }
    }
    return tc_lookup_index_from_mask(mask);
}

static uint8_t tc_alpha_partition_lut_index(const uint8_t pix[16][4]) {
    const uint8_t ids[4] = {0, 3, 12, 15};
    uint32_t i, j, bit = 0, mask = 0;
    uint8_t cls[4], amin = 255, amax = 0, median;
    for (i = 0; i < 16u; ++i) {
        if (pix[i][3] < amin) amin = pix[i][3];
        if (pix[i][3] > amax) amax = pix[i][3];
    }
    median = (uint8_t)(((uint32_t)amin + (uint32_t)amax) >> 1u);
    for (i = 0; i < 4u; ++i) cls[i] = pix[ids[i]][3] > median ? 1u : 0u;
    for (i = 0; i < 4u; ++i) {
        for (j = i + 1u; j < 4u; ++j) {
            if (cls[i] == cls[j]) mask |= 1u << bit;
            ++bit;
        }
    }
    return tc_lookup_index_from_mask(mask);
}

static uint64_t tc_partition_cluster_score(const uint8_t pix[16][4],
                                           uint32_t partition,
                                           uint32_t include_alpha) {
    const uint8_t *part = &tc_bc7_partition2[(partition & 63u) * 16u];
    uint32_t sums[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    uint32_t total[4] = {0, 0, 0, 0};
    uint32_t counts[2] = {0, 0};
    uint32_t i, c, comps = include_alpha ? 4u : 3u;
    uint64_t score = 0;
    for (i = 0; i < 16u; ++i) {
        uint32_t s = part[i];
        ++counts[s];
        for (c = 0; c < comps; ++c) {
            sums[s][c] += pix[i][c];
            total[c] += pix[i][c];
        }
    }
    for (i = 0; i < 2u; ++i) {
        if (!counts[i]) continue;
        for (c = 0; c < comps; ++c) {
            int32_t d = (int32_t)(16u * sums[i][c]) -
                        (int32_t)(counts[i] * total[c]);
            score += ((uint64_t)(uint32_t)(d < 0 ? -d : d) *
                      (uint64_t)(uint32_t)(d < 0 ? -d : d)) /
                     counts[i];
        }
    }
    return score;
}

static uint32_t tc_select_partition2(const uint8_t pix[16][4], uint32_t mode,
                                     uint32_t quick) {
    uint32_t best_partition = 0, i, count;
    uint64_t best_score = 0;
    if (!quick) {
        count = 64u;
        for (i = 0; i < count; ++i) {
            uint64_t score = tc_partition_cluster_score(pix, i, mode == 7u);
            if (score > best_score || i == 0u) {
                best_score = score;
                best_partition = i;
            }
        }
    } else if (mode == 7u) {
        uint8_t lut = tc_alpha_partition_lut_index(pix);
        (void)best_score;
        best_partition = tc_bc7_alpha_part_lut[lut][0];
    } else {
        uint8_t lut = tc_rgb_partition_lut_index(pix);
        (void)best_score;
        best_partition = tc_bc7_part_lut[lut][0];
    }
    return best_partition;
}

static void tc_encode_bc7_all_modes_block(const uint8_t pix[16][4],
                                          const tc_bc7_options *opt,
                                          uint8_t out[16]) {
    uint32_t mode;
    uint64_t best_err = UINT64_MAX;
    uint8_t best_block[16];
    uint32_t mask = opt && opt->mode_mask ? opt->mode_mask : 0xffu;
    if (opt && opt->quick) mask = tc_block_quick_mask(pix, mask);
    memset(best_block, 0, sizeof(best_block));
    for (mode = 0; mode < 8u; ++mode) {
        tc_bc7_candidate cand;
        uint64_t err;
        if ((mask & (1u << mode)) == 0u) continue;
        err = tc_build_candidate(mode,
                                 (mode == 1u || mode == 7u)
                                     ? tc_select_partition2(pix, mode,
                                                            opt && opt->quick)
                                     : 0u,
                                 pix, &cand);
        if (err < best_err) {
            best_err = err;
            tc_pack_candidate(&cand, best_block);
        }
    }
    memcpy(out, best_block, 16);
}

tc_result tc_bc7_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                const tc_bc7_options *opt, uint8_t *out_bc7,
                                size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    uint8_t block[16][4];
    size_t need, off = 0;
    tc_bc7_options defopt;

    if (!rgba || !out_bc7 || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc7_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;
    if (!opt) {
        tc_bc7_options_init(&defopt);
        opt = &defopt;
    }

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    memcpy(block[yy * 4u + xx], src, 4);
                }
            }
            tc_encode_bc7_all_modes_block(block, opt, out_bc7 + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}

tc_result tc_bc5_compress_rg8(const uint8_t *rg, uint32_t width,
                              uint32_t height, size_t stride,
                              const tc_bc5_options *opt, uint8_t *out_bc5,
                              size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    uint8_t block[16][2];
    size_t need, off = 0;
    (void)opt;

    if (!rg || !out_bc5 || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 2u) return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc5_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rg + (size_t)y * stride + (size_t)x * 2u;
                    block[yy * 4u + xx][0] = src[0];
                    block[yy * 4u + xx][1] = src[1];
                }
            }
            tc_encode_bc5_block(block, out_bc5 + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}

tc_result tc_bc5_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                const tc_bc5_options *opt, uint8_t *out_bc5,
                                size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    uint8_t block[16][2];
    size_t need, off = 0;
    (void)opt;

    if (!rgba || !out_bc5 || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc5_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    block[yy * 4u + xx][0] = src[0];
                    block[yy * 4u + xx][1] = src[1];
                }
            }
            tc_encode_bc5_block(block, out_bc5 + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}

tc_result tc_bc6h_compress_rgb32f(const float *rgb, uint32_t width,
                                  uint32_t height, size_t stride_bytes,
                                  const tc_bc6h_options *opt,
                                  uint8_t *out_bc6h, size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    float block[16][3];
    size_t need, off = 0;
    (void)opt;

    if (!rgb || !out_bc6h || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride_bytes < (size_t)width * 3u * sizeof(float))
        return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc6h_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const float *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = (const float *)((const uint8_t *)rgb +
                                          (size_t)y * stride_bytes) +
                          (size_t)x * 3u;
                    block[yy * 4u + xx][0] = src[0];
                    block[yy * 4u + xx][1] = src[1];
                    block[yy * 4u + xx][2] = src[2];
                }
            }
            if (opt && opt->signed_float)
                tc_encode_bc6h_block_sf16(block, out_bc6h + off);
            else
                tc_encode_bc6h_block_uf16(block, out_bc6h + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}

static uint64_t tc_encode_etc2_rgb_flat(const uint8_t block[16][4]) {
    uint32_t i, r = 0, g = 0, b = 0;
    for (i = 0; i < 16u; ++i) {
        r += block[i][0];
        g += block[i][1];
        b += block[i][2];
    }
    r = (r + 8u) >> 4;
    g = (g + 8u) >> 4;
    b = (b + 8u) >> 4;
    return 0x02000000u | ((b & 0xf8u) << 16) | ((g & 0xf8u) << 8) |
           (r & 0xf8u);
}

static uint8_t tc_etc2_convert6(float f) {
    int32_t i = (tc_clamp_i32((int32_t)f, 0, 1023) - 15) >> 1;
    return (uint8_t)((i + 11 - ((i + 11) >> 7) - ((i + 4) >> 7)) >> 3);
}

static uint8_t tc_etc2_convert7(float f) {
    int32_t i = (tc_clamp_i32((int32_t)f, 0, 1023) - 15) >> 1;
    return (uint8_t)((i + 9 - ((i + 9) >> 8) - ((i + 6) >> 8)) >> 2);
}

static uint8_t tc_etc2_expand6(uint32_t v) {
    return (uint8_t)((v >> 4) | (v << 2));
}

static uint8_t tc_etc2_expand7(uint32_t v) {
    return (uint8_t)((v >> 6) | (v << 1));
}

static uint64_t tc_etc2_flat_error(const uint8_t block[16][4], uint64_t flat) {
    uint8_t r = (uint8_t)(flat & 0xffu);
    uint8_t g = (uint8_t)((flat >> 8) & 0xffu);
    uint8_t b = (uint8_t)((flat >> 16) & 0xffu);
    uint64_t err = 0;
    uint32_t i;
    r = (uint8_t)(r | (r >> 5));
    g = (uint8_t)(g | (g >> 5));
    b = (uint8_t)(b | (b >> 5));
    for (i = 0; i < 16u; ++i) {
        int32_t dr = (int32_t)block[i][0] - r;
        int32_t dg = (int32_t)block[i][1] - g;
        int32_t db = (int32_t)block[i][2] - b;
        err += (uint64_t)(dr * dr + dg * dg + db * db);
    }
    return err;
}

static uint8_t tc_etc1_quant4(uint32_t v) {
    return (uint8_t)((v * 15u + 127u) / 255u);
}

static uint8_t tc_etc1_expand4(uint32_t v) {
    return (uint8_t)((v << 4) | v);
}

static uint8_t tc_etc1_quant5(uint32_t v) {
    return (uint8_t)((v * 31u + 127u) / 255u);
}

static uint8_t tc_etc1_expand5(uint32_t v) {
    return (uint8_t)((v << 3) | (v >> 2));
}

static uint32_t tc_etc1_subset_for_split(uint32_t split, uint32_t i) {
    uint32_t x = i & 3u;
    uint32_t y = i >> 2;
    return split == 0u ? (y < 2u ? 1u : 0u) : (x < 2u ? 1u : 0u);
}

static uint64_t tc_encode_etc1_individual(const uint8_t block[16][4],
                                          uint64_t *out_err) {
    uint64_t best_bits = 0;
    uint64_t best_err = ~(uint64_t)0;
    uint32_t split;
    for (split = 0; split < 2u; ++split) {
        uint32_t sum[2][3] = {{0, 0, 0}, {0, 0, 0}};
        uint32_t cnt[2] = {0, 0};
        uint8_t base[2][3];
        uint32_t table[2] = {0, 0};
        uint8_t sel[16];
        uint64_t err_total = 0;
        uint32_t s, c, i, t;
        uint64_t d = (uint64_t)split << 24;

        for (i = 0; i < 16u; ++i) {
            s = tc_etc1_subset_for_split(split, i);
            for (c = 0; c < 3u; ++c) sum[s][c] += block[i][c];
            ++cnt[s];
        }
        for (s = 0; s < 2u; ++s) {
            for (c = 0; c < 3u; ++c) {
                uint32_t avg = (sum[s][c] + cnt[s] / 2u) / cnt[s];
                uint8_t q = tc_etc1_quant4(avg);
                base[s][c] = tc_etc1_expand4(q);
            }
        }

        for (s = 0; s < 2u; ++s) {
            uint64_t best_tab_err = ~(uint64_t)0;
            uint8_t best_tab_sel[16];
            uint32_t best_tab = 0;
            memset(best_tab_sel, 0, sizeof(best_tab_sel));
            for (t = 0; t < 8u; ++t) {
                uint64_t tab_err = 0;
                uint8_t tab_sel[16];
                for (i = 0; i < 16u; ++i) {
                    uint64_t pix_best = ~(uint64_t)0;
                    uint32_t pix_sel = 0;
                    if (tc_etc1_subset_for_split(split, i) != s) {
                        tab_sel[i] = 0;
                        continue;
                    }
                    for (c = 0; c < 4u; ++c) {
                        int32_t rr = tc_clamp_i32((int32_t)base[s][0] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t gg = tc_clamp_i32((int32_t)base[s][1] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t bb = tc_clamp_i32((int32_t)base[s][2] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t er = (int32_t)block[i][0] - rr;
                        int32_t eg = (int32_t)block[i][1] - gg;
                        int32_t eb = (int32_t)block[i][2] - bb;
                        uint64_t e = (uint64_t)(er * er + eg * eg + eb * eb);
                        if (e < pix_best) {
                            pix_best = e;
                            pix_sel = c;
                        }
                    }
                    tab_sel[i] = (uint8_t)pix_sel;
                    tab_err += pix_best;
                }
                if (tab_err < best_tab_err) {
                    best_tab_err = tab_err;
                    best_tab = t;
                    memcpy(best_tab_sel, tab_sel, sizeof(best_tab_sel));
                }
            }
            table[s] = best_tab;
            for (i = 0; i < 16u; ++i) {
                if (tc_etc1_subset_for_split(split, i) == s) sel[i] = best_tab_sel[i];
            }
            err_total += best_tab_err;
        }

        d |= (uint64_t)(base[0][0] >> 4) << 0;
        d |= (uint64_t)(base[1][0] >> 4) << 4;
        d |= (uint64_t)(base[0][1] >> 4) << 8;
        d |= (uint64_t)(base[1][1] >> 4) << 12;
        d |= (uint64_t)(base[0][2] >> 4) << 16;
        d |= (uint64_t)(base[1][2] >> 4) << 20;
        d |= (uint64_t)table[0] << 26;
        d |= (uint64_t)table[1] << 29;
        for (i = 0; i < 16u; ++i) {
            uint64_t sv = sel[i];
            d |= (sv & 1u) << (i + 32u);
            d |= (sv & 2u) << (i + 47u);
        }
        d = tc_etc2_fix_byte_order(d);
        if (err_total < best_err) {
            best_err = err_total;
            best_bits = d;
        }
    }
    if (out_err) *out_err = best_err;
    return best_bits;
}

static uint64_t tc_encode_etc1_differential(const uint8_t block[16][4],
                                            uint64_t *out_err) {
    uint64_t best_bits = 0;
    uint64_t best_err = ~(uint64_t)0;
    uint32_t split;
    for (split = 0; split < 2u; ++split) {
        uint32_t sum[2][3] = {{0, 0, 0}, {0, 0, 0}};
        uint32_t cnt[2] = {0, 0};
        uint8_t q[2][3];
        int32_t diff[3];
        uint8_t base[2][3];
        uint32_t table[2] = {0, 0};
        uint8_t sel[16];
        uint64_t err_total = 0;
        uint32_t s, c, i, t;
        uint64_t d = ((uint64_t)split << 24) | (1ULL << 25);

        for (i = 0; i < 16u; ++i) {
            s = tc_etc1_subset_for_split(split, i);
            for (c = 0; c < 3u; ++c) sum[s][c] += block[i][c];
            ++cnt[s];
        }
        for (s = 0; s < 2u; ++s) {
            for (c = 0; c < 3u; ++c) {
                uint32_t avg = (sum[s][c] + cnt[s] / 2u) / cnt[s];
                q[s][c] = tc_etc1_quant5(avg);
            }
        }
        for (c = 0; c < 3u; ++c) {
            diff[c] = (int32_t)q[0][c] - (int32_t)q[1][c];
            if (diff[c] < -4 || diff[c] > 3) {
                if (out_err) *out_err = ~(uint64_t)0;
                return 0;
            }
            base[0][c] = tc_etc1_expand5(q[0][c]);
            base[1][c] = tc_etc1_expand5(q[1][c]);
        }

        for (s = 0; s < 2u; ++s) {
            uint64_t best_tab_err = ~(uint64_t)0;
            uint8_t best_tab_sel[16];
            uint32_t best_tab = 0;
            memset(best_tab_sel, 0, sizeof(best_tab_sel));
            for (t = 0; t < 8u; ++t) {
                uint64_t tab_err = 0;
                uint8_t tab_sel[16];
                for (i = 0; i < 16u; ++i) {
                    uint64_t pix_best = ~(uint64_t)0;
                    uint32_t pix_sel = 0;
                    if (tc_etc1_subset_for_split(split, i) != s) {
                        tab_sel[i] = 0;
                        continue;
                    }
                    for (c = 0; c < 4u; ++c) {
                        int32_t rr = tc_clamp_i32((int32_t)base[s][0] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t gg = tc_clamp_i32((int32_t)base[s][1] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t bb = tc_clamp_i32((int32_t)base[s][2] + tc_etc1_mod[t][c],
                                                   0, 255);
                        int32_t er = (int32_t)block[i][0] - rr;
                        int32_t eg = (int32_t)block[i][1] - gg;
                        int32_t eb = (int32_t)block[i][2] - bb;
                        uint64_t e = (uint64_t)(er * er + eg * eg + eb * eb);
                        if (e < pix_best) {
                            pix_best = e;
                            pix_sel = c;
                        }
                    }
                    tab_sel[i] = (uint8_t)pix_sel;
                    tab_err += pix_best;
                }
                if (tab_err < best_tab_err) {
                    best_tab_err = tab_err;
                    best_tab = t;
                    memcpy(best_tab_sel, tab_sel, sizeof(best_tab_sel));
                }
            }
            table[s] = best_tab;
            for (i = 0; i < 16u; ++i) {
                if (tc_etc1_subset_for_split(split, i) == s) sel[i] = best_tab_sel[i];
            }
            err_total += best_tab_err;
        }

        d |= (uint64_t)q[1][0] << 0;
        d |= (uint64_t)(diff[0] & 7) << 3;
        d |= (uint64_t)q[1][1] << 8;
        d |= (uint64_t)(diff[1] & 7) << 11;
        d |= (uint64_t)q[1][2] << 16;
        d |= (uint64_t)(diff[2] & 7) << 19;
        d |= (uint64_t)table[0] << 26;
        d |= (uint64_t)table[1] << 29;
        for (i = 0; i < 16u; ++i) {
            uint64_t sv = sel[i];
            d |= (sv & 1u) << (i + 32u);
            d |= (sv & 2u) << (i + 47u);
        }
        d = tc_etc2_fix_byte_order(d);
        if (err_total < best_err) {
            best_err = err_total;
            best_bits = d;
        }
    }
    if (out_err) *out_err = best_err;
    return best_bits;
}

static uint64_t tc_encode_etc2_rgb_planar(const uint8_t block[16][4],
                                          uint64_t *out_err) {
    int32_t sum_r = 0, sum_g = 0, sum_b = 0;
    int32_t dif_rxz = 0, dif_gxz = 0, dif_bxz = 0;
    int32_t dif_ryz = 0, dif_gyz = 0, dif_byz = 0;
    const int32_t scaling[4] = {-255, -85, 85, 255};
    const float scale = -4.0f / ((255.0f * 255.0f * 8.0f + 85.0f * 85.0f * 8.0f) *
                                 16.0f);
    float ar, ag, ab, br, bg, bb, dr, dg, db;
    uint32_t co_r, co_g, co_b, ch_r, ch_g, ch_b, cv_r, cv_g, cv_b;
    uint32_t rgbv, rgbh, lo, hi, idx;
    uint64_t err = 0;
    uint32_t i;

    for (i = 0; i < 16u; ++i) {
        sum_r += block[i][0];
        sum_g += block[i][1];
        sum_b += block[i][2];
    }
    for (i = 0; i < 16u; ++i) {
        int32_t x = (int32_t)(i / 4u);
        int32_t y = (int32_t)(i & 3u);
        int32_t dif_r = ((int32_t)block[i][0] << 4) - sum_r;
        int32_t dif_g = ((int32_t)block[i][1] << 4) - sum_g;
        int32_t dif_b = ((int32_t)block[i][2] << 4) - sum_b;
        dif_rxz += dif_r * scaling[x];
        dif_gxz += dif_g * scaling[x];
        dif_bxz += dif_b * scaling[x];
        dif_ryz += dif_r * scaling[y];
        dif_gyz += dif_g * scaling[y];
        dif_byz += dif_b * scaling[y];
    }

    ar = (float)dif_rxz * scale;
    ag = (float)dif_gxz * scale;
    ab = (float)dif_bxz * scale;
    br = (float)dif_ryz * scale;
    bg = (float)dif_gyz * scale;
    bb = (float)dif_byz * scale;
    dr = (float)sum_r * 0.25f;
    dg = (float)sum_g * 0.25f;
    db = (float)sum_b * 0.25f;

    co_r = tc_etc2_convert6(ar * 255.0f + br * 255.0f + dr);
    co_g = tc_etc2_convert7(ag * 255.0f + bg * 255.0f + dg);
    co_b = tc_etc2_convert6(ab * 255.0f + bb * 255.0f + db);
    ch_r = tc_etc2_convert6(ar * -425.0f + br * 255.0f + dr);
    ch_g = tc_etc2_convert7(ag * -425.0f + bg * 255.0f + dg);
    ch_b = tc_etc2_convert6(ab * -425.0f + bb * 255.0f + db);
    cv_r = tc_etc2_convert6(ar * 255.0f + br * -425.0f + dr);
    cv_g = tc_etc2_convert7(ag * 255.0f + bg * -425.0f + dg);
    cv_b = tc_etc2_convert6(ab * 255.0f + bb * -425.0f + db);

    {
        int32_t ro = tc_etc2_expand6(co_r);
        int32_t go = tc_etc2_expand7(co_g);
        int32_t bo = tc_etc2_expand6(co_b);
        int32_t rh = (int32_t)tc_etc2_expand6(ch_r) - ro;
        int32_t gh = (int32_t)tc_etc2_expand7(ch_g) - go;
        int32_t bh = (int32_t)tc_etc2_expand6(ch_b) - bo;
        int32_t rv = (int32_t)tc_etc2_expand6(cv_r) - ro;
        int32_t gv = (int32_t)tc_etc2_expand7(cv_g) - go;
        int32_t bv = (int32_t)tc_etc2_expand6(cv_b) - bo;
        int32_t ro2 = (ro << 2) + 2;
        int32_t go2 = (go << 2) + 2;
        int32_t bo2 = (bo << 2) + 2;
        for (i = 0; i < 16u; ++i) {
            int32_t x = (int32_t)(i / 4u);
            int32_t y = (int32_t)(i & 3u);
            int32_t rr = tc_clamp_u8_i32((rh * x + rv * y + ro2) >> 2);
            int32_t gg = tc_clamp_u8_i32((gh * x + gv * y + go2) >> 2);
            int32_t bbv = tc_clamp_u8_i32((bh * x + bv * y + bo2) >> 2);
            int32_t er = (int32_t)block[i][0] - rr;
            int32_t eg = (int32_t)block[i][1] - gg;
            int32_t eb = (int32_t)block[i][2] - bbv;
            err += (uint64_t)(er * er + eg * eg + eb * eb);
        }
    }

    rgbv = cv_b | (cv_g << 6) | (cv_r << 13);
    rgbh = ch_b | (ch_g << 6) | (ch_r << 13);
    hi = rgbv | ((rgbh & 0x1fffu) << 19);
    lo = (ch_r & 0x1u) | 0x2u | ((ch_r << 1) & 0x7cu);
    lo |= ((co_b & 0x07u) << 7) | ((co_b & 0x18u) << 8) | ((co_b & 0x20u) << 11);
    lo |= ((co_g & 0x3fu) << 17) | ((co_g & 0x40u) << 18);
    lo |= co_r << 25;
    idx = (co_r & 0x20u) | ((co_g & 0x20u) >> 1) | ((co_b & 0x1eu) >> 1);
    lo |= tc_etc2_planar_flags[idx];
    if (out_err) *out_err = err;
    return (uint64_t)tc_bswap32(lo) | ((uint64_t)tc_bswap32(hi) << 32);
}

static uint64_t tc_encode_etc2_rgb(const uint8_t block[16][4]) {
    uint64_t flat = tc_encode_etc2_rgb_flat(block);
    uint64_t flat_err = tc_etc2_flat_error(block, flat);
    uint64_t individual_err = 0;
    uint64_t individual = tc_encode_etc1_individual(block, &individual_err);
    uint64_t differential_err = 0;
    uint64_t differential = tc_encode_etc1_differential(block, &differential_err);
    uint64_t planar_err = 0;
    uint64_t planar = tc_encode_etc2_rgb_planar(block, &planar_err);
    uint64_t best = flat;
    uint64_t best_err = flat_err;
    if (individual_err < best_err) {
        best = individual;
        best_err = individual_err;
    }
    if (differential_err < best_err) {
        best = differential;
        best_err = differential_err;
    }
    if (planar_err < best_err) best = planar;
    return best;
}

static uint64_t tc_encode_eac_alpha(const uint8_t alpha[16]) {
    uint32_t i, j, tab;
    uint8_t minv = 255, maxv = 0, mid;
    int32_t range, best_err = INT_MAX, best_tab = 0, best_mul = 0;
    uint8_t best_sel[16];

    for (i = 0; i < 16u; ++i) {
        if (alpha[i] < minv) minv = alpha[i];
        if (alpha[i] > maxv) maxv = alpha[i];
    }
    if (minv == maxv) return minv;

    mid = (uint8_t)((minv + maxv) >> 1);
    range = (int32_t)maxv - (int32_t)minv;
    memset(best_sel, 0, sizeof(best_sel));
    for (tab = 0; tab < 16u; ++tab) {
        int32_t mul = ((range * tc_eac_alpha_range[tab]) >> 16) + 1;
        int32_t err = 0;
        uint8_t sel[16];
        for (i = 0; i < 16u; ++i) {
            int32_t local_best = INT_MAX;
            uint32_t local_sel = 0;
            for (j = 0; j < 8u; ++j) {
                int32_t rec = tc_clamp_i32((int32_t)mid + tc_eac_alpha[tab][j] * mul,
                                           0, 255);
                int32_t diff = (int32_t)alpha[i] - rec;
                int32_t e = diff * diff;
                if (e < local_best) {
                    local_best = e;
                    local_sel = j;
                }
            }
            sel[i] = (uint8_t)local_sel;
            err += local_best;
        }
        if (err < best_err) {
            best_err = err;
            best_tab = (int32_t)tab;
            best_mul = mul;
            memcpy(best_sel, sel, sizeof(best_sel));
            if (err == 0) break;
        }
    }

    {
        uint64_t d = ((uint64_t)mid << 56) | ((uint64_t)(best_mul & 15) << 52) |
                     ((uint64_t)(best_tab & 15) << 48);
        int shift = 45;
        for (i = 0; i < 16u; ++i) {
            d |= (uint64_t)best_sel[i] << shift;
            shift -= 3;
        }
        return tc_bswap64(d);
    }
}

tc_result tc_etc2_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                 uint32_t height, size_t stride,
                                 const tc_etc2_options *opt,
                                 uint8_t *out_etc2, size_t out_size) {
    tc_etc2_options defopt;
    uint32_t bx, by, x, y, xx, yy;
    uint8_t block[16][4];
    uint8_t alpha[16];
    size_t need, off = 0;

    if (!opt) {
        tc_etc2_options_init(&defopt);
        opt = &defopt;
    }
    if (!rgba || !out_etc2 || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = opt->alpha ? tc_etc2_rgba_compressed_size(width, height)
                      : tc_etc2_rgb_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4u) {
        for (bx = 0; bx < width; bx += 4u) {
            for (yy = 0; yy < 4u; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4u; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    memcpy(block[yy * 4u + xx], src, 4u);
                    alpha[yy * 4u + xx] = src[3];
                }
            }
            if (opt->alpha) {
                tc_wr_u64(out_etc2 + off, tc_encode_eac_alpha(alpha));
                off += 8u;
            }
            tc_wr_u64(out_etc2 + off, tc_encode_etc2_rgb(block));
            off += 8u;
        }
    }

    return TC_SUCCESS;
}

tc_result tc_eac_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                int rg11, uint8_t *out_eac, size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    uint8_t red[16], green[16];
    size_t need, off = 0;

    if (!rgba || !out_eac || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = rg11 ? tc_eac_rg11_compressed_size(width, height)
                : tc_eac_r11_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4u) {
        for (bx = 0; bx < width; bx += 4u) {
            for (yy = 0; yy < 4u; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4u; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    red[yy * 4u + xx] = src[0];
                    green[yy * 4u + xx] = src[1];
                }
            }
            tc_wr_u64(out_eac + off, tc_encode_eac_alpha(red));
            off += 8u;
            if (rg11) {
                tc_wr_u64(out_eac + off, tc_encode_eac_alpha(green));
                off += 8u;
            }
        }
    }

    return TC_SUCCESS;
}

static void tc_astc_write_const_from_sum(const uint32_t sum[4], uint32_t count,
                                         uint8_t out[16]) {
    uint32_t c;
    static const uint8_t prefix[8] = {0xfc, 0xfd, 0xff, 0xff,
                                      0xff, 0xff, 0xff, 0xff};
    memcpy(out, prefix, 8u);
    for (c = 0; c < 4u; ++c) {
        uint32_t avg = (sum[c] + count / 2u) / count;
        uint32_t v16 = avg * 257u;
        out[8u + c * 2u] = (uint8_t)v16;
        out[9u + c * 2u] = (uint8_t)(v16 >> 8);
    }
}

static uint8_t tc_bitrev8(uint8_t v) {
    v = (uint8_t)(((v & 0x0fu) << 4) | ((v >> 4) & 0x0fu));
    v = (uint8_t)(((v & 0x33u) << 2) | ((v >> 2) & 0x33u));
    v = (uint8_t)(((v & 0x55u) << 1) | ((v >> 1) & 0x55u));
    return v;
}

static int tc_astc_block_is_solid(const uint8_t block[144][4], uint32_t count) {
    uint32_t i, c;
    for (i = 1; i < count; ++i) {
        for (c = 0; c < 4u; ++c) {
            if (block[i][c] != block[0][c]) return 0;
        }
    }
    return 1;
}

static int tc_astc_block_is_opaque(const uint8_t block[144][4], uint32_t count) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
        if (block[i][3] != 255u) return 0;
    }
    return 1;
}

static int tc_astc_block_is_luminance(const uint8_t block[144][4],
                                      uint32_t count) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
        if (block[i][0] != block[i][1] || block[i][0] != block[i][2]) return 0;
    }
    return 1;
}

static int tc_astc_block_is_rgb_scale(const uint8_t block[144][4],
                                      uint32_t count) {
    uint32_t i, c, anchor = 0, major = 0;
    uint32_t best_sum = 0, major_v;
    for (i = 0; i < count; ++i) {
        uint32_t sum = (uint32_t)block[i][0] + block[i][1] + block[i][2];
        if (sum > best_sum) {
            best_sum = sum;
            anchor = i;
        }
    }
    if (!best_sum) return 0;
    for (c = 1; c < 3u; ++c) {
        if (block[anchor][c] > block[anchor][major]) major = c;
    }
    major_v = block[anchor][major];
    if (!major_v) return 0;
    for (i = 0; i < count; ++i) {
        for (c = 0; c < 3u; ++c) {
            int32_t a = (int32_t)block[i][c] * (int32_t)major_v;
            int32_t b = (int32_t)block[i][major] * (int32_t)block[anchor][c];
            int32_t d = a > b ? a - b : b - a;
            if (d > (int32_t)(major_v * 6u)) return 0;
        }
    }
    return 1;
}

static uint32_t tc_astc_axis_value(const uint8_t p[4], uint32_t axis) {
    if (axis < 4u) return p[axis];
    return tc_luma_u8(p);
}

static uint32_t tc_astc_weight_unquant(uint32_t q, uint32_t quant_level) {
    static const uint8_t table[12][32] = {
        {0, 64},
        {0, 32, 64},
        {0, 21, 43, 64},
        {0, 16, 32, 48, 64},
        {0, 12, 25, 39, 52, 64},
        {0, 9, 18, 27, 37, 46, 55, 64},
        {0, 7, 14, 21, 28, 36, 43, 50, 57, 64},
        {0, 5, 11, 17, 23, 28, 36, 41, 47, 53, 59, 64},
        {0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64},
        {0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 35, 38, 41, 45, 48, 51,
         55, 58, 61, 64},
        {0, 2, 5, 8, 11, 13, 16, 19, 22, 24, 27, 30, 34, 37, 40, 42,
         45, 48, 51, 53, 56, 59, 62, 64},
        {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
         34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64}};
    return table[quant_level][q];
}

static uint8_t tc_astc_weight_scramble(uint32_t q, uint32_t quant_level) {
    static const uint8_t table[12][32] = {
        {0, 1},
        {0, 1, 2},
        {0, 1, 2, 3},
        {0, 1, 2, 3, 4},
        {0, 2, 4, 5, 3, 1},
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 2, 4, 6, 8, 9, 7, 5, 3, 1},
        {0, 4, 8, 2, 6, 10, 11, 7, 3, 9, 5, 1},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 4, 8, 12, 16, 2, 6, 10, 14, 18, 19, 15, 11, 7, 3, 17,
         13, 9, 5, 1},
        {0, 8, 16, 2, 10, 18, 4, 12, 20, 6, 14, 22, 23, 15, 7, 21,
         13, 5, 19, 11, 3, 17, 9, 1},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}};
    return table[quant_level][q];
}

static uint32_t tc_astc_quant_levels(uint32_t quant_method) {
    static const uint16_t levels[21] = {2,  3,  4,  5,  6,  8,   10,
                                        12, 16, 20, 24, 32, 40,  48,
                                        64, 80, 96, 128, 160, 192, 256};
    return levels[quant_method];
}

static const uint8_t tc_astc_color_pquant_to_uquant_q6[6] = {
    0u, 255u, 51u, 204u, 102u, 153u};
static const uint8_t tc_astc_color_pquant_to_uquant_q8[8] = {
    0u, 36u, 73u, 109u, 146u, 182u, 219u, 255u};
static const uint8_t tc_astc_color_pquant_to_uquant_q10[10] = {
    0u, 255u, 28u, 227u, 56u, 199u, 84u, 171u, 113u, 142u};
static const uint8_t tc_astc_color_pquant_to_uquant_q12[12] = {
    0u, 255u, 69u, 186u, 23u, 232u, 92u, 163u, 46u, 209u, 116u, 139u};
static const uint8_t tc_astc_color_pquant_to_uquant_q16[16] = {
    0u,   17u,  34u,  51u,  68u,  85u,  102u, 119u,
    136u, 153u, 170u, 187u, 204u, 221u, 238u, 255u};
static const uint8_t tc_astc_color_pquant_to_uquant_q20[20] = {
    0u,  255u, 67u, 188u, 13u, 242u, 80u, 175u, 27u, 228u,
    94u, 161u, 40u, 215u, 107u, 148u, 54u, 201u, 121u, 134u};
static const uint8_t tc_astc_color_pquant_to_uquant_q24[24] = {
    0u,  255u, 33u, 222u, 66u, 189u, 99u, 156u, 11u, 244u, 44u, 211u,
    77u, 178u, 110u, 145u, 22u, 233u, 55u, 200u, 88u, 167u, 121u, 134u};
static const uint8_t tc_astc_color_pquant_to_uquant_q32[32] = {
    0u,   8u,   16u,  24u,  33u,  41u,  49u,  57u,
    66u,  74u,  82u,  90u,  99u,  107u, 115u, 123u,
    132u, 140u, 148u, 156u, 165u, 173u, 181u, 189u,
    198u, 206u, 214u, 222u, 231u, 239u, 247u, 255u};
static const uint8_t tc_astc_color_pquant_to_uquant_q40[40] = {
    0u,   255u, 32u, 223u, 65u, 190u, 97u, 158u, 6u,   249u,
    39u,  216u, 71u, 184u, 104u, 151u, 13u, 242u, 45u, 210u,
    78u,  177u, 110u, 145u, 19u, 236u, 52u, 203u, 84u, 171u,
    117u, 138u, 26u, 229u, 58u, 197u, 91u, 164u, 123u, 132u};
static const uint8_t tc_astc_color_pquant_to_uquant_q48[48] = {
    0u,   255u, 16u, 239u, 32u, 223u, 48u, 207u, 65u, 190u, 81u, 174u,
    97u,  158u, 113u, 142u, 5u,  250u, 21u, 234u, 38u, 217u, 54u, 201u,
    70u,  185u, 86u, 169u, 103u, 152u, 119u, 136u, 11u, 244u, 27u, 228u,
    43u,  212u, 59u, 196u, 76u, 179u, 92u, 163u, 108u, 147u, 124u, 131u};
static const uint8_t tc_astc_color_pquant_to_uquant_q64[64] = {
    0u,   4u,   8u,   12u,  16u,  20u,  24u,  28u,
    32u,  36u,  40u,  44u,  48u,  52u,  56u,  60u,
    65u,  69u,  73u,  77u,  81u,  85u,  89u,  93u,
    97u,  101u, 105u, 109u, 113u, 117u, 121u, 125u,
    130u, 134u, 138u, 142u, 146u, 150u, 154u, 158u,
    162u, 166u, 170u, 174u, 178u, 182u, 186u, 190u,
    195u, 199u, 203u, 207u, 211u, 215u, 219u, 223u,
    227u, 231u, 235u, 239u, 243u, 247u, 251u, 255u};
static const uint8_t tc_astc_color_pquant_to_uquant_q80[80] = {
    0u,   255u, 16u, 239u, 32u, 223u, 48u, 207u, 64u, 191u, 80u, 175u,
    96u,  159u, 112u, 143u, 3u,  252u, 19u, 236u, 35u, 220u, 51u, 204u,
    67u,  188u, 83u, 172u, 100u, 155u, 116u, 139u, 6u,  249u, 22u, 233u,
    38u,  217u, 54u, 201u, 71u, 184u, 87u, 168u, 103u, 152u, 119u, 136u,
    9u,   246u, 25u, 230u, 42u, 213u, 58u, 197u, 74u, 181u, 90u, 165u,
    106u, 149u, 122u, 133u, 13u, 242u, 29u, 226u, 45u, 210u, 61u, 194u,
    77u,  178u, 93u, 162u, 109u, 146u, 125u, 130u};
static const uint8_t tc_astc_color_pquant_to_uquant_q96[96] = {
    0u,   255u, 8u,   247u, 16u, 239u, 24u, 231u, 32u, 223u, 40u, 215u,
    48u,  207u, 56u,  199u, 64u, 191u, 72u, 183u, 80u, 175u, 88u, 167u,
    96u,  159u, 104u, 151u, 112u, 143u, 120u, 135u, 2u,  253u, 10u, 245u,
    18u,  237u, 26u,  229u, 35u, 220u, 43u, 212u, 51u, 204u, 59u, 196u,
    67u,  188u, 75u,  180u, 83u, 172u, 91u, 164u, 99u, 156u, 107u, 148u,
    115u, 140u, 123u, 132u, 5u,  250u, 13u, 242u, 21u, 234u, 29u, 226u,
    37u,  218u, 45u,  210u, 53u, 202u, 61u, 194u, 70u, 185u, 78u, 177u,
    86u,  169u, 94u,  161u, 102u, 153u, 110u, 145u, 118u, 137u, 126u, 129u};
static const uint8_t tc_astc_color_pquant_to_uquant_q128[128] = {
    0u,   2u,   4u,   6u,   8u,   10u,  12u,  14u,  16u,  18u,  20u,
    22u,  24u,  26u,  28u,  30u,  32u,  34u,  36u,  38u,  40u,  42u,
    44u,  46u,  48u,  50u,  52u,  54u,  56u,  58u,  60u,  62u,  64u,
    66u,  68u,  70u,  72u,  74u,  76u,  78u,  80u,  82u,  84u,  86u,
    88u,  90u,  92u,  94u,  96u,  98u,  100u, 102u, 104u, 106u, 108u,
    110u, 112u, 114u, 116u, 118u, 120u, 122u, 124u, 126u, 129u, 131u,
    133u, 135u, 137u, 139u, 141u, 143u, 145u, 147u, 149u, 151u, 153u,
    155u, 157u, 159u, 161u, 163u, 165u, 167u, 169u, 171u, 173u, 175u,
    177u, 179u, 181u, 183u, 185u, 187u, 189u, 191u, 193u, 195u, 197u,
    199u, 201u, 203u, 205u, 207u, 209u, 211u, 213u, 215u, 217u, 219u,
    221u, 223u, 225u, 227u, 229u, 231u, 233u, 235u, 237u, 239u, 241u,
    243u, 245u, 247u, 249u, 251u, 253u, 255u};
static const uint8_t tc_astc_color_pquant_to_uquant_q160[160] = {
    0u,   255u, 8u,   247u, 16u, 239u, 24u, 231u, 32u, 223u, 40u, 215u,
    48u,  207u, 56u,  199u, 64u, 191u, 72u, 183u, 80u, 175u, 88u, 167u,
    96u,  159u, 104u, 151u, 112u, 143u, 120u, 135u, 1u,  254u, 9u,  246u,
    17u,  238u, 25u,  230u, 33u, 222u, 41u, 214u, 49u, 206u, 57u, 198u,
    65u,  190u, 73u,  182u, 81u, 174u, 89u, 166u, 97u, 158u, 105u, 150u,
    113u, 142u, 121u, 134u, 3u,  252u, 11u, 244u, 19u, 236u, 27u, 228u,
    35u,  220u, 43u,  212u, 51u, 204u, 59u, 196u, 67u, 188u, 75u, 180u,
    83u,  172u, 91u,  164u, 99u, 156u, 107u, 148u, 115u, 140u, 123u, 132u,
    4u,   251u, 12u,  243u, 20u, 235u, 28u, 227u, 36u, 219u, 44u, 211u,
    52u,  203u, 60u,  195u, 68u, 187u, 76u, 179u, 84u, 171u, 92u, 163u,
    100u, 155u, 108u, 147u, 116u, 139u, 124u, 131u, 6u,  249u, 14u, 241u,
    22u,  233u, 30u,  225u, 38u, 217u, 46u, 209u, 54u, 201u, 62u, 193u,
    70u,  185u, 78u,  177u, 86u, 169u, 94u, 161u, 102u, 153u, 110u, 145u,
    118u, 137u, 126u, 129u};
static const uint8_t tc_astc_color_pquant_to_uquant_q192[192] = {
    0u,   255u, 4u,   251u, 8u,   247u, 12u,  243u, 16u,  239u, 20u,  235u,
    24u,  231u, 28u,  227u, 32u,  223u, 36u,  219u, 40u,  215u, 44u,  211u,
    48u,  207u, 52u,  203u, 56u,  199u, 60u,  195u, 64u,  191u, 68u,  187u,
    72u,  183u, 76u,  179u, 80u,  175u, 84u,  171u, 88u,  167u, 92u,  163u,
    96u,  159u, 100u, 155u, 104u, 151u, 108u, 147u, 112u, 143u, 116u, 139u,
    120u, 135u, 124u, 131u, 1u,   254u, 5u,   250u, 9u,   246u, 13u,  242u,
    17u,  238u, 21u,  234u, 25u,  230u, 29u,  226u, 33u,  222u, 37u,  218u,
    41u,  214u, 45u,  210u, 49u,  206u, 53u,  202u, 57u,  198u, 61u,  194u,
    65u,  190u, 69u,  186u, 73u,  182u, 77u,  178u, 81u,  174u, 85u,  170u,
    89u,  166u, 93u,  162u, 97u,  158u, 101u, 154u, 105u, 150u, 109u, 146u,
    113u, 142u, 117u, 138u, 121u, 134u, 125u, 130u, 2u,   253u, 6u,   249u,
    10u,  245u, 14u,  241u, 18u,  237u, 22u,  233u, 26u,  229u, 30u,  225u,
    34u,  221u, 38u,  217u, 42u,  213u, 46u,  209u, 50u,  205u, 54u,  201u,
    58u,  197u, 62u,  193u, 66u,  189u, 70u,  185u, 74u,  181u, 78u,  177u,
    82u,  173u, 86u,  169u, 90u,  165u, 94u,  161u, 98u,  157u, 102u, 153u,
    106u, 149u, 110u, 145u, 114u, 141u, 118u, 137u, 122u, 133u, 126u, 129u};

static const uint8_t *const tc_astc_color_pquant_to_uquant[17] = {
    tc_astc_color_pquant_to_uquant_q6,   tc_astc_color_pquant_to_uquant_q8,
    tc_astc_color_pquant_to_uquant_q10,  tc_astc_color_pquant_to_uquant_q12,
    tc_astc_color_pquant_to_uquant_q16,  tc_astc_color_pquant_to_uquant_q20,
    tc_astc_color_pquant_to_uquant_q24,  tc_astc_color_pquant_to_uquant_q32,
    tc_astc_color_pquant_to_uquant_q40,  tc_astc_color_pquant_to_uquant_q48,
    tc_astc_color_pquant_to_uquant_q64,  tc_astc_color_pquant_to_uquant_q80,
    tc_astc_color_pquant_to_uquant_q96,  tc_astc_color_pquant_to_uquant_q128,
    tc_astc_color_pquant_to_uquant_q160, tc_astc_color_pquant_to_uquant_q192,
    NULL};

typedef struct tc_astc_block_mode_info {
    uint16_t block_mode;
    uint8_t weight_x;
    uint8_t weight_y;
    uint8_t quant_method;
    uint8_t weight_bits;
    uint8_t dual_plane;
} tc_astc_block_mode_info;

typedef struct tc_astc_decim_cache_entry {
    uint8_t block_x;
    uint8_t block_y;
    uint8_t weight_x;
    uint8_t weight_y;
    uint8_t tw_idx[144][4];
    uint8_t tw_contrib[144][4];
    uint8_t tw_count[144];
} tc_astc_decim_cache_entry;

typedef struct tc_astc_candidate_cache_entry {
    uint8_t valid;
    uint8_t endpoint_end_bit;
    uint32_t count;
    tc_astc_block_mode_info candidates[2048];
} tc_astc_candidate_cache_entry;

typedef struct tc_astc_partition_info {
    uint16_t partition_index;
    uint8_t partition_of_texel[144];
    uint8_t texels_of_partition[4][144];
    uint8_t partition_texel_count[4];
} tc_astc_partition_info;

typedef struct tc_astc_encode_context {
    uint32_t block_x;
    uint32_t block_y;
    uint32_t texel_count;
    int quality;
    tc_astc_candidate_cache_entry candidate_cache[4];
    tc_astc_decim_cache_entry decim_cache[169];
    tc_astc_partition_info part2_cache[1024];
    tc_astc_partition_info part3_cache[1024];
    tc_astc_partition_info part4_cache[1024];
    uint8_t color_pquant_lut[21][256];
    uint32_t decim_cache_count;
    uint32_t part2_count;
    uint32_t part3_count;
    uint32_t part4_count;
} tc_astc_encode_context;

static void tc_astc_build_color_pquant_lut(tc_astc_encode_context *ctx);
static uint32_t tc_astc_color_symbol_uquant(uint32_t quant_method,
                                            uint32_t sym);
static uint32_t tc_astc_color_roundtrip(const tc_astc_encode_context *ctx,
                                        uint32_t quant_method, uint32_t v);

static int tc_astc_decode_block_mode_2d(uint32_t block_mode,
                                        tc_astc_block_mode_info *info) {
    uint32_t base_quant_mode = (block_mode >> 4) & 1u;
    uint32_t h = (block_mode >> 9) & 1u;
    uint32_t d = (block_mode >> 10) & 1u;
    uint32_t a = (block_mode >> 5) & 3u;
    uint32_t weights_x = 0, weights_y = 0, weight_count, real_weight_count;
    uint32_t quant_method, weight_bits;

    if ((block_mode & 3u) != 0u) {
        uint32_t b;
        base_quant_mode |= (block_mode & 3u) << 1;
        b = (block_mode >> 7) & 3u;
        switch ((block_mode >> 2) & 3u) {
            case 0:
                weights_x = b + 4u;
                weights_y = a + 2u;
                break;
            case 1:
                weights_x = b + 8u;
                weights_y = a + 2u;
                break;
            case 2:
                weights_x = a + 2u;
                weights_y = b + 8u;
                break;
            default:
                b &= 1u;
                if (block_mode & 0x100u) {
                    weights_x = b + 2u;
                    weights_y = a + 2u;
                } else {
                    weights_x = a + 2u;
                    weights_y = b + 6u;
                }
                break;
        }
    } else {
        uint32_t b;
        base_quant_mode |= ((block_mode >> 2) & 3u) << 1;
        if (((block_mode >> 2) & 3u) == 0u) return 0;
        b = (block_mode >> 9) & 3u;
        switch ((block_mode >> 7) & 3u) {
            case 0:
                weights_x = 12u;
                weights_y = a + 2u;
                break;
            case 1:
                weights_x = a + 2u;
                weights_y = 12u;
                break;
            case 2:
                weights_x = a + 6u;
                weights_y = b + 6u;
                d = 0u;
                h = 0u;
                break;
            default:
                if (a == 0u) {
                    weights_x = 6u;
                    weights_y = 10u;
                } else if (a == 1u) {
                    weights_x = 10u;
                    weights_y = 6u;
                } else {
                    return 0;
                }
                break;
        }
    }

    if (base_quant_mode < 2u) return 0;
    quant_method = (base_quant_mode - 2u) + 6u * h;
    if (quant_method >= 12u) return 0;
    weight_count = weights_x * weights_y;
    real_weight_count = weight_count * (d + 1u);
    weight_bits = tc_astc_ise_sequence_bitcount(real_weight_count, quant_method);
    if (real_weight_count > 64u || weight_bits < 24u || weight_bits > 96u)
        return 0;
    info->block_mode = (uint16_t)block_mode;
    info->weight_x = (uint8_t)weights_x;
    info->weight_y = (uint8_t)weights_y;
    info->quant_method = (uint8_t)quant_method;
    info->weight_bits = (uint8_t)weight_bits;
    info->dual_plane = (uint8_t)d;
    return 1;
}

static uint32_t tc_astc_build_block_mode_candidates(uint32_t block_x,
                                                   uint32_t block_y,
                                                   uint32_t endpoint_end_bit,
                                                   int quality,
                                                   tc_astc_block_mode_info out[2048]) {
    uint32_t mode, count = 0;
    uint32_t max_quant = quality > 1 ? 11u : (quality > 0 ? 5u : 2u);
    uint8_t seen[13][13][12];
    memset(seen, 0, sizeof(seen));
    for (mode = 0; mode < 2048u; ++mode) {
        tc_astc_block_mode_info info;
        if (!tc_astc_decode_block_mode_2d(mode, &info)) continue;
        if (info.dual_plane) continue;
        if (info.weight_x > block_x || info.weight_y > block_y) continue;
        if (info.quant_method > max_quant) continue;
        if (endpoint_end_bit > 128u - info.weight_bits) continue;
        if (seen[info.weight_y][info.weight_x][info.quant_method]) continue;
        seen[info.weight_y][info.weight_x][info.quant_method] = 1u;
        out[count++] = info;
    }
    return count;
}

static uint32_t tc_astc_build_dual_block_mode_candidates(
    uint32_t block_x, uint32_t block_y, uint32_t endpoint_end_bit, int quality,
    tc_astc_block_mode_info out[2048]) {
    uint32_t mode, count = 0;
    uint32_t max_quant = quality > 1 ? 11u : 5u;
    uint8_t seen[13][13][12];
    memset(seen, 0, sizeof(seen));
    for (mode = 0; mode < 2048u; ++mode) {
        tc_astc_block_mode_info info;
        if (!tc_astc_decode_block_mode_2d(mode, &info)) continue;
        if (!info.dual_plane) continue;
        if (info.weight_x > block_x || info.weight_y > block_y) continue;
        if (info.quant_method > max_quant) continue;
        if (endpoint_end_bit + 2u > 128u - info.weight_bits) continue;
        if (seen[info.weight_y][info.weight_x][info.quant_method]) continue;
        seen[info.weight_y][info.weight_x][info.quant_method] = 1u;
        out[count++] = info;
    }
    return count;
}

static uint32_t tc_astc_hash52(uint32_t inp) {
    inp ^= inp >> 15;
    inp *= 0xeede0891u;
    inp ^= inp >> 5;
    inp += inp << 16;
    inp ^= inp >> 7;
    inp ^= inp >> 3;
    inp ^= inp << 6;
    inp ^= inp >> 17;
    return inp;
}

static uint8_t tc_astc_select_partition(uint32_t seed, uint32_t x, uint32_t y,
                                        uint32_t partition_count,
                                        int small_block) {
    uint32_t rnum;
    uint32_t seed1, seed2, seed3, seed4, seed5, seed6, seed7, seed8;
    uint32_t seed9, seed10, seed11, seed12;
    uint32_t sh1, sh2, sh3, a, b, c, d;
    if (small_block) {
        x <<= 1;
        y <<= 1;
    }
    seed += (partition_count - 1u) * 1024u;
    rnum = tc_astc_hash52(seed);
    seed1 = rnum & 15u;
    seed2 = (rnum >> 4) & 15u;
    seed3 = (rnum >> 8) & 15u;
    seed4 = (rnum >> 12) & 15u;
    seed5 = (rnum >> 16) & 15u;
    seed6 = (rnum >> 20) & 15u;
    seed7 = (rnum >> 24) & 15u;
    seed8 = (rnum >> 28) & 15u;
    seed9 = (rnum >> 18) & 15u;
    seed10 = (rnum >> 22) & 15u;
    seed11 = (rnum >> 26) & 15u;
    seed12 = ((rnum >> 30) | (rnum << 2)) & 15u;
    seed1 *= seed1;
    seed2 *= seed2;
    seed3 *= seed3;
    seed4 *= seed4;
    seed5 *= seed5;
    seed6 *= seed6;
    seed7 *= seed7;
    seed8 *= seed8;
    seed9 *= seed9;
    seed10 *= seed10;
    seed11 *= seed11;
    seed12 *= seed12;
    if (seed & 1u) {
        sh1 = (seed & 2u) ? 4u : 5u;
        sh2 = partition_count == 3u ? 6u : 5u;
    } else {
        sh1 = partition_count == 3u ? 6u : 5u;
        sh2 = (seed & 2u) ? 4u : 5u;
    }
    sh3 = (seed & 0x10u) ? sh1 : sh2;
    seed1 >>= sh1;
    seed2 >>= sh2;
    seed3 >>= sh1;
    seed4 >>= sh2;
    seed5 >>= sh1;
    seed6 >>= sh2;
    seed7 >>= sh1;
    seed8 >>= sh2;
    seed9 >>= sh3;
    seed10 >>= sh3;
    seed11 >>= sh3;
    seed12 >>= sh3;
    a = seed1 * x + seed2 * y + (rnum >> 14);
    b = seed3 * x + seed4 * y + (rnum >> 10);
    c = seed5 * x + seed6 * y + (rnum >> 6);
    d = seed7 * x + seed8 * y + (rnum >> 2);
    a &= 63u;
    b &= 63u;
    c &= partition_count > 2u ? 63u : 0u;
    d &= partition_count > 3u ? 63u : 0u;
    if (a >= b && a >= c && a >= d) return 0;
    if (b >= c && b >= d) return 1;
    if (c >= d) return 2;
    return 3;
}

static void tc_astc_build_partition_cache(tc_astc_encode_context *ctx,
                                          uint32_t partition_count,
                                          tc_astc_partition_info cache[1024],
                                          uint32_t *out_count) {
    uint32_t seed, texel_count = ctx->texel_count;
    int small_block = texel_count < 32u;
    *out_count = 0;
    for (seed = 0; seed < 1024u; ++seed) {
        tc_astc_partition_info *pi = cache + *out_count;
        uint32_t i, p, counts[4] = {0, 0, 0, 0};
        for (i = 0; i < texel_count; ++i) {
            uint32_t x = i % ctx->block_x;
            uint32_t y = i / ctx->block_x;
            uint8_t part =
                tc_astc_select_partition(seed, x, y, partition_count, small_block);
            if (part >= partition_count) part = (uint8_t)(partition_count - 1u);
            pi->partition_of_texel[i] = part;
            pi->texels_of_partition[part][counts[part]++] = (uint8_t)i;
        }
        for (p = 0; p < partition_count; ++p) {
            if (!counts[p]) break;
        }
        if (p != partition_count) continue;
        pi->partition_index = (uint16_t)seed;
        for (p = 0; p < partition_count; ++p)
            pi->partition_texel_count[p] = (uint8_t)counts[p];
        ++*out_count;
    }
}

static void tc_astc_encode_context_init(tc_astc_encode_context *ctx,
                                        uint32_t block_x, uint32_t block_y,
                                        int quality) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->block_x = block_x;
    ctx->block_y = block_y;
    ctx->texel_count = block_x * block_y;
    ctx->quality = quality;
    tc_astc_build_color_pquant_lut(ctx);
    tc_astc_build_partition_cache(ctx, 2u, ctx->part2_cache, &ctx->part2_count);
    if (quality > 1)
        tc_astc_build_partition_cache(ctx, 3u, ctx->part3_cache, &ctx->part3_count);
    if (quality > 1)
        tc_astc_build_partition_cache(ctx, 4u, ctx->part4_cache, &ctx->part4_count);
}

static uint32_t tc_astc_get_candidates(
    tc_astc_encode_context *ctx, uint32_t endpoint_end_bit,
    const tc_astc_block_mode_info **out_candidates) {
    uint32_t i, empty = 4u;
    for (i = 0; i < 4u; ++i) {
        if (ctx->candidate_cache[i].valid &&
            ctx->candidate_cache[i].endpoint_end_bit == endpoint_end_bit) {
            *out_candidates = ctx->candidate_cache[i].candidates;
            return ctx->candidate_cache[i].count;
        }
        if (!ctx->candidate_cache[i].valid && empty == 4u) empty = i;
    }
    if (empty == 4u) {
        *out_candidates = NULL;
        return 0;
    }
    ctx->candidate_cache[empty].endpoint_end_bit = (uint8_t)endpoint_end_bit;
    ctx->candidate_cache[empty].count = tc_astc_build_block_mode_candidates(
        ctx->block_x, ctx->block_y, endpoint_end_bit, ctx->quality,
        ctx->candidate_cache[empty].candidates);
    ctx->candidate_cache[empty].valid = 1u;
    *out_candidates = ctx->candidate_cache[empty].candidates;
    return ctx->candidate_cache[empty].count;
}

static uint32_t tc_astc_texel_weights_2d(uint32_t block_x, uint32_t block_y,
                                         uint32_t weight_x, uint32_t weight_y,
                                         uint32_t x, uint32_t y,
                                         uint8_t idx[4], uint8_t contrib[4]) {
    uint32_t n = 0;
    uint32_t x_weight =
        (((1024u + block_x / 2u) / (block_x - 1u)) * x * (weight_x - 1u) + 32u) >>
        6;
    uint32_t y_weight =
        (((1024u + block_y / 2u) / (block_y - 1u)) * y * (weight_y - 1u) + 32u) >>
        6;
    uint32_t x_frac = x_weight & 15u;
    uint32_t y_frac = y_weight & 15u;
    uint32_t x_int = x_weight >> 4;
    uint32_t y_int = y_weight >> 4;
    uint32_t qweight[4];
    uint32_t weight[4];
    uint32_t prod = x_frac * y_frac;
    qweight[0] = x_int + y_int * weight_x;
    qweight[1] = qweight[0] + 1u;
    qweight[2] = qweight[0] + weight_x;
    qweight[3] = qweight[2] + 1u;
    weight[3] = (prod + 8u) >> 4;
    weight[1] = x_frac - weight[3];
    weight[2] = y_frac - weight[3];
    weight[0] = 16u - x_frac - y_frac + weight[3];
    for (prod = 0; prod < 4u; ++prod) {
        if (weight[prod]) {
            idx[n] = (uint8_t)qweight[prod];
            contrib[n] = (uint8_t)weight[prod];
            ++n;
        }
    }
    return n;
}

static const tc_astc_decim_cache_entry *tc_astc_get_decim_cache(
    tc_astc_encode_context *ctx, uint32_t weight_x, uint32_t weight_y) {
    uint32_t i;
    for (i = 0; i < ctx->decim_cache_count; ++i) {
        if (ctx->decim_cache[i].weight_x == weight_x &&
            ctx->decim_cache[i].weight_y == weight_y)
            return ctx->decim_cache + i;
    }
    if (ctx->decim_cache_count >= 169u) return NULL;
    {
        tc_astc_decim_cache_entry *entry = ctx->decim_cache + ctx->decim_cache_count;
        entry->block_x = (uint8_t)ctx->block_x;
        entry->block_y = (uint8_t)ctx->block_y;
        entry->weight_x = (uint8_t)weight_x;
        entry->weight_y = (uint8_t)weight_y;
        for (i = 0; i < ctx->texel_count; ++i) {
            uint32_t x = i % ctx->block_x;
            uint32_t y = i / ctx->block_x;
            entry->tw_count[i] = (uint8_t)tc_astc_texel_weights_2d(
                ctx->block_x, ctx->block_y, weight_x, weight_y, x, y,
                entry->tw_idx[i], entry->tw_contrib[i]);
        }
        ++ctx->decim_cache_count;
        return entry;
    }
}

static uint32_t tc_astc_infilled_weight_table(const uint8_t weights[64],
                                              const uint8_t idx[4],
                                              const uint8_t contrib[4],
                                              uint32_t n,
                                              uint32_t quant_method) {
    uint32_t i, sum = 8u;
    for (i = 0; i < n; ++i)
        sum += tc_astc_weight_unquant(weights[idx[i]], quant_method) * contrib[i];
    return sum >> 4;
}

static uint64_t tc_astc_try_endpoint_axis(const uint8_t block[144][4],
                                          uint32_t count, uint32_t axis,
                                          uint32_t weight_x, uint32_t weight_y,
                                          uint32_t quant_method,
                                          const uint8_t tw_idx[144][4],
                                          const uint8_t tw_contrib[144][4],
                                          const uint8_t tw_count[144],
                                          uint32_t best_lo[4],
                                          uint32_t best_hi[4],
                                          uint8_t best_weights[64]) {
    uint8_t weights[64];
    uint32_t weight_count = weight_x * weight_y;
    uint32_t lo[4], hi[4];
    uint32_t i, c, lo_i = 0, hi_i = 0;
    uint32_t lo_key = 0xffffffffu, hi_key = 0u;
    uint32_t dr, dg, db, da, denom;
    uint32_t weight_accum[64];
    uint32_t weight_contrib[64];
    uint64_t err = 0;

    for (i = 0; i < count; ++i) {
        uint32_t key = tc_astc_axis_value(block[i], axis);
        if (key < lo_key) {
            lo_key = key;
            lo_i = i;
        }
        if (key > hi_key) {
            hi_key = key;
            hi_i = i;
        }
    }

    for (c = 0; c < 4u; ++c) {
        lo[c] = block[lo_i][c];
        hi[c] = block[hi_i][c];
    }

    dr = hi[0] > lo[0] ? hi[0] - lo[0] : lo[0] - hi[0];
    dg = hi[1] > lo[1] ? hi[1] - lo[1] : lo[1] - hi[1];
    db = hi[2] > lo[2] ? hi[2] - lo[2] : lo[2] - hi[2];
    da = hi[3] > lo[3] ? hi[3] - lo[3] : lo[3] - hi[3];
    denom = dr * dr + dg * dg + db * db + da * da;
    if (!denom) return UINT64_MAX;

    for (i = 0; i < weight_count; ++i) {
        weight_accum[i] = 0;
        weight_contrib[i] = 0;
    }

    for (i = 0; i < count; ++i) {
        uint32_t n, j;
        const uint8_t *p = block[i];
        int32_t dot = 0;
        uint32_t ideal;
        for (c = 0; c < 4u; ++c) {
            dot += (int32_t)((int32_t)p[c] - (int32_t)lo[c]) *
                   (int32_t)((int32_t)hi[c] - (int32_t)lo[c]);
        }
        if (dot <= 0) {
            ideal = 0;
        } else {
            uint32_t w = ((uint32_t)dot * 64u + denom / 2u) / denom;
            ideal = w > 64u ? 64u : w;
        }
        n = tw_count[i];
        for (j = 0; j < n; ++j) {
            weight_accum[tw_idx[i][j]] += ideal * tw_contrib[i][j];
            weight_contrib[tw_idx[i][j]] += tw_contrib[i][j];
        }
    }

    for (i = 0; i < weight_count; ++i) {
        uint32_t maxq = tc_astc_quant_levels(quant_method) - 1u;
        uint32_t ideal = weight_contrib[i]
                             ? (weight_accum[i] + weight_contrib[i] / 2u) /
                                   weight_contrib[i]
                             : 0u;
        uint32_t q = (ideal * maxq + 32u) / 64u;
        weights[i] = (uint8_t)(q > maxq ? maxq : q);
    }

    for (i = 0; i < count; ++i) {
        uint32_t w = tc_astc_infilled_weight_table(weights, tw_idx[i], tw_contrib[i],
                                                   tw_count[i], quant_method);
        for (c = 0; c < 4u; ++c) {
            uint32_t recon = (lo[c] * (64u - w) + hi[c] * w + 32u) >> 6;
            int32_t d = (int32_t)block[i][c] - (int32_t)recon;
            err += (uint64_t)(d * d);
        }
    }

    for (c = 0; c < 4u; ++c) {
        best_lo[c] = lo[c];
        best_hi[c] = hi[c];
    }
    /* Weights are returned as unscrambled quantized indices; callers must
     * scramble with tc_astc_scramble_weights() before ISE-encoding them. */
    memcpy(best_weights, weights, weight_count);
    return err;
}

static void tc_astc_scramble_weights(uint8_t *weights, uint32_t count,
                                     uint32_t quant_method) {
    uint32_t i;
    for (i = 0; i < count; ++i)
        weights[i] = tc_astc_weight_scramble(weights[i], quant_method);
}

/* Inverts unscrambled quantized weights (q -> maxq - q). The unquantized
 * weight tables are symmetric, so this reproduces exactly the mirrored
 * interpolation needed when endpoint pairs are swapped. */
static void tc_astc_invert_weights(uint8_t *weights, uint32_t count,
                                   uint32_t quant_method) {
    uint32_t i, maxq = tc_astc_quant_levels(quant_method) - 1u;
    for (i = 0; i < count; ++i)
        weights[i] = (uint8_t)(maxq - weights[i]);
}

/* Signed rounded division, den > 0. Avoids C's truncation-toward-zero
 * asymmetry so the integer least-squares solve matches the real solution
 * within one step. */
static int64_t tc_div_round_s64(int64_t num, int64_t den) {
    int64_t half = den / 2;
    if (num >= 0) return (num + half) / den;
    return -((-num + half) / den);
}

/* Expands the quantized decimated weight grid into per-texel interpolated
 * weights [0,64]. */
static void tc_astc_infill_weights(const uint8_t *weights,
                                   const tc_astc_decim_cache_entry *decim,
                                   uint32_t count, uint32_t quant_method,
                                   uint8_t wt[144]) {
    uint32_t i;
    for (i = 0; i < count; ++i) {
        wt[i] = (uint8_t)tc_astc_infilled_weight_table(
            weights, decim->tw_idx[i], decim->tw_contrib[i], decim->tw_count[i],
            quant_method);
    }
}

/* Least-squares endpoint solve: given the per-texel interpolated weights,
 * finds the endpoint pair minimizing the sum of squared errors of
 * lerp(lo, hi, w/64) per channel (2x2 normal equations). Returns 0 when the
 * system is degenerate (all weights equal). Magnitude bound: every term is
 * at most 64 * (144*64*255) * (144*64^2) < 2^47, safely inside int64. */
static int tc_astc_lsq_endpoints(const uint8_t block[144][4], uint32_t count,
                                 const uint8_t wt[144], uint32_t lo[4],
                                 uint32_t hi[4]) {
    int64_t saa = 0, sab = 0, sbb = 0, det;
    int64_t sap[4] = {0, 0, 0, 0}, sbp[4] = {0, 0, 0, 0};
    uint32_t i, c;
    for (i = 0; i < count; ++i) {
        int64_t a = 64 - wt[i];
        int64_t b = wt[i];
        saa += a * a;
        sab += a * b;
        sbb += b * b;
        for (c = 0; c < 4u; ++c) {
            sap[c] += a * block[i][c];
            sbp[c] += b * block[i][c];
        }
    }
    det = saa * sbb - sab * sab;
    if (det <= 0) return 0;
    for (c = 0; c < 4u; ++c) {
        int64_t l = tc_div_round_s64((sap[c] * sbb - sbp[c] * sab) * 64, det);
        int64_t h = tc_div_round_s64((sbp[c] * saa - sap[c] * sab) * 64, det);
        lo[c] = (uint32_t)tc_clamp_i32((int32_t)l, 0, 255);
        hi[c] = (uint32_t)tc_clamp_i32((int32_t)h, 0, 255);
    }
    return 1;
}

/* Quantization-aware reconstruction error of a single-partition encoding:
 * round-trips the endpoints through the color quantizer exactly as they
 * will be emitted for this endpoint format, then evaluates the interpolated
 * weights against the source texels. Matches the reference decoder model
 * (CEM 8/12 endpoint swaps are error-neutral thanks to exact weight
 * mirroring, so no swap handling is needed for scoring). */
static uint64_t tc_astc_eval_single_sse(const tc_astc_encode_context *ctx,
                                        const uint8_t block[144][4],
                                        uint32_t count, const uint8_t wt[144],
                                        uint32_t endpoint_format,
                                        const uint32_t lo[4],
                                        const uint32_t hi[4],
                                        uint32_t color_quant_method) {
    uint32_t e0[4], e1[4];
    uint32_t i, c;
    uint64_t err = 0;
    switch (endpoint_format) {
        case 0u:
            e0[0] = e0[1] = e0[2] =
                tc_astc_color_roundtrip(ctx, color_quant_method, lo[0]);
            e1[0] = e1[1] = e1[2] =
                tc_astc_color_roundtrip(ctx, color_quant_method, hi[0]);
            e0[3] = e1[3] = 255u;
            break;
        case 4u:
            e0[0] = e0[1] = e0[2] =
                tc_astc_color_roundtrip(ctx, color_quant_method, lo[0]);
            e1[0] = e1[1] = e1[2] =
                tc_astc_color_roundtrip(ctx, color_quant_method, hi[0]);
            e0[3] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[3]);
            e1[3] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[3]);
            break;
        case 6u:
        case 10u: {
            uint32_t sum_lo = lo[0] + lo[1] + lo[2];
            uint32_t sum_hi = hi[0] + hi[1] + hi[2];
            uint32_t scale =
                sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
            if (scale > 255u) scale = 255u;
            scale = tc_astc_color_roundtrip(ctx, color_quant_method, scale);
            for (c = 0; c < 3u; ++c) {
                e1[c] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[c]);
                e0[c] = (e1[c] * scale) >> 8;
            }
            if (endpoint_format == 10u) {
                e0[3] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[3]);
                e1[3] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[3]);
            } else {
                e0[3] = e1[3] = 255u;
            }
            break;
        }
        default: /* CEM 8 / 12: direct per-channel pairs. */
            for (c = 0; c < 3u; ++c) {
                e0[c] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[c]);
                e1[c] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[c]);
            }
            if (endpoint_format == 12u) {
                e0[3] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[3]);
                e1[3] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[3]);
            } else {
                e0[3] = e1[3] = 255u;
            }
            break;
    }
    for (i = 0; i < count; ++i) {
        uint32_t w = wt[i];
        for (c = 0; c < 4u; ++c) {
            uint32_t recon = (e0[c] * (64u - w) + e1[c] * w + 32u) >> 6;
            int32_t d = (int32_t)block[i][c] - (int32_t)recon;
            err += (uint64_t)(d * d);
        }
    }
    return err;
}

/* Dual-plane variant: RGB follows the color-plane weights, alpha the
 * alpha-plane weights; endpoints round-trip through the color quantizer
 * with CEM 10/12 semantics. */
static uint64_t tc_astc_eval_dual_sse(const tc_astc_encode_context *ctx,
                                      const uint8_t block[144][4],
                                      uint32_t count, const uint8_t wtc[144],
                                      const uint8_t wta[144],
                                      uint32_t endpoint_format,
                                      const uint32_t lo[4],
                                      const uint32_t hi[4],
                                      uint32_t color_quant_method) {
    uint32_t e0[4], e1[4];
    uint32_t i, c;
    uint64_t err = 0;
    if (endpoint_format == 10u) {
        uint32_t sum_lo = lo[0] + lo[1] + lo[2];
        uint32_t sum_hi = hi[0] + hi[1] + hi[2];
        uint32_t scale = sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
        if (scale > 255u) scale = 255u;
        scale = tc_astc_color_roundtrip(ctx, color_quant_method, scale);
        for (c = 0; c < 3u; ++c) {
            e1[c] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[c]);
            e0[c] = (e1[c] * scale) >> 8;
        }
    } else {
        for (c = 0; c < 3u; ++c) {
            e0[c] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[c]);
            e1[c] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[c]);
        }
    }
    e0[3] = tc_astc_color_roundtrip(ctx, color_quant_method, lo[3]);
    e1[3] = tc_astc_color_roundtrip(ctx, color_quant_method, hi[3]);
    for (i = 0; i < count; ++i) {
        for (c = 0; c < 4u; ++c) {
            uint32_t w = c == 3u ? wta[i] : wtc[i];
            uint32_t recon = (e0[c] * (64u - w) + e1[c] * w + 32u) >> 6;
            int32_t d = (int32_t)block[i][c] - (int32_t)recon;
            err += (uint64_t)(d * d);
        }
    }
    return err;
}

static int tc_astc_color_quant_supported(uint32_t quant_method) {
    if (quant_method == 20u) return 1;
    return quant_method >= 4u && quant_method <= 19u &&
           tc_astc_color_pquant_to_uquant[quant_method - 4u] != NULL;
}

static uint8_t tc_astc_quant_color_pquant_slow(uint32_t v, uint32_t quant_method) {
    uint32_t levels = tc_astc_quant_levels(quant_method);
    uint32_t best = 0, best_err = UINT32_MAX, i;
    const uint8_t *table;
    if (quant_method == 20u) return (uint8_t)v;
    table = tc_astc_color_pquant_to_uquant[quant_method - 4u];
    for (i = 0; i < levels; ++i) {
        uint32_t u = table[i];
        uint32_t err = u > v ? u - v : v - u;
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    return (uint8_t)best;
}

static void tc_astc_build_color_pquant_lut(tc_astc_encode_context *ctx) {
    uint32_t q, v;
    for (q = 0; q < 21u; ++q) {
        for (v = 0; v < 256u; ++v) {
            ctx->color_pquant_lut[q][v] =
                tc_astc_color_quant_supported(q)
                    ? tc_astc_quant_color_pquant_slow(v, q)
                    : 0u;
        }
    }
}

static int tc_astc_choose_color_quant(uint32_t value_count, uint32_t bit_budget,
                                      uint32_t *quant_method) {
    int q;
    for (q = 20; q >= 4; --q) {
        uint32_t qm = (uint32_t)q;
        if (!tc_astc_color_quant_supported(qm)) continue;
        if (tc_astc_ise_sequence_bitcount(value_count, qm) <= bit_budget) {
            *quant_method = qm;
            return 1;
        }
    }
    return 0;
}

static void tc_astc_quantize_color_values(const tc_astc_encode_context *ctx,
                                          uint32_t quant_method,
                                          uint32_t value_count,
                                          const uint8_t in_values[8],
                                          uint8_t out_values[8]) {
    uint32_t i;
    for (i = 0; i < value_count; ++i)
        out_values[i] = ctx->color_pquant_lut[quant_method][in_values[i]];
}

/* Unquantized value of an ISE color symbol. */
static uint32_t tc_astc_color_symbol_uquant(uint32_t quant_method, uint32_t sym) {
    if (quant_method >= 20u) return sym;
    return tc_astc_color_pquant_to_uquant[quant_method - 4u][sym];
}

/* Value as the decoder will reconstruct it after quantization. */
static uint32_t tc_astc_color_roundtrip(const tc_astc_encode_context *ctx,
                                        uint32_t quant_method, uint32_t v) {
    return tc_astc_color_symbol_uquant(quant_method,
                                       ctx->color_pquant_lut[quant_method][v]);
}

static uint64_t tc_astc_rgb_sse_from_stats(const uint64_t sum[3],
                                           const uint64_t sumsq[3],
                                           uint32_t count) {
    uint32_t c;
    uint64_t err = 0;
    if (!count) return UINT64_MAX;
    for (c = 0; c < 3u; ++c) {
        err += sumsq[c] - (sum[c] * sum[c] + count / 2u) / count;
    }
    return err;
}

static int tc_astc_block_has_rgb_clusters(const uint8_t block[144][4],
                                          uint32_t count,
                                          uint32_t wanted_count) {
    uint32_t centers[4][3];
    uint32_t cluster_count = 0;
    uint32_t i, c;
    for (i = 0; i < count; ++i) {
        uint32_t best_dist = UINT32_MAX;
        for (c = 0; c < cluster_count; ++c) {
            int32_t dr = (int32_t)block[i][0] - (int32_t)centers[c][0];
            int32_t dg = (int32_t)block[i][1] - (int32_t)centers[c][1];
            int32_t db = (int32_t)block[i][2] - (int32_t)centers[c][2];
            uint32_t dist = (uint32_t)(dr * dr + dg * dg + db * db);
            if (dist < best_dist) best_dist = dist;
        }
        if (cluster_count == 0u || best_dist > 4096u) {
            if (cluster_count >= wanted_count) return 1;
            centers[cluster_count][0] = block[i][0];
            centers[cluster_count][1] = block[i][1];
            centers[cluster_count][2] = block[i][2];
            ++cluster_count;
        }
    }
    return cluster_count >= wanted_count;
}

static int tc_astc_find_best_partition(const uint8_t block[144][4],
                                       const tc_astc_partition_info cache[1024],
                                       uint32_t cache_count,
                                       uint32_t partition_count,
                                       const tc_astc_encode_context *ctx,
                                       const tc_astc_partition_info **out_pi) {
    uint32_t i, c, p;
    uint64_t sum_all[3] = {0, 0, 0}, sumsq_all[3] = {0, 0, 0};
    uint64_t base_err, best_err = UINT64_MAX;
    *out_pi = NULL;
    for (i = 0; i < ctx->texel_count; ++i) {
        for (c = 0; c < 3u; ++c) {
            uint32_t v = block[i][c];
            sum_all[c] += v;
            sumsq_all[c] += (uint64_t)v * v;
        }
    }
    base_err = tc_astc_rgb_sse_from_stats(sum_all, sumsq_all, ctx->texel_count);
    if (base_err < 4096u) return 0;

    for (i = 0; i < cache_count; ++i) {
        const tc_astc_partition_info *pi = cache + i;
        uint64_t err = 0;
        for (p = 0; p < partition_count; ++p) {
            uint64_t sum[3] = {0, 0, 0}, sumsq[3] = {0, 0, 0};
            uint32_t n = pi->partition_texel_count[p];
            uint32_t ti;
            for (ti = 0; ti < n; ++ti) {
                uint32_t idx = pi->texels_of_partition[p][ti];
                for (c = 0; c < 3u; ++c) {
                    uint32_t v = block[idx][c];
                    sum[c] += v;
                    sumsq[c] += (uint64_t)v * v;
                }
            }
            err += tc_astc_rgb_sse_from_stats(sum, sumsq, n);
        }
        if (err < best_err) {
            best_err = err;
            *out_pi = pi;
        }
    }
    return *out_pi && best_err * 4u < base_err * 3u;
}

static int tc_astc_partition_rgb_scale_ok(const uint32_t lo[3],
                                          const uint32_t hi[3]) {
    uint32_t range = (hi[0] - lo[0]) + (hi[1] - lo[1]) + (hi[2] - lo[2]);
    uint32_t sum_hi = hi[0] + hi[1] + hi[2];
    uint32_t c, scale;
    if (range <= 24u) return 1;
    if (!sum_hi) return 0;
    scale = ((lo[0] + lo[1] + lo[2]) * 256u + sum_hi / 2u) / sum_hi;
    if (scale > 255u) scale = 255u;
    for (c = 0; c < 3u; ++c) {
        uint32_t recon = (hi[c] * scale + 128u) >> 8;
        uint32_t diff = recon > lo[c] ? recon - lo[c] : lo[c] - recon;
        if (diff > 24u) return 0;
    }
    return 1;
}

static uint32_t tc_astc_partition_endpoint_type_code(uint32_t partition_count,
                                                     const uint8_t formats[4]) {
    uint32_t i, bitpos, encoded_type, low_class = 4u;
    for (i = 0; i < partition_count; ++i) {
        uint32_t cls = formats[i] >> 2;
        if (cls < low_class) low_class = cls;
    }
    if (low_class == 3u) low_class = 2u;
    encoded_type = low_class + 1u;
    bitpos = 2u;
    for (i = 0; i < partition_count; ++i) {
        encoded_type |= (((uint32_t)formats[i] >> 2) - low_class) << bitpos;
        ++bitpos;
    }
    for (i = 0; i < partition_count; ++i) {
        encoded_type |= ((uint32_t)formats[i] & 3u) << bitpos;
        bitpos += 2u;
    }
    return encoded_type;
}

static uint32_t tc_astc_partition_endpoint_value_count(uint32_t partition_count,
                                                       const uint8_t formats[4]) {
    uint32_t i, count = 0;
    for (i = 0; i < partition_count; ++i)
        count += (((uint32_t)formats[i] >> 2) + 1u) * 2u;
    return count;
}

static uint32_t tc_astc_emit_partition_endpoint_values(
    const tc_astc_encode_context *ctx, uint32_t quant_method,
    uint32_t partition_count, const uint8_t formats[4],
    const uint32_t part_lo[4][4], const uint32_t part_hi[4][4],
    uint8_t color_values[18]) {
    uint32_t i, n = 0;
    for (i = 0; i < partition_count; ++i) {
        if (formats[i] == 6u) {
            uint32_t sum_lo = part_lo[i][0] + part_lo[i][1] + part_lo[i][2];
            uint32_t sum_hi = part_hi[i][0] + part_hi[i][1] + part_hi[i][2];
            uint32_t scale = sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
            if (scale > 255u) scale = 255u;
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][0]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][1]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][2]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][scale];
        } else if (formats[i] == 10u) {
            uint32_t sum_lo = part_lo[i][0] + part_lo[i][1] + part_lo[i][2];
            uint32_t sum_hi = part_hi[i][0] + part_hi[i][1] + part_hi[i][2];
            uint32_t scale = sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
            if (scale > 255u) scale = 255u;
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][0]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][1]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][2]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][scale];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_lo[i][3]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][3]];
        } else if (formats[i] == 12u) {
            uint32_t c;
            for (c = 0; c < 4u; ++c) {
                color_values[n++] = ctx->color_pquant_lut[quant_method][part_lo[i][c]];
                color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][c]];
            }
        } else {
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_lo[i][0]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][0]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_lo[i][1]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][1]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_lo[i][2]];
            color_values[n++] = ctx->color_pquant_lut[quant_method][part_hi[i][2]];
        }
    }
    return n;
}

/* Fits one shared decimated weight grid against per-partition endpoint
 * lines: each texel's ideal weight comes from projecting onto its own
 * partition's lo->hi diagonal. Returns the reconstruction SSE with the
 * (unquantized) per-partition endpoints. */
static uint64_t tc_astc_try_partition_fit(
    const uint8_t block[144][4], uint32_t count,
    const tc_astc_partition_info *pi, uint32_t partition_count,
    const uint32_t part_lo[4][4], const uint32_t part_hi[4][4],
    uint32_t quant_method, const uint8_t tw_idx[144][4],
    const uint8_t tw_contrib[144][4], const uint8_t tw_count[144],
    uint32_t weight_count, uint8_t out_weights[64]) {
    uint32_t denom[4];
    uint32_t weight_accum[64];
    uint32_t weight_contrib[64];
    uint32_t i, c, p;
    uint64_t err = 0;

    for (p = 0; p < partition_count; ++p) {
        uint32_t d = 0;
        for (c = 0; c < 4u; ++c) {
            uint32_t range = part_hi[p][c] - part_lo[p][c];
            d += range * range;
        }
        denom[p] = d;
    }
    for (i = 0; i < weight_count; ++i) {
        weight_accum[i] = 0;
        weight_contrib[i] = 0;
    }
    for (i = 0; i < count; ++i) {
        uint32_t part = pi->partition_of_texel[i];
        const uint8_t *px = block[i];
        int32_t dot = 0;
        uint32_t ideal, n, j;
        if (!denom[part]) continue; /* solid partition: no constraint */
        for (c = 0; c < 4u; ++c) {
            dot += (int32_t)((int32_t)px[c] - (int32_t)part_lo[part][c]) *
                   (int32_t)((int32_t)part_hi[part][c] -
                             (int32_t)part_lo[part][c]);
        }
        if (dot <= 0) {
            ideal = 0;
        } else {
            uint32_t w = ((uint32_t)dot * 64u + denom[part] / 2u) / denom[part];
            ideal = w > 64u ? 64u : w;
        }
        n = tw_count[i];
        for (j = 0; j < n; ++j) {
            weight_accum[tw_idx[i][j]] += ideal * tw_contrib[i][j];
            weight_contrib[tw_idx[i][j]] += tw_contrib[i][j];
        }
    }
    for (i = 0; i < weight_count; ++i) {
        uint32_t maxq = tc_astc_quant_levels(quant_method) - 1u;
        uint32_t ideal = weight_contrib[i]
                             ? (weight_accum[i] + weight_contrib[i] / 2u) /
                                   weight_contrib[i]
                             : 0u;
        uint32_t q = (ideal * maxq + 32u) / 64u;
        out_weights[i] = (uint8_t)(q > maxq ? maxq : q);
    }
    for (i = 0; i < count; ++i) {
        uint32_t part = pi->partition_of_texel[i];
        uint32_t w = tc_astc_infilled_weight_table(out_weights, tw_idx[i],
                                                   tw_contrib[i], tw_count[i],
                                                   quant_method);
        for (c = 0; c < 4u; ++c) {
            uint32_t recon =
                (part_lo[part][c] * (64u - w) + part_hi[part][c] * w + 32u) >>
                6;
            int32_t d = (int32_t)block[i][c] - (int32_t)recon;
            err += (uint64_t)(d * d);
        }
    }
    return err;
}

static int tc_encode_astc_partition_rgb_block(const uint8_t block[144][4],
                                              uint32_t count,
                                              uint32_t partition_count,
                                              tc_astc_encode_context *ctx,
                                              uint8_t out[16],
                                              uint64_t *out_err) {
    const tc_astc_block_mode_info *candidates;
    const tc_astc_decim_cache_entry *decim;
    uint8_t weightbuf[16], weights[64], color_values[18];
    uint8_t endpoint_formats[4];
    uint32_t cand_count, ci, c, i, bitpos, best = 0xffffffffu;
    uint32_t part_lo[4][4], part_hi[4][4];
    const tc_astc_partition_info *pi = NULL;
    const tc_astc_partition_info *cache = ctx->part2_cache;
    uint32_t cache_count = ctx->part2_count;
    uint32_t endpoint_format = partition_count == 4u ? 6u : 8u;
    uint32_t endpoint_values_per_partition = endpoint_format == 6u ? 4u : 6u;
    uint32_t endpoint_value_count = partition_count * endpoint_values_per_partition;
    uint32_t color_quant_method = 14u;
    uint32_t endpoint_highpart_size = 0u;
    int endpoint_formats_matched = 1;
    int is_opaque = tc_astc_block_is_opaque(block, count);
    int is_luminance = tc_astc_block_is_luminance(block, count);
    uint64_t best_err = UINT64_MAX;
    if (partition_count == 4u) {
        cache = ctx->part4_cache;
        cache_count = ctx->part4_count;
        if (!tc_astc_block_has_rgb_clusters(block, count, 4u)) return 0;
    } else if (partition_count == 3u) {
        cache = ctx->part3_cache;
        cache_count = ctx->part3_count;
        if (!tc_astc_block_has_rgb_clusters(block, count, 3u)) return 0;
    }
    if (is_luminance || (!is_opaque && (partition_count != 2u || count < 25u)) ||
        !tc_astc_find_best_partition(block, cache, cache_count, partition_count, ctx,
                                     &pi))
        return 0;
    if (!is_opaque) {
        endpoint_format = tc_astc_block_is_rgb_scale(block, count) ? 10u : 12u;
        endpoint_values_per_partition = endpoint_format == 10u ? 6u : 8u;
        endpoint_value_count = partition_count * endpoint_values_per_partition;
    }

    for (i = 0; i < partition_count; ++i) {
        for (c = 0; c < 4u; ++c) {
            part_lo[i][c] = 255u;
            part_hi[i][c] = 0u;
        }
    }
    for (i = 0; i < count; ++i) {
        uint32_t part = pi->partition_of_texel[i];
        for (c = 0; c < 4u; ++c) {
            if (block[i][c] < part_lo[part][c]) part_lo[part][c] = block[i][c];
            if (block[i][c] > part_hi[part][c]) part_hi[part][c] = block[i][c];
        }
    }

    cand_count = tc_astc_get_candidates(ctx, 29u, &candidates);
    for (ci = 0; ci < cand_count; ++ci) {
        uint8_t cand_weights[64];
        uint32_t cand_color_quant;
        uint32_t color_bits_available = 99u - candidates[ci].weight_bits;
        uint64_t err;
        if (candidates[ci].weight_bits >= 99u ||
            !tc_astc_choose_color_quant(endpoint_value_count, color_bits_available,
                                        &cand_color_quant))
            continue;
        decim = tc_astc_get_decim_cache(ctx, candidates[ci].weight_x,
                                        candidates[ci].weight_y);
        if (!decim) continue;
        err = tc_astc_try_partition_fit(
            block, count, pi, partition_count,
            (const uint32_t(*)[4])part_lo, (const uint32_t(*)[4])part_hi,
            candidates[ci].quant_method, decim->tw_idx, decim->tw_contrib,
            decim->tw_count,
            (uint32_t)candidates[ci].weight_x * candidates[ci].weight_y,
            cand_weights);
        if (err < best_err) {
            best_err = err;
            best = ci;
            color_quant_method = cand_color_quant;
            memcpy(weights, cand_weights,
                   (uint32_t)candidates[ci].weight_x * candidates[ci].weight_y);
        }
    }
    if (best == 0xffffffffu) return 0;
    for (i = 0; i < partition_count; ++i) endpoint_formats[i] = (uint8_t)endpoint_format;
    if (is_opaque && (partition_count == 2u || partition_count == 3u)) {
        uint32_t scale_count = 0u;
        for (i = 0; i < partition_count; ++i) {
            if (tc_astc_partition_rgb_scale_ok(part_lo[i], part_hi[i])) {
                endpoint_formats[i] = 6u;
                ++scale_count;
            } else {
                endpoint_formats[i] = 8u;
            }
        }
        if (scale_count != 0u && scale_count != partition_count) {
            endpoint_formats_matched = 0;
        } else {
            for (i = 0; i < partition_count; ++i)
                endpoint_formats[i] = (uint8_t)endpoint_format;
        }
    }
    endpoint_value_count =
        tc_astc_partition_endpoint_value_count(partition_count, endpoint_formats);
    if (!endpoint_formats_matched) endpoint_highpart_size = 3u * partition_count - 4u;
    {
        uint32_t color_bits_available;
        if (candidates[best].weight_bits + endpoint_highpart_size >= 99u) {
            endpoint_formats_matched = 1;
            endpoint_highpart_size = 0u;
            for (i = 0; i < partition_count; ++i)
                endpoint_formats[i] = (uint8_t)endpoint_format;
            endpoint_value_count = tc_astc_partition_endpoint_value_count(
                partition_count, endpoint_formats);
        }
        color_bits_available = 99u - candidates[best].weight_bits - endpoint_highpart_size;
        if (!tc_astc_choose_color_quant(endpoint_value_count, color_bits_available,
                                        &color_quant_method)) {
            if (!endpoint_formats_matched) {
                endpoint_formats_matched = 1;
                endpoint_highpart_size = 0u;
                for (i = 0; i < partition_count; ++i)
                    endpoint_formats[i] = (uint8_t)endpoint_format;
                endpoint_value_count = tc_astc_partition_endpoint_value_count(
                    partition_count, endpoint_formats);
                color_bits_available =
                    99u - candidates[best].weight_bits - endpoint_highpart_size;
            }
            if (!tc_astc_choose_color_quant(endpoint_value_count, color_bits_available,
                                            &color_quant_method))
                return 0;
        }
    }

    endpoint_value_count = tc_astc_emit_partition_endpoint_values(
        ctx, color_quant_method, partition_count, endpoint_formats, part_lo, part_hi,
        color_values);

    /* Reconstruction error of the emitted representation: decode the
     * quantized endpoint symbols the way a conformant decoder does and
     * evaluate the infilled weights against the source texels. The caller
     * compares this against the other encoding paths. */
    {
        uint32_t uq_e0[4][4], uq_e1[4][4], vi = 0;
        uint64_t err = 0;
        decim = tc_astc_get_decim_cache(ctx, candidates[best].weight_x,
                                        candidates[best].weight_y);
        if (!decim) return 0;
        for (i = 0; i < partition_count; ++i) {
            if (endpoint_formats[i] == 6u || endpoint_formats[i] == 10u) {
                uint32_t scale;
                for (c = 0; c < 3u; ++c) {
                    uq_e1[i][c] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + c]);
                }
                scale = tc_astc_color_symbol_uquant(color_quant_method,
                                                    color_values[vi + 3u]);
                for (c = 0; c < 3u; ++c)
                    uq_e0[i][c] = (uq_e1[i][c] * scale) >> 8;
                if (endpoint_formats[i] == 10u) {
                    uq_e0[i][3] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + 4u]);
                    uq_e1[i][3] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + 5u]);
                    vi += 6u;
                } else {
                    uq_e0[i][3] = uq_e1[i][3] = 255u;
                    vi += 4u;
                }
            } else { /* CEM 8 or 12: per-channel (lo,hi) pairs. */
                uint32_t s0 = 0, s1 = 0;
                for (c = 0; c < 3u; ++c) {
                    uq_e0[i][c] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + c * 2u]);
                    uq_e1[i][c] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + c * 2u + 1u]);
                    s0 += uq_e0[i][c];
                    s1 += uq_e1[i][c];
                }
                if (endpoint_formats[i] == 12u) {
                    uq_e0[i][3] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + 6u]);
                    uq_e1[i][3] = tc_astc_color_symbol_uquant(
                        color_quant_method, color_values[vi + 7u]);
                    vi += 8u;
                } else {
                    uq_e0[i][3] = uq_e1[i][3] = 255u;
                    vi += 6u;
                }
                /* Componentwise bounds keep the pair sum-ordered; a swapped
                 * pair would decode blue-contracted, which this path cannot
                 * compensate with shared weights, so refuse the encoding. */
                if (s0 > s1) return 0;
            }
        }
        for (i = 0; i < count; ++i) {
            uint32_t part = pi->partition_of_texel[i];
            uint32_t w = tc_astc_infilled_weight_table(
                weights, decim->tw_idx[i], decim->tw_contrib[i],
                decim->tw_count[i], candidates[best].quant_method);
            for (c = 0; c < 4u; ++c) {
                uint32_t recon =
                    (uq_e0[part][c] * (64u - w) + uq_e1[part][c] * w + 32u) >>
                    6;
                int32_t d = (int32_t)block[i][c] - (int32_t)recon;
                err += (uint64_t)(d * d);
            }
        }
        *out_err = err;
    }

    memset(out, 0, 16u);
    memset(weightbuf, 0, sizeof(weightbuf));
    tc_astc_scramble_weights(weights,
                             (uint32_t)candidates[best].weight_x *
                                 candidates[best].weight_y,
                             candidates[best].quant_method);
    (void)tc_astc_ise_encode_bits(candidates[best].quant_method,
                                  (uint32_t)candidates[best].weight_x *
                                      candidates[best].weight_y,
                                  weights, weightbuf, sizeof(weightbuf), 0u);
    for (i = 0; i < 16u; ++i) out[i] = tc_bitrev8(weightbuf[15u - i]);
    bitpos = 0;
    tc_set_bits(out, &bitpos, candidates[best].block_mode, 11u);
    tc_set_bits(out, &bitpos, partition_count - 1u, 2u);
    tc_set_bits(out, &bitpos, pi->partition_index & 63u, 6u);
    tc_set_bits(out, &bitpos, pi->partition_index >> 6, 4u);
    bitpos = 23u;
    if (endpoint_formats_matched) {
        tc_set_bits(out, &bitpos, ((uint32_t)endpoint_formats[0]) << 2, 6u);
    } else {
        uint32_t endpoint_type =
            tc_astc_partition_endpoint_type_code(partition_count, endpoint_formats);
        uint32_t endpoint_highpart_pos =
            128u - candidates[best].weight_bits - endpoint_highpart_size;
        tc_set_bits(out, &bitpos, endpoint_type & 63u, 6u);
        bitpos = endpoint_highpart_pos;
        tc_set_bits(out, &bitpos, endpoint_type >> 6, endpoint_highpart_size);
    }
    (void)tc_astc_ise_encode_bits(color_quant_method, endpoint_value_count,
                                  color_values, out, 16u, 29u);
    return 1;
}

static int tc_encode_astc_dual_rgba_block(const uint8_t block[144][4],
                                          uint32_t count,
                                          int quality,
                                          tc_astc_encode_context *ctx,
                                          uint8_t out[16],
                                          uint64_t *out_err) {
    tc_astc_block_mode_info candidates[2048];
    const tc_astc_decim_cache_entry *decim;
    uint8_t weightbuf[16], weights[128], color_weights[64], alpha_weights[64];
    uint8_t values[8];
    uint32_t cand_count, ci, c, i, bitpos, best = 0xffffffffu;
    uint32_t color_lo[4], color_hi[4], alpha_lo[4], alpha_hi[4];
    uint32_t alpha_min = 255u, alpha_max = 0u;
    int is_rgb_scale = tc_astc_block_is_rgb_scale(block, count);
    uint32_t endpoint_format = is_rgb_scale ? 10u : 12u;
    uint32_t endpoint_value_count = is_rgb_scale ? 6u : 8u;
    uint32_t color_quant_method = 20u;
    uint64_t best_err = UINT64_MAX;
    if (quality < 1 || (quality < 2 && is_rgb_scale) ||
        tc_astc_block_is_opaque(block, count) ||
        tc_astc_block_is_luminance(block, count))
        return 0;
    for (i = 0; i < count; ++i) {
        if (block[i][3] < alpha_min) alpha_min = block[i][3];
        if (block[i][3] > alpha_max) alpha_max = block[i][3];
    }
    if (alpha_max - alpha_min < 48u) return 0;

    cand_count = tc_astc_build_dual_block_mode_candidates(
        ctx->block_x, ctx->block_y, 17u, quality, candidates);
    for (ci = 0; ci < cand_count; ++ci) {
        uint32_t cand_color_lo[4], cand_color_hi[4];
        uint32_t cand_alpha_lo[4], cand_alpha_hi[4];
        uint8_t cand_color_weights[64], cand_alpha_weights[64];
        uint64_t color_err, alpha_err, err;
        uint32_t cand_color_quant_method;
        uint32_t color_bits_available;
        uint32_t weight_count =
            (uint32_t)candidates[ci].weight_x * candidates[ci].weight_y;
        if (candidates[ci].weight_bits >= 109u) continue;
        color_bits_available = 109u - candidates[ci].weight_bits;
        if (!tc_astc_choose_color_quant(endpoint_value_count,
                                        color_bits_available,
                                        &cand_color_quant_method))
            continue;
        decim = tc_astc_get_decim_cache(ctx, candidates[ci].weight_x,
                                        candidates[ci].weight_y);
        if (!decim) continue;
        color_err = tc_astc_try_endpoint_axis(
            block, count, 4u, candidates[ci].weight_x, candidates[ci].weight_y,
            candidates[ci].quant_method, decim->tw_idx, decim->tw_contrib,
            decim->tw_count, cand_color_lo, cand_color_hi, cand_color_weights);
        alpha_err = tc_astc_try_endpoint_axis(
            block, count, 3u, candidates[ci].weight_x, candidates[ci].weight_y,
            candidates[ci].quant_method, decim->tw_idx, decim->tw_contrib,
            decim->tw_count, cand_alpha_lo, cand_alpha_hi, cand_alpha_weights);
        err = color_err + alpha_err;
        if (err < best_err) {
            best_err = err;
            best = ci;
            memcpy(color_lo, cand_color_lo, sizeof(color_lo));
            memcpy(color_hi, cand_color_hi, sizeof(color_hi));
            memcpy(alpha_lo, cand_alpha_lo, sizeof(alpha_lo));
            memcpy(alpha_hi, cand_alpha_hi, sizeof(alpha_hi));
            memcpy(color_weights, cand_color_weights, weight_count);
            memcpy(alpha_weights, cand_alpha_weights, weight_count);
            color_quant_method = cand_color_quant_method;
        }
    }
    if (best == 0xffffffffu) return 0;

    /* Quantization-aware error of the emitted dual-plane block, with a
     * least-squares refinement pass over both planes' endpoints. */
    {
        const tc_astc_decim_cache_entry *bd = tc_astc_get_decim_cache(
            ctx, candidates[best].weight_x, candidates[best].weight_y);
        uint8_t wtc[144], wta[144];
        uint32_t cur[2][4], lsq_lo[4], lsq_hi[4];
        uint64_t err;
        if (!bd) return 0;
        tc_astc_infill_weights(color_weights, bd, count,
                               candidates[best].quant_method, wtc);
        tc_astc_infill_weights(alpha_weights, bd, count,
                               candidates[best].quant_method, wta);
        for (c = 0; c < 4u; ++c) {
            cur[0][c] = c == 3u ? alpha_lo[3] : color_lo[c];
            cur[1][c] = c == 3u ? alpha_hi[3] : color_hi[c];
        }
        err = tc_astc_eval_dual_sse(ctx, block, count, wtc, wta,
                                    endpoint_format, cur[0], cur[1],
                                    color_quant_method);
        if (tc_astc_lsq_endpoints(block, count, wtc, lsq_lo, lsq_hi)) {
            uint32_t try_lo[4], try_hi[4];
            uint64_t lsq_err;
            memcpy(try_lo, cur[0], sizeof(try_lo));
            memcpy(try_hi, cur[1], sizeof(try_hi));
            for (c = 0; c < 3u; ++c) {
                try_lo[c] = lsq_lo[c];
                try_hi[c] = lsq_hi[c];
            }
            lsq_err = tc_astc_eval_dual_sse(ctx, block, count, wtc, wta,
                                            endpoint_format, try_lo, try_hi,
                                            color_quant_method);
            if (lsq_err < err) {
                err = lsq_err;
                for (c = 0; c < 3u; ++c) {
                    color_lo[c] = try_lo[c];
                    color_hi[c] = try_hi[c];
                }
                memcpy(cur[0], try_lo, sizeof(try_lo));
                memcpy(cur[1], try_hi, sizeof(try_hi));
            }
        }
        if (tc_astc_lsq_endpoints(block, count, wta, lsq_lo, lsq_hi)) {
            uint32_t try_lo[4], try_hi[4];
            uint64_t lsq_err;
            memcpy(try_lo, cur[0], sizeof(try_lo));
            memcpy(try_hi, cur[1], sizeof(try_hi));
            try_lo[3] = lsq_lo[3];
            try_hi[3] = lsq_hi[3];
            lsq_err = tc_astc_eval_dual_sse(ctx, block, count, wtc, wta,
                                            endpoint_format, try_lo, try_hi,
                                            color_quant_method);
            if (lsq_err < err) {
                err = lsq_err;
                alpha_lo[3] = try_lo[3];
                alpha_hi[3] = try_hi[3];
            }
        }
        *out_err = err;
    }

    if (endpoint_format == 10u) {
        uint32_t sum_lo = color_lo[0] + color_lo[1] + color_lo[2];
        uint32_t sum_hi = color_hi[0] + color_hi[1] + color_hi[2];
        uint32_t scale = sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
        if (scale > 255u) scale = 255u;
        values[0] = (uint8_t)color_hi[0];
        values[1] = (uint8_t)color_hi[1];
        values[2] = (uint8_t)color_hi[2];
        values[3] = (uint8_t)scale;
        values[4] = (uint8_t)alpha_lo[3];
        values[5] = (uint8_t)alpha_hi[3];
    } else {
        for (c = 0; c < 3u; ++c) {
            values[c * 2u + 0u] = (uint8_t)color_lo[c];
            values[c * 2u + 1u] = (uint8_t)color_hi[c];
        }
        values[6] = (uint8_t)alpha_lo[3];
        values[7] = (uint8_t)alpha_hi[3];
    }
    {
        uint32_t weight_count =
            (uint32_t)candidates[best].weight_x * candidates[best].weight_y;
        /* CEM 12: keep the emitted pair in ascending quantized-RGB-sum order
         * so the decoder takes the direct (non-blue-contract) path; both
         * planes' weights mirror to compensate for the endpoint swap. */
        if (endpoint_format == 12u) {
            uint32_t s_lo = 0, s_hi = 0;
            for (c = 0; c < 3u; ++c) {
                s_lo += tc_astc_color_roundtrip(ctx, color_quant_method,
                                                values[c * 2u]);
                s_hi += tc_astc_color_roundtrip(ctx, color_quant_method,
                                                values[c * 2u + 1u]);
            }
            if (s_lo > s_hi) {
                for (c = 0; c < 4u; ++c) {
                    uint8_t t = values[c * 2u];
                    values[c * 2u] = values[c * 2u + 1u];
                    values[c * 2u + 1u] = t;
                }
                tc_astc_invert_weights(color_weights, weight_count,
                                       candidates[best].quant_method);
                tc_astc_invert_weights(alpha_weights, weight_count,
                                       candidates[best].quant_method);
            }
        }
        tc_astc_scramble_weights(color_weights, weight_count,
                                 candidates[best].quant_method);
        tc_astc_scramble_weights(alpha_weights, weight_count,
                                 candidates[best].quant_method);
        /* Dual-plane weights are interleaved per texel: plane 0 (color) in
         * the even ISE positions, plane 1 (alpha) in the odd ones. */
        for (i = 0; i < weight_count; ++i) {
            weights[i * 2u] = color_weights[i];
            weights[i * 2u + 1u] = alpha_weights[i];
        }
        memset(out, 0, 16u);
        memset(weightbuf, 0, sizeof(weightbuf));
        (void)tc_astc_ise_encode_bits(candidates[best].quant_method,
                                      weight_count * 2u, weights, weightbuf,
                                      sizeof(weightbuf), 0u);
    }
    for (i = 0; i < 16u; ++i) out[i] = tc_bitrev8(weightbuf[15u - i]);
    bitpos = 0;
    tc_set_bits(out, &bitpos, candidates[best].block_mode, 11u);
    tc_set_bits(out, &bitpos, 0u, 2u);
    tc_set_bits(out, &bitpos, endpoint_format, 4u);
    bitpos = 128u - candidates[best].weight_bits - 2u;
    tc_set_bits(out, &bitpos, 3u, 2u);

    {
        uint8_t packed_values[8];
        tc_astc_quantize_color_values(ctx, color_quant_method,
                                      endpoint_value_count, values,
                                      packed_values);
        (void)tc_astc_ise_encode_bits(color_quant_method, endpoint_value_count,
                                      packed_values, out, 16u, 17u);
    }
    return 1;
}

static void tc_encode_astc_ldr_block(const uint8_t block[144][4],
                                     uint32_t count,
                                     int quality,
                                     tc_astc_encode_context *ctx,
                                     uint8_t out[16]) {
    uint8_t weightbuf[16];
    uint8_t weights[64];
    uint8_t values[8];
    uint32_t bitpos;
    uint32_t i, c, axis, ci;
    uint32_t lo[4], hi[4];
    uint32_t best_block_mode = 66u;
    uint32_t best_quant_method = 2u;
    uint32_t best_weight_count = 16u;
    int is_opaque = tc_astc_block_is_opaque(block, count);
    int is_luminance = tc_astc_block_is_luminance(block, count);
    int is_rgb_scale = !is_luminance && tc_astc_block_is_rgb_scale(block, count);
    uint32_t endpoint_format = is_luminance && is_opaque
                                   ? 0u
                                   : (is_luminance
                                          ? 4u
                                          : (is_rgb_scale ? (is_opaque ? 6u : 10u)
                                                          : (is_opaque ? 8u : 12u)));
    uint32_t endpoint_value_count = endpoint_format == 0u
                                        ? 2u
                                        : (endpoint_format == 4u || endpoint_format == 6u
                                               ? 4u
                                               : (endpoint_format == 8u || endpoint_format == 10u ? 6u : 8u));
    uint32_t endpoint_end_bit = 17u + endpoint_value_count * 8u;
    uint32_t candidate_endpoint_end_bit = quality > 0 ? 17u : endpoint_end_bit;
    uint32_t color_quant_method = 20u;
    const tc_astc_block_mode_info *candidates;
    uint32_t candidate_count;
    uint16_t selected_candidates[64];
    uint64_t selected_errors[64];
    uint32_t selected_count = 0;
    uint32_t selected_limit = quality > 1 ? 48u : (quality > 0 ? 16u : 4u);
    uint64_t best_err = UINT64_MAX;

    uint8_t path_out[16];
    uint8_t cand_out[16];
    uint64_t path_err = UINT64_MAX, cand_err;
    int have_path = 0;

    memset(out, 0, 16u);
    memset(weightbuf, 0, 16u);
    memset(path_out, 0, 16u);
    /* Multi-partition and dual-plane candidates compete on reconstruction
     * error against each other and against the single-partition search
     * below; the lowest error wins the block. Fewer partitions are tried
     * first and win ties (they leave more bits for color precision). */
    if (quality > 0 &&
        tc_encode_astc_partition_rgb_block(block, count, 2u, ctx, cand_out,
                                           &cand_err) &&
        cand_err < path_err) {
        path_err = cand_err;
        memcpy(path_out, cand_out, 16u);
        have_path = 1;
    }
    if (quality > 1 &&
        tc_encode_astc_partition_rgb_block(block, count, 3u, ctx, cand_out,
                                           &cand_err) &&
        cand_err < path_err) {
        path_err = cand_err;
        memcpy(path_out, cand_out, 16u);
        have_path = 1;
    }
    if (quality > 1 &&
        tc_encode_astc_partition_rgb_block(block, count, 4u, ctx, cand_out,
                                           &cand_err) &&
        cand_err < path_err) {
        path_err = cand_err;
        memcpy(path_out, cand_out, 16u);
        have_path = 1;
    }
    if (tc_encode_astc_dual_rgba_block(block, count, quality, ctx, cand_out,
                                       &cand_err) &&
        cand_err < path_err) {
        path_err = cand_err;
        memcpy(path_out, cand_out, 16u);
        have_path = 1;
    }
    candidate_count = tc_astc_get_candidates(ctx, candidate_endpoint_end_bit, &candidates);
    if (selected_limit > candidate_count) selected_limit = candidate_count;

    for (ci = 0; ci < candidate_count; ++ci) {
        const tc_astc_decim_cache_entry *decim = tc_astc_get_decim_cache(
            ctx, candidates[ci].weight_x, candidates[ci].weight_y);
        uint32_t cand_lo[4], cand_hi[4], pos;
        uint8_t cand_weights[64];
        uint32_t cand_color_quant;
        uint64_t err;
        if (quality > 0) {
            uint32_t color_bits_available = 111u - candidates[ci].weight_bits;
            if (candidates[ci].weight_bits >= 111u ||
                !tc_astc_choose_color_quant(endpoint_value_count,
                                            color_bits_available,
                                            &cand_color_quant))
                continue;
        }
        if (!decim) continue;
        err = tc_astc_try_endpoint_axis(
            block, count, 4u, candidates[ci].weight_x, candidates[ci].weight_y,
            candidates[ci].quant_method, decim->tw_idx, decim->tw_contrib,
            decim->tw_count, cand_lo, cand_hi, cand_weights);
        if (selected_count < selected_limit) {
            pos = selected_count++;
        } else if (selected_count && err < selected_errors[selected_count - 1u]) {
            pos = selected_count - 1u;
        } else {
            continue;
        }
        while (pos > 0u && err < selected_errors[pos - 1u]) {
            selected_errors[pos] = selected_errors[pos - 1u];
            selected_candidates[pos] = selected_candidates[pos - 1u];
            --pos;
        }
        selected_errors[pos] = err;
        selected_candidates[pos] = (uint16_t)ci;
    }

    for (ci = 0; ci < selected_count; ++ci) {
        uint32_t candidate_index = selected_candidates[ci];
        const tc_astc_decim_cache_entry *decim = tc_astc_get_decim_cache(
            ctx, candidates[candidate_index].weight_x,
            candidates[candidate_index].weight_y);
        uint32_t cand_color_quant_method = 20u;
        if (!decim) continue;
        if (quality > 0) {
            uint32_t color_bits_available =
                111u - candidates[candidate_index].weight_bits;
            if (candidates[candidate_index].weight_bits >= 111u ||
                !tc_astc_choose_color_quant(endpoint_value_count,
                                            color_bits_available,
                                            &cand_color_quant_method))
                continue;
        } else {
            cand_color_quant_method = 20u;
        }
        for (axis = quality > 0 ? 0u : 4u; axis < 5u; ++axis) {
            uint32_t cand_lo[4], cand_hi[4], lsq_lo[4], lsq_hi[4];
            uint8_t cand_weights[64];
            uint8_t wt[144];
            uint64_t err;
            (void)tc_astc_try_endpoint_axis(
                block, count, axis, candidates[candidate_index].weight_x,
                candidates[candidate_index].weight_y,
                candidates[candidate_index].quant_method, decim->tw_idx, decim->tw_contrib,
                decim->tw_count,
                cand_lo, cand_hi, cand_weights);
            /* Score with the endpoints as they will actually decode
             * (quantized), and try replacing the axis-extreme endpoints
             * with the least-squares solution for the fitted weights. */
            tc_astc_infill_weights(cand_weights, decim, count,
                                   candidates[candidate_index].quant_method,
                                   wt);
            err = tc_astc_eval_single_sse(ctx, block, count, wt,
                                          endpoint_format, cand_lo, cand_hi,
                                          cand_color_quant_method);
            if (tc_astc_lsq_endpoints(block, count, wt, lsq_lo, lsq_hi)) {
                uint64_t lsq_err = tc_astc_eval_single_sse(
                    ctx, block, count, wt, endpoint_format, lsq_lo, lsq_hi,
                    cand_color_quant_method);
                if (lsq_err < err) {
                    err = lsq_err;
                    memcpy(cand_lo, lsq_lo, sizeof(cand_lo));
                    memcpy(cand_hi, lsq_hi, sizeof(cand_hi));
                }
            }
            if (err < best_err) {
                best_err = err;
                best_block_mode = candidates[candidate_index].block_mode;
                best_quant_method = candidates[candidate_index].quant_method;
                best_weight_count =
                    (uint32_t)candidates[candidate_index].weight_x *
                    candidates[candidate_index].weight_y;
                color_quant_method = cand_color_quant_method;
                memcpy(lo, cand_lo, sizeof(lo));
                memcpy(hi, cand_hi, sizeof(hi));
                memcpy(weights, cand_weights, best_weight_count);
            }
            if (quality <= 0) break;
        }
    }

    if (best_err == UINT64_MAX) {
        uint32_t sum[4] = {0, 0, 0, 0};
        if (have_path) {
            memcpy(out, path_out, 16u);
            return;
        }
        for (i = 0; i < count; ++i) {
            for (c = 0; c < 4u; ++c) sum[c] += block[i][c];
        }
        tc_astc_write_const_from_sum(sum, count, out);
        return;
    }
    if (have_path && path_err < best_err) {
        memcpy(out, path_out, 16u);
        return;
    }

    values[0] = (uint8_t)lo[0];
    values[1] = (uint8_t)hi[0];
    if (endpoint_format == 4u) {
        values[2] = (uint8_t)lo[3];
        values[3] = (uint8_t)hi[3];
    } else if (endpoint_format == 6u || endpoint_format == 10u) {
        uint32_t sum_lo = lo[0] + lo[1] + lo[2];
        uint32_t sum_hi = hi[0] + hi[1] + hi[2];
        uint32_t scale = sum_hi ? (sum_lo * 256u + sum_hi / 2u) / sum_hi : 0u;
        if (scale > 255u) scale = 255u;
        values[0] = (uint8_t)hi[0];
        values[1] = (uint8_t)hi[1];
        values[2] = (uint8_t)hi[2];
        values[3] = (uint8_t)scale;
        if (endpoint_format == 10u) {
            values[4] = (uint8_t)lo[3];
            values[5] = (uint8_t)hi[3];
        }
    } else if (endpoint_format != 0u) {
        values[2] = (uint8_t)lo[1];
        values[3] = (uint8_t)hi[1];
        values[4] = (uint8_t)lo[2];
        values[5] = (uint8_t)hi[2];
        if (endpoint_format == 12u) {
            values[6] = (uint8_t)lo[3];
            values[7] = (uint8_t)hi[3];
        }
    }
    /* CEM 8/12: a decoder swaps the endpoints and applies blue-contract when
     * the first endpoint's quantized RGB sum is larger, so emit the pair in
     * ascending-sum order and mirror the weights to compensate. */
    if (endpoint_format == 8u || endpoint_format == 12u) {
        uint32_t s_lo = tc_astc_color_roundtrip(ctx, color_quant_method, values[0]) +
                        tc_astc_color_roundtrip(ctx, color_quant_method, values[2]) +
                        tc_astc_color_roundtrip(ctx, color_quant_method, values[4]);
        uint32_t s_hi = tc_astc_color_roundtrip(ctx, color_quant_method, values[1]) +
                        tc_astc_color_roundtrip(ctx, color_quant_method, values[3]) +
                        tc_astc_color_roundtrip(ctx, color_quant_method, values[5]);
        if (s_lo > s_hi) {
            uint32_t pair_count = endpoint_format == 12u ? 4u : 3u;
            for (i = 0; i < pair_count; ++i) {
                uint8_t t = values[i * 2u];
                values[i * 2u] = values[i * 2u + 1u];
                values[i * 2u + 1u] = t;
            }
            tc_astc_invert_weights(weights, best_weight_count,
                                   best_quant_method);
        }
    }

    tc_astc_scramble_weights(weights, best_weight_count, best_quant_method);
    (void)tc_astc_ise_encode_bits(best_quant_method, best_weight_count, weights,
                                  weightbuf, sizeof(weightbuf), 0u);
    for (i = 0; i < 16u; ++i) out[i] = tc_bitrev8(weightbuf[15u - i]);

    bitpos = 0;
    tc_set_bits(out, &bitpos, best_block_mode, 11u);
    tc_set_bits(out, &bitpos, 0u, 2u);   /* one partition. */
    tc_set_bits(out, &bitpos, endpoint_format, 4u);

    {
        uint8_t packed_values[8];
        tc_astc_quantize_color_values(ctx, color_quant_method,
                                      endpoint_value_count, values,
                                      packed_values);
        (void)tc_astc_ise_encode_bits(color_quant_method, endpoint_value_count,
                                      packed_values, out, 16u, 17u);
    }
}

#if defined(TC_X86)
TC_TARGET("sse2")
static void tc_encode_astc_const_block_sse2(const uint8_t block[16][4],
                                            uint8_t out[16]) {
    uint32_t sum[4];
    uint64_t lanes[2];
    uint32_t c;
    const __m128i masks[4] = {
        _mm_set1_epi32(0x000000ff),
        _mm_set1_epi32(0x0000ff00),
        _mm_set1_epi32(0x00ff0000),
        _mm_set1_epi32((int)0xff000000u)};
    const __m128i z = _mm_setzero_si128();
    __m128i v0 = _mm_loadu_si128((const __m128i *)(const void *)(block + 0));
    __m128i v1 = _mm_loadu_si128((const __m128i *)(const void *)(block + 4));
    __m128i v2 = _mm_loadu_si128((const __m128i *)(const void *)(block + 8));
    __m128i v3 = _mm_loadu_si128((const __m128i *)(const void *)(block + 12));
    for (c = 0; c < 4u; ++c) {
        __m128i s = _mm_sad_epu8(_mm_and_si128(v0, masks[c]), z);
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_and_si128(v1, masks[c]), z));
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_and_si128(v2, masks[c]), z));
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_and_si128(v3, masks[c]), z));
        _mm_storeu_si128((__m128i *)(void *)lanes, s);
        sum[c] = (uint32_t)(lanes[0] + lanes[1]);
    }
    tc_astc_write_const_from_sum(sum, 16u, out);
}

TC_TARGET("sse4.1")
static void tc_encode_astc_const_block_sse41(const uint8_t block[16][4],
                                             uint8_t out[16]) {
    uint32_t sum[4];
    uint64_t lanes[2];
    uint32_t c;
    const __m128i mask[4] = {
        _mm_setr_epi8(0, 4, 8, 12, -128, -128, -128, -128, -128, -128, -128,
                      -128, -128, -128, -128, -128),
        _mm_setr_epi8(1, 5, 9, 13, -128, -128, -128, -128, -128, -128, -128,
                      -128, -128, -128, -128, -128),
        _mm_setr_epi8(2, 6, 10, 14, -128, -128, -128, -128, -128, -128, -128,
                      -128, -128, -128, -128, -128),
        _mm_setr_epi8(3, 7, 11, 15, -128, -128, -128, -128, -128, -128, -128,
                      -128, -128, -128, -128, -128)};
    const __m128i z = _mm_setzero_si128();
    __m128i v0 = _mm_loadu_si128((const __m128i *)(const void *)(block + 0));
    __m128i v1 = _mm_loadu_si128((const __m128i *)(const void *)(block + 4));
    __m128i v2 = _mm_loadu_si128((const __m128i *)(const void *)(block + 8));
    __m128i v3 = _mm_loadu_si128((const __m128i *)(const void *)(block + 12));
    for (c = 0; c < 4u; ++c) {
        __m128i s = _mm_sad_epu8(_mm_shuffle_epi8(v0, mask[c]), z);
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_shuffle_epi8(v1, mask[c]), z));
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_shuffle_epi8(v2, mask[c]), z));
        s = _mm_add_epi64(s, _mm_sad_epu8(_mm_shuffle_epi8(v3, mask[c]), z));
        _mm_storeu_si128((__m128i *)(void *)lanes, s);
        sum[c] = (uint32_t)(lanes[0] + lanes[1]);
    }
    tc_astc_write_const_from_sum(sum, 16u, out);
}

TC_TARGET("avx2")
static void tc_encode_astc_const_block_avx2(const uint8_t block[16][4],
                                            uint8_t out[16]) {
    uint32_t sum[4];
    uint64_t lanes[4];
    uint32_t c;
    const __m256i mask[4] = {
        _mm256_setr_epi8(0, 4, 8, 12, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128, -128, -128, -128, 0, 4, 8, 12, -128,
                         -128, -128, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128),
        _mm256_setr_epi8(1, 5, 9, 13, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128, -128, -128, -128, 1, 5, 9, 13, -128,
                         -128, -128, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128),
        _mm256_setr_epi8(2, 6, 10, 14, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128, -128, -128, -128, 2, 6, 10, 14, -128,
                         -128, -128, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128),
        _mm256_setr_epi8(3, 7, 11, 15, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128, -128, -128, -128, 3, 7, 11, 15, -128,
                         -128, -128, -128, -128, -128, -128, -128, -128, -128,
                         -128, -128)};
    const __m256i z = _mm256_setzero_si256();
    __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)(block + 0));
    __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(block + 8));
    for (c = 0; c < 4u; ++c) {
        __m256i s = _mm256_sad_epu8(_mm256_shuffle_epi8(v0, mask[c]), z);
        s = _mm256_add_epi64(s, _mm256_sad_epu8(_mm256_shuffle_epi8(v1, mask[c]), z));
        _mm256_storeu_si256((__m256i *)(void *)lanes, s);
        sum[c] = (uint32_t)(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
    }
    tc_astc_write_const_from_sum(sum, 16u, out);
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static uint32_t tc_neon_sum_u8(uint8x16_t v) {
    uint16x8_t s16 = vpaddlq_u8(v);
    uint32x4_t s32 = vpaddlq_u16(s16);
    uint64x2_t s64 = vpaddlq_u32(s32);
    return (uint32_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
}

static void tc_encode_astc_const_block_neon(const uint8_t block[16][4],
                                            uint8_t out[16]) {
    uint32_t sum[4];
    uint8x16x4_t px = vld4q_u8((const uint8_t *)block);
    sum[0] = tc_neon_sum_u8(px.val[0]);
    sum[1] = tc_neon_sum_u8(px.val[1]);
    sum[2] = tc_neon_sum_u8(px.val[2]);
    sum[3] = tc_neon_sum_u8(px.val[3]);
    tc_astc_write_const_from_sum(sum, 16u, out);
}
#endif

static void tc_encode_astc_const_block(const uint8_t block[144][4],
                                       uint32_t count, uint8_t out[16]) {
    uint32_t i, c;
    uint32_t sum[4] = {0, 0, 0, 0};
#if defined(TC_X86)
    if (count == 16u) {
        uint32_t caps = tc_cpu_caps();
        if (caps & TC_CPU_AVX2) {
            tc_encode_astc_const_block_avx2((const uint8_t(*)[4])block, out);
            return;
        }
        if (caps & TC_CPU_SSE41) {
            tc_encode_astc_const_block_sse41((const uint8_t(*)[4])block, out);
            return;
        }
        if (caps & TC_CPU_SSE2) {
            tc_encode_astc_const_block_sse2((const uint8_t(*)[4])block, out);
            return;
        }
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (count == 16u && (tc_cpu_caps() & TC_CPU_NEON)) {
        tc_encode_astc_const_block_neon((const uint8_t(*)[4])block, out);
        return;
    }
#endif
    for (i = 0; i < count; ++i) {
        for (c = 0; c < 4u; ++c) sum[c] += block[i][c];
    }
    tc_astc_write_const_from_sum(sum, count, out);
}

tc_result tc_astc_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                 uint32_t height, size_t stride,
                                 const tc_astc_options *opt,
                                 uint8_t *out_astc, size_t out_size) {
    tc_astc_options defopt;
    uint32_t bx, by, xx, yy, x, y;
    uint8_t block[144][4];
    tc_astc_encode_context *astc_ctx;
    size_t need, off = 0;

    if (!opt) {
        tc_astc_options_init(&defopt);
        opt = &defopt;
    }
    if (!rgba || !out_astc || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (!tc_astc_valid_block(opt->block_x, opt->block_y))
        return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = tc_astc_compressed_size(width, height, opt);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;
    /* The context holds several MB of partition/decimation caches; keep it
     * off the stack (one allocation per image, not per block). */
    astc_ctx = (tc_astc_encode_context *)malloc(sizeof(*astc_ctx));
    if (!astc_ctx) return TC_ERROR_OUT_OF_MEMORY;
    tc_astc_encode_context_init(astc_ctx, opt->block_x, opt->block_y,
                                opt->quality);

    for (by = 0; by < height; by += opt->block_y) {
        for (bx = 0; bx < width; bx += opt->block_x) {
            uint32_t count = 0;
            for (yy = 0; yy < opt->block_y; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < opt->block_x; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    memcpy(block[count], src, 4u);
                    ++count;
                }
            }
            if (!tc_astc_block_is_solid(block, count)) {
                tc_encode_astc_ldr_block(block, count, opt->quality, astc_ctx,
                                         out_astc + off);
            } else {
                tc_encode_astc_const_block(block, count, out_astc + off);
            }
            off += 16u;
        }
    }

    free(astc_ctx);
    return TC_SUCCESS;
}

size_t tc_dds_bc7_size(uint32_t width, uint32_t height) {
    size_t bc7 = tc_bc7_compressed_size(width, height);
    if (!bc7 || bc7 > SIZE_MAX - 148u) return 0;
    return 148u + bc7;
}

size_t tc_dds_bc5_size(uint32_t width, uint32_t height) {
    size_t bc5 = tc_bc5_compressed_size(width, height);
    if (!bc5 || bc5 > SIZE_MAX - 148u) return 0;
    return 148u + bc5;
}

size_t tc_dds_bc6h_size(uint32_t width, uint32_t height) {
    size_t bc6h = tc_bc6h_compressed_size(width, height);
    if (!bc6h || bc6h > SIZE_MAX - 148u) return 0;
    return 148u + bc6h;
}

static tc_result tc_dds_write_bc_memory(const uint8_t *bc, uint32_t width,
                                        uint32_t height, uint32_t dxgi_format,
                                        uint8_t *out_dds, size_t out_size) {
    size_t bc_size, need, pitch;
    if (!bc || !out_dds || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    bc_size = tc_bc7_compressed_size(width, height);
    need = 148u + bc_size;
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;
    pitch = (((size_t)width + 3u) >> 2) * 16u;

    memset(out_dds, 0, 148);
    memcpy(out_dds, "DDS ", 4);
    tc_wr_u32(out_dds + 4, 124);
    tc_wr_u32(out_dds + 8, 0x0002100Fu);
    tc_wr_u32(out_dds + 12, height);
    tc_wr_u32(out_dds + 16, width);
    tc_wr_u32(out_dds + 20, (uint32_t)pitch);
    tc_wr_u32(out_dds + 28, 1);
    tc_wr_u32(out_dds + 76, 32);
    tc_wr_u32(out_dds + 80, 0x00000004u);
    memcpy(out_dds + 84, "DX10", 4);
    tc_wr_u32(out_dds + 108, 0x00001000u);
    tc_wr_u32(out_dds + 112, dxgi_format);
    tc_wr_u32(out_dds + 116, 3u);
    tc_wr_u32(out_dds + 124, 1u);
    memcpy(out_dds + 148, bc, bc_size);
    return TC_SUCCESS;
}

tc_result tc_dds_write_bc7_memory(const uint8_t *bc7, uint32_t width,
                                  uint32_t height, const tc_bc7_options *opt,
                                  uint8_t *out_dds, size_t out_size) {
    tc_bc7_options defopt;
    if (!opt) {
        tc_bc7_options_init(&defopt);
        opt = &defopt;
    }
    return tc_dds_write_bc_memory(bc7, width, height, 98u + (opt->srgb ? 1u : 0u),
                                  out_dds, out_size);
}

tc_result tc_dds_write_bc5_memory(const uint8_t *bc5, uint32_t width,
                                  uint32_t height, const tc_bc5_options *opt,
                                  uint8_t *out_dds, size_t out_size) {
    tc_bc5_options defopt;
    if (!opt) {
        tc_bc5_options_init(&defopt);
        opt = &defopt;
    }
    return tc_dds_write_bc_memory(bc5, width, height, opt->snorm ? 84u : 83u,
                                  out_dds, out_size);
}

tc_result tc_dds_write_bc6h_memory(const uint8_t *bc6h, uint32_t width,
                                   uint32_t height,
                                   const tc_bc6h_options *opt,
                                   uint8_t *out_dds, size_t out_size) {
    tc_bc6h_options defopt;
    if (!opt) {
        tc_bc6h_options_init(&defopt);
        opt = &defopt;
    }
    return tc_dds_write_bc_memory(bc6h, width, height, opt->signed_float ? 96u : 95u,
                                  out_dds, out_size);
}

size_t tc_ktx_etc2_size(uint32_t width, uint32_t height,
                        const tc_etc2_options *opt) {
    tc_etc2_options defopt;
    size_t payload;
    if (!opt) {
        tc_etc2_options_init(&defopt);
        opt = &defopt;
    }
    payload = opt->alpha ? tc_etc2_rgba_compressed_size(width, height)
                         : tc_etc2_rgb_compressed_size(width, height);
    if (!payload || payload > SIZE_MAX - 68u) return 0;
    return 68u + payload;
}

static tc_result tc_ktx_write_compressed_memory(const uint8_t *payload_data,
                                                uint32_t width, uint32_t height,
                                                uint32_t internal_format,
                                                uint32_t base_format,
                                                size_t payload,
                                                uint8_t *out_ktx,
                                                size_t out_size) {
    size_t need;
    static const uint8_t ktx_id[12] = {0xab, 0x4b, 0x54, 0x58, 0x20, 0x31,
                                       0x31, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
    if (!payload_data || !out_ktx || !width || !height || !payload)
        return TC_ERROR_INVALID_ARGUMENT;
    if (payload > SIZE_MAX - 68u) return TC_ERROR_INVALID_ARGUMENT;
    need = 68u + payload;
    if (out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    memset(out_ktx, 0, need);
    memcpy(out_ktx, ktx_id, sizeof(ktx_id));
    tc_wr_u32(out_ktx + 12, 0x04030201u);
    tc_wr_u32(out_ktx + 20, 1u);
    tc_wr_u32(out_ktx + 28, internal_format);
    tc_wr_u32(out_ktx + 32, base_format);
    tc_wr_u32(out_ktx + 36, width);
    tc_wr_u32(out_ktx + 40, height);
    tc_wr_u32(out_ktx + 52, 1u);
    tc_wr_u32(out_ktx + 56, 1u);
    tc_wr_u32(out_ktx + 64, (uint32_t)payload);
    memcpy(out_ktx + 68, payload_data, payload);
    return TC_SUCCESS;
}

tc_result tc_ktx_write_etc2_memory(const uint8_t *etc2, uint32_t width,
                                   uint32_t height,
                                   const tc_etc2_options *opt,
                                   uint8_t *out_ktx, size_t out_size) {
    tc_etc2_options defopt;
    size_t payload;
    if (!opt) {
        tc_etc2_options_init(&defopt);
        opt = &defopt;
    }
    payload = opt->alpha ? tc_etc2_rgba_compressed_size(width, height)
                         : tc_etc2_rgb_compressed_size(width, height);
    return tc_ktx_write_compressed_memory(
        etc2, width, height,
        opt->alpha ? (opt->srgb ? 0x9279u : 0x9278u)
                   : (opt->srgb ? 0x9275u : 0x9274u),
        opt->alpha ? 0x1908u : 0x1907u, payload, out_ktx, out_size);
}

tc_result tc_ktx_write_eac_memory(const uint8_t *eac, uint32_t width,
                                  uint32_t height, int rg11,
                                  uint8_t *out_ktx, size_t out_size) {
    size_t payload = rg11 ? tc_eac_rg11_compressed_size(width, height)
                          : tc_eac_r11_compressed_size(width, height);
    return tc_ktx_write_compressed_memory(eac, width, height,
                                          rg11 ? 0x9272u : 0x9270u,
                                          rg11 ? 0x8227u : 0x1903u, payload,
                                          out_ktx, out_size);
}

size_t tc_astc_file_size(uint32_t width, uint32_t height,
                         const tc_astc_options *opt) {
    size_t payload = tc_astc_compressed_size(width, height, opt);
    if (!payload || payload > SIZE_MAX - 16u) return 0;
    return 16u + payload;
}

tc_result tc_astc_write_file_memory(const uint8_t *astc_blocks, uint32_t width,
                                    uint32_t height,
                                    const tc_astc_options *opt,
                                    uint8_t *out_astc_file, size_t out_size) {
    tc_astc_options defopt;
    size_t payload, need;
    if (!opt) {
        tc_astc_options_init(&defopt);
        opt = &defopt;
    }
    if (!astc_blocks || !out_astc_file || !width || !height)
        return TC_ERROR_INVALID_ARGUMENT;
    if (!tc_astc_valid_block(opt->block_x, opt->block_y))
        return TC_ERROR_INVALID_ARGUMENT;
    payload = tc_astc_compressed_size(width, height, opt);
    need = tc_astc_file_size(width, height, opt);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    memset(out_astc_file, 0, 16u);
    out_astc_file[0] = 0x13u;
    out_astc_file[1] = 0xabu;
    out_astc_file[2] = 0xa1u;
    out_astc_file[3] = 0x5cu;
    out_astc_file[4] = (uint8_t)opt->block_x;
    out_astc_file[5] = (uint8_t)opt->block_y;
    out_astc_file[6] = 1u;
    tc_wr_u24(out_astc_file + 7, width);
    tc_wr_u24(out_astc_file + 10, height);
    tc_wr_u24(out_astc_file + 13, 1u);
    memcpy(out_astc_file + 16, astc_blocks, payload);
    return TC_SUCCESS;
}
