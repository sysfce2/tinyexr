/*
 * TinyEXR texcomp - pure-C11 texture compression helpers.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef TINYEXR_TEXCOMP_H_
#define TINYEXR_TEXCOMP_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tc_result {
    TC_SUCCESS = 0,
    TC_ERROR_INVALID_ARGUMENT = -1,
    TC_ERROR_OUT_OF_MEMORY = -2,
    TC_ERROR_IO = -3,
    TC_ERROR_UNSUPPORTED = -4,
    TC_ERROR_CORRUPT = -5
} tc_result;

typedef enum tc_bc7_quality {
    TC_BC7_QUALITY_FAST = 0,
    TC_BC7_QUALITY_QUICKBC7 = 1
} tc_bc7_quality;

typedef struct tc_bc7_options {
    tc_bc7_quality quality;
    int perceptual;
    int srgb;
    int quick;
    int threads;
    uint32_t mode_mask; /* bit N enables BC7 mode N; 0 means all modes */
} tc_bc7_options;

typedef struct tc_bc5_options {
    int snorm;
} tc_bc5_options;

typedef struct tc_bc6h_options {
    int signed_float;
    int reserved;
} tc_bc6h_options;

typedef struct tc_etc2_options {
    int srgb;
    int alpha;
} tc_etc2_options;

typedef struct tc_astc_options {
    uint32_t block_x;
    uint32_t block_y;
    int srgb;
    int quality;
    /* Worker threads for tc_astc_compress_rgba8 (<=1 = serial). Uses C11
     * <threads.h>; builds where the implementation defines
     * __STDC_NO_THREADS__ always encode serially. Output is byte-identical
     * for any thread count. */
    int threads;
} tc_astc_options;

typedef enum tc_backend_mask {
    TC_BACKEND_SCALAR = 0u,
    TC_BACKEND_SSE2 = 1u << 0,
    TC_BACKEND_SSE41 = 1u << 1,
    TC_BACKEND_AVX2 = 1u << 2,
    TC_BACKEND_NEON = 1u << 3,
    TC_BACKEND_SVE = 1u << 4,
    TC_BACKEND_ALL = 0xffffffffu
} tc_backend_mask;

const char *tc_result_string(tc_result r);
const char *tc_backend_name(void);
uint32_t tc_backend_available_mask(void);
void tc_backend_force_mask(uint32_t mask);
void tc_bc7_options_init(tc_bc7_options *opt);
void tc_bc5_options_init(tc_bc5_options *opt);
void tc_bc6h_options_init(tc_bc6h_options *opt);
void tc_etc2_options_init(tc_etc2_options *opt);
void tc_astc_options_init(tc_astc_options *opt);

size_t tc_bc7_compressed_size(uint32_t width, uint32_t height);
size_t tc_bc5_compressed_size(uint32_t width, uint32_t height);
size_t tc_bc6h_compressed_size(uint32_t width, uint32_t height);
size_t tc_etc2_rgb_compressed_size(uint32_t width, uint32_t height);
size_t tc_etc2_rgba_compressed_size(uint32_t width, uint32_t height);
size_t tc_eac_r11_compressed_size(uint32_t width, uint32_t height);
size_t tc_eac_rg11_compressed_size(uint32_t width, uint32_t height);
size_t tc_astc_compressed_size(uint32_t width, uint32_t height,
                               const tc_astc_options *opt);
unsigned int tc_astc_ise_sequence_bitcount(unsigned int value_count,
                                           unsigned int quant_level);
tc_result tc_astc_ise_encode_bits(unsigned int quant_level, unsigned int value_count,
                                  const uint8_t *values, uint8_t *out,
                                  size_t out_size, unsigned int bit_offset);

tc_result tc_bc7_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                const tc_bc7_options *opt, uint8_t *out_bc7,
                                size_t out_size);
tc_result tc_bc5_compress_rg8(const uint8_t *rg, uint32_t width,
                              uint32_t height, size_t stride,
                              const tc_bc5_options *opt, uint8_t *out_bc5,
                              size_t out_size);
tc_result tc_bc5_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                const tc_bc5_options *opt, uint8_t *out_bc5,
                                size_t out_size);
tc_result tc_bc6h_compress_rgb32f(const float *rgb, uint32_t width,
                                  uint32_t height, size_t stride_bytes,
                                  const tc_bc6h_options *opt,
                                  uint8_t *out_bc6h, size_t out_size);
tc_result tc_etc2_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                 uint32_t height, size_t stride,
                                 const tc_etc2_options *opt,
                                 uint8_t *out_etc2, size_t out_size);
tc_result tc_eac_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                int rg11, uint8_t *out_eac, size_t out_size);
tc_result tc_astc_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                 uint32_t height, size_t stride,
                                 const tc_astc_options *opt,
                                 uint8_t *out_astc, size_t out_size);

size_t tc_dds_bc7_size(uint32_t width, uint32_t height);
size_t tc_dds_bc5_size(uint32_t width, uint32_t height);
size_t tc_dds_bc6h_size(uint32_t width, uint32_t height);
size_t tc_ktx_etc2_size(uint32_t width, uint32_t height,
                        const tc_etc2_options *opt);
size_t tc_astc_file_size(uint32_t width, uint32_t height,
                         const tc_astc_options *opt);
tc_result tc_dds_write_bc7_memory(const uint8_t *bc7, uint32_t width,
                                  uint32_t height, const tc_bc7_options *opt,
                                  uint8_t *out_dds, size_t out_size);
tc_result tc_dds_write_bc5_memory(const uint8_t *bc5, uint32_t width,
                                  uint32_t height, const tc_bc5_options *opt,
                                  uint8_t *out_dds, size_t out_size);
tc_result tc_dds_write_bc6h_memory(const uint8_t *bc6h, uint32_t width,
                                   uint32_t height,
                                   const tc_bc6h_options *opt,
                                   uint8_t *out_dds, size_t out_size);
tc_result tc_ktx_write_etc2_memory(const uint8_t *etc2, uint32_t width,
                                   uint32_t height,
                                   const tc_etc2_options *opt,
                                   uint8_t *out_ktx, size_t out_size);
tc_result tc_ktx_write_eac_memory(const uint8_t *eac, uint32_t width,
                                  uint32_t height, int rg11,
                                  uint8_t *out_ktx, size_t out_size);
tc_result tc_astc_write_file_memory(const uint8_t *astc_blocks, uint32_t width,
                                    uint32_t height,
                                    const tc_astc_options *opt,
                                    uint8_t *out_astc_file, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif
