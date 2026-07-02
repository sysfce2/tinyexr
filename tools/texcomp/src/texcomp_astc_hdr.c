/*
 * TinyEXR texcomp - ASTC HDR (UASTC HDR 4x4) encoder.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Emits 100%% standard ASTC HDR 4x4 blocks. The current implementation encodes
 * constant-colour (void-extent) FP16 blocks -- the block-average of each 4x4
 * tile. Per-texel HDR endpoint modes (ASTC CEM 7 base+scale, CEM 11 RGB
 * direct, with qlog endpoints) are being ported incrementally on top of this
 * pipeline. Output validates against a conformant ASTC HDR decoder
 * (deps/astcenc) via `make texcomp-astc-hdr-gate`.
 */
#include "texcomp.h"
#include "texcomp_internal.h"

#include <string.h>

void tc_astc_hdr_options_init(tc_astc_hdr_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
    opt->quality = 1;
}

size_t tc_astc_hdr_compressed_size(uint32_t width, uint32_t height) {
    size_t bx, by, blocks;
    if (!width || !height) return 0;
    bx = ((size_t)width + 3u) >> 2;  /* 4x4 blocks */
    by = ((size_t)height + 3u) >> 2;
    blocks = bx * by;
    if (blocks > (size_t)-1 / 16u) return 0;
    return blocks * 16u; /* 16 bytes per ASTC block */
}

/* Write a standard ASTC HDR constant-colour (void-extent) block: FP16 RGBA. */
static void tc_astc_hdr_write_void_extent(uint16_t r, uint16_t g, uint16_t b,
                                          uint16_t a, uint8_t out[16]) {
    /* Void-extent marker + "all coordinates" (no extent) for an FP16 (HDR)
     * constant-colour block. See the ASTC spec / astcenc symbolic_to_physical:
     * bytes 0..7 = FC FF FF FF FF FF FF FF, bytes 8..15 = 4x FP16 (little
     * endian) in R,G,B,A order. (An LDR UNORM16 void-extent uses byte 1 = FD.) */
    out[0] = 0xFCu;
    out[1] = 0xFFu;
    out[2] = 0xFFu;
    out[3] = 0xFFu;
    out[4] = 0xFFu;
    out[5] = 0xFFu;
    out[6] = 0xFFu;
    out[7] = 0xFFu;
    out[8] = (uint8_t)(r & 0xFFu);
    out[9] = (uint8_t)(r >> 8);
    out[10] = (uint8_t)(g & 0xFFu);
    out[11] = (uint8_t)(g >> 8);
    out[12] = (uint8_t)(b & 0xFFu);
    out[13] = (uint8_t)(b >> 8);
    out[14] = (uint8_t)(a & 0xFFu);
    out[15] = (uint8_t)(a >> 8);
}

tc_result tc_astc_hdr_compress_rgbf(const float *rgb, uint32_t width,
                                    uint32_t height, size_t stride_bytes,
                                    const tc_astc_hdr_options *opt,
                                    uint8_t *out_astc, size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    size_t need, off = 0;
    (void)opt;

    if (!rgb || !out_astc || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride_bytes < (size_t)width * 3u * sizeof(float))
        return TC_ERROR_INVALID_ARGUMENT;
    need = tc_astc_hdr_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            double sr = 0.0, sg = 0.0, sb = 0.0;
            uint16_t hr, hg, hb;
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const float *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = (const float *)((const uint8_t *)rgb +
                                          (size_t)y * stride_bytes +
                                          (size_t)x * 3u * sizeof(float));
                    sr += src[0];
                    sg += src[1];
                    sb += src[2];
                }
            }
            hr = tc_float_to_half_bits((float)(sr / 16.0));
            hg = tc_float_to_half_bits((float)(sg / 16.0));
            hb = tc_float_to_half_bits((float)(sb / 16.0));
            tc_astc_hdr_write_void_extent(hr, hg, hb, 0x3C00u /* 1.0 */,
                                          out_astc + off);
            off += 16u;
        }
    }
    return TC_SUCCESS;
}
