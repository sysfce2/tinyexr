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

/* ---- LNS (logarithmic) HDR value domain ---------------------------------
 * ASTC HDR interpolates endpoints in a 16-bit "LNS" (log) domain and converts
 * the interpolated value to FP16 with lns_to_sf16. These are scalar ports of
 * astcenc's float_to_lns / lns_to_sf16 (astcenc_vecmathlib.h), the reference
 * decoder we validate against. */
static uint32_t tc_f2u(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    return v.u;
}

static float tc_u2f(uint32_t u) {
    union {
        float f;
        uint32_t u;
    } v;
    v.u = u;
    return v.f;
}

/* astcenc-compatible frexp: mantissa in [0.5,1), exponent via the raw bits. */
static float tc_astc_frexp(float a, int *expo) {
    uint32_t ai = tc_f2u(a);
    *expo = (int)((ai >> 23) & 0xFFu) - 126;
    return tc_u2f((ai & 0x807FFFFFu) | 0x3F000000u);
}

/* float -> 16-bit LNS value (rounded, clamped to [0, 65535]). */
int tc_astc_float_to_lns16(float a) {
    int expo, ri, expv;
    float mant, av, a2, r;
    int underflow = !(a > (1.0f / 67108864.0f)); /* a <= 2^-26 (incl. neg/NaN) */
    int infinity = a >= 65536.0f;
    mant = tc_astc_frexp(a, &expo);
    if (expo < -13) {
        av = a * 33554432.0f; /* 2^25 */
        expv = 0;
    } else {
        av = (mant - 0.5f) * 4096.0f;
        expv = expo + 14;
    }
    if (av < 384.0f)
        a2 = av * (4.0f / 3.0f);
    else if (av <= 1408.0f)
        a2 = av + 128.0f;
    else
        a2 = (av + 512.0f) * (4.0f / 5.0f);
    r = a2 + (float)expv * 2048.0f + 1.0f;
    if (infinity) r = 65535.0f;
    if (underflow) r = 0.0f;
    ri = (int)(r + 0.5f);
    if (ri < 0) ri = 0;
    if (ri > 65535) ri = 65535;
    return ri;
}

/* 16-bit LNS value -> FP16 bits. */
uint16_t tc_astc_lns16_to_sf16(int p) {
    int mc = p & 0x7FF;
    int ec = (int)((unsigned)p >> 11);
    int mt, res;
    if (mc < 512)
        mt = mc * 3;
    else if (mc < 1536)
        mt = mc * 4 - 512;
    else
        mt = mc * 5 - 2048;
    res = (ec << 10) | (mt >> 3);
    if (res > 0x7BFF) res = 0x7BFF;
    return (uint16_t)res;
}

/* ---- CEM 11 (HDR RGB direct) endpoint codec -----------------------------
 * Endpoints are 16-bit LNS values per channel. pack uses the majcomp==3
 * "direct" sub-mode (R/G at 8-bit, B at 7-bit qlog precision); unpack
 * implements the ASTC hdr_rgb decode for that sub-mode. The full 8-mode /
 * base+offset packing (astcenc quantize_hdr_rgb) is a later refinement. */
void tc_astc_cem11_pack(const int lns0[3], const int lns1[3], uint8_t v[6]) {
    int r0 = (lns0[0] + 128) >> 8, r1 = (lns1[0] + 128) >> 8;
    int g0 = (lns0[1] + 128) >> 8, g1 = (lns1[1] + 128) >> 8;
    int b0 = (lns0[2] + 256) >> 9, b1 = (lns1[2] + 256) >> 9;
    if (r0 > 255) r0 = 255;
    if (r1 > 255) r1 = 255;
    if (g0 > 255) g0 = 255;
    if (g1 > 255) g1 = 255;
    if (b0 > 127) b0 = 127;
    if (b1 > 127) b1 = 127;
    v[0] = (uint8_t)r0;
    v[1] = (uint8_t)r1;
    v[2] = (uint8_t)g0;
    v[3] = (uint8_t)g1;
    v[4] = (uint8_t)(b0 | 0x80); /* bit7 = majcomp low  */
    v[5] = (uint8_t)(b1 | 0x80); /* bit7 = majcomp high -> majcomp==3 */
}

/* Decode CEM 11 endpoints to 16-bit LNS per channel. Currently the majcomp==3
 * direct sub-mode (what pack emits); returns 0 for other sub-modes (TODO). */
int tc_astc_cem11_unpack(const uint8_t v[6], int out0[3], int out1[3]) {
    int majcomp = ((v[4] & 0x80) >> 7) | (((v[5] & 0x80) >> 7) << 1);
    if (majcomp == 3) {
        out0[0] = v[0] << 8;
        out0[1] = v[2] << 8;
        out0[2] = (v[4] & 0x7F) << 9;
        out1[0] = v[1] << 8;
        out1[1] = v[3] << 8;
        out1[2] = (v[5] & 0x7F) << 9;
        return 1;
    }
    return 0;
}

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
            int lns[16][3];
            float src0[3] = {0.0f, 0.0f, 0.0f};
            int allsame = 1;
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const float *src;
                    int idx = (int)(yy * 4u + xx);
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = (const float *)((const uint8_t *)rgb +
                                          (size_t)y * stride_bytes +
                                          (size_t)x * 3u * sizeof(float));
                    lns[idx][0] = tc_astc_float_to_lns16(src[0]);
                    lns[idx][1] = tc_astc_float_to_lns16(src[1]);
                    lns[idx][2] = tc_astc_float_to_lns16(src[2]);
                    if (idx == 0) {
                        src0[0] = src[0];
                        src0[1] = src[1];
                        src0[2] = src[2];
                    } else if (lns[idx][0] != lns[0][0] ||
                               lns[idx][1] != lns[0][1] ||
                               lns[idx][2] != lns[0][2]) {
                        allsame = 0;
                    }
                }
            }
            if (allsame) {
                /* Constant block: an FP16 void-extent is exact and cheaper. */
                tc_astc_hdr_write_void_extent(tc_float_to_half_bits(src0[0]),
                                              tc_float_to_half_bits(src0[1]),
                                              tc_float_to_half_bits(src0[2]),
                                              0x3C00u, out_astc + off);
            } else {
                /* Encode CEM 11 and a constant (block-mean void-extent); keep
                 * whichever reconstructs the block with less LNS-domain error,
                 * so the per-texel path never loses to the constant one. */
                uint8_t cem[16];
                uint64_t cem_sse, ve_sse = 0;
                int mean[3];
                int64_t sum[3] = {0, 0, 0};
                int i, cc;
                cem_sse = tc_encode_astc_hdr_cem11_block(lns, cem);
                for (i = 0; i < 16; ++i)
                    for (cc = 0; cc < 3; ++cc) sum[cc] += lns[i][cc];
                for (cc = 0; cc < 3; ++cc) mean[cc] = (int)(sum[cc] / 16);
                for (i = 0; i < 16; ++i)
                    for (cc = 0; cc < 3; ++cc) {
                        int64_t e = (int64_t)lns[i][cc] - mean[cc];
                        ve_sse += (uint64_t)(e * e);
                    }
                if (ve_sse <= cem_sse) {
                    tc_astc_hdr_write_void_extent(
                        tc_astc_lns16_to_sf16(mean[0]),
                        tc_astc_lns16_to_sf16(mean[1]),
                        tc_astc_lns16_to_sf16(mean[2]), 0x3C00u, out_astc + off);
                } else {
                    memcpy(out_astc + off, cem, 16u);
                }
            }
            off += 16u;
        }
    }
    return TC_SUCCESS;
}
