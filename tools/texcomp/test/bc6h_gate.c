/*
 * TinyEXR texcomp - BC6H conformance + quality gate.
 *
 * Encodes HDR float images with the texcomp BC6H encoder and decodes every
 * emitted block with an independent reference decoder (bc6h_ref_decode.h, a port
 * of bcdec covering all 14 modes). Asserts the decoded image is close to the
 * source -- a true end-to-end conformance check (the block bit layout, mode
 * field, delta transform and interpolation must all be correct) and a quality
 * floor. Covers both the unsigned (uf16) and signed (sf16) paths.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "texcomp.h"
#include "bc6h_ref_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Minimal IEEE half -> float (handles normals, subnormals, inf/NaN, sign). */
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    float f;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    /* type-pun via memcpy to stay strict-aliasing clean */
    {
        float out;
        unsigned char *d = (unsigned char *)&out;
        d[0] = (unsigned char)(bits & 0xFFu);
        d[1] = (unsigned char)((bits >> 8) & 0xFFu);
        d[2] = (unsigned char)((bits >> 16) & 0xFFu);
        d[3] = (unsigned char)((bits >> 24) & 0xFFu);
        f = out;
    }
    return f;
}

/* Decode a full BC6H image (WxH, blocks row-major) to float RGB. */
static void decode_image(const uint8_t *blocks, uint32_t W, uint32_t H,
                         int is_signed, float *out_rgb) {
    uint32_t bx, by, tx, ty;
    size_t bi = 0;
    for (by = 0; by < H; by += 4)
        for (bx = 0; bx < W; bx += 4, bi += 16) {
            uint16_t tex[16][3];
            tc_bc6h_ref_decode_block(blocks + bi, is_signed, tex);
            for (ty = 0; ty < 4; ++ty)
                for (tx = 0; tx < 4; ++tx) {
                    uint32_t x = bx + tx, y = by + ty;
                    const uint16_t *t = tex[ty * 4 + tx];
                    if (x < W && y < H) {
                        float *p = out_rgb + ((size_t)y * W + x) * 3;
                        p[0] = half_to_float(t[0]);
                        p[1] = half_to_float(t[1]);
                        p[2] = half_to_float(t[2]);
                    }
                }
        }
}

static double psnr(const float *a, const float *b, size_t n, double peak) {
    double sse = 0.0;
    size_t i;
    for (i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sse += d * d;
    }
    if (sse <= 0.0) return 99.0;
    return 10.0 * log10(peak * peak / (sse / (double)n));
}

int main(void) {
    enum { W = 64, H = 64 };
    static float src[W * H * 3];
    static float dec[W * H * 4]; /* 3 used; padded */
    uint8_t *blocks;
    tc_bc6h_options opt;
    size_t need;
    uint32_t x, y;
    int fail = 0;

    need = tc_bc6h_compressed_size(W, H);
    blocks = (uint8_t *)malloc(need);

    /* (a) unsigned: a smooth HDR gradient with a diagonal split (exercises both
     * one-region and two-region modes across the image). */
    for (y = 0; y < H; ++y)
        for (x = 0; x < W; ++x) {
            float t = (float)(x + y) / (float)(2 * (W - 1));
            float *p = src + ((size_t)y * W + x) * 3;
            p[0] = 0.1f + t * 30.0f;
            p[1] = 0.2f + (1.0f - t) * 18.0f;
            p[2] = 0.05f + t * t * 12.0f;
        }
    tc_bc6h_options_init(&opt);
    opt.signed_float = 0;
    if (tc_bc6h_compress_rgb32f(src, W, H, (size_t)W * 3u * sizeof(float), &opt,
                                blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: bc6h unsigned encode\n");
        return 1;
    }
    decode_image(blocks, W, H, 0, dec);
    {
        double p = psnr(src, dec, (size_t)W * H * 3, 30.0);
        printf("bc6h unsigned gradient 64x64: %.2f dB (ref decoder)\n", p);
        if (p < 45.0) {
            fprintf(stderr, "FAIL: bc6h unsigned psnr %.2f below floor\n", p);
            fail = 1;
        }
    }

    /* (b) signed: values spanning +/- (exercises the sf16 unquantize path). */
    for (y = 0; y < H; ++y)
        for (x = 0; x < W; ++x) {
            float t = (float)x / (float)(W - 1);
            float *p = src + ((size_t)y * W + x) * 3;
            p[0] = -8.0f + t * 16.0f;
            p[1] = 6.0f - t * 12.0f;
            p[2] = -2.0f + (float)y / (float)(H - 1) * 4.0f;
        }
    opt.signed_float = 1;
    if (tc_bc6h_compress_rgb32f(src, W, H, (size_t)W * 3u * sizeof(float), &opt,
                                blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: bc6h signed encode\n");
        return 1;
    }
    decode_image(blocks, W, H, 1, dec);
    {
        double p = psnr(src, dec, (size_t)W * H * 3, 8.0);
        /* Signed BC6H uses the one-region mode only (no two-region signed
         * path), so it is coarser than the unsigned gradient above. */
        printf("bc6h signed gradient 64x64: %.2f dB (ref decoder)\n", p);
        if (p < 32.0) {
            fprintf(stderr, "FAIL: bc6h signed psnr %.2f below floor\n", p);
            fail = 1;
        }
    }

    free(blocks);
    if (fail) {
        printf("bc6h gate: FAIL\n");
        return 1;
    }
    printf("bc6h gate: OK\n");
    return 0;
}
