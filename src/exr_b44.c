/*
 * TinyEXR - B44 / B44A codec (lossy 4x4 block, HALF channels only).
 *
 * The 4x4 block unpack and the exp/log perceptual tables follow the
 * well-tested decoder in the legacy single-header tinyexr (which matches
 * OpenEXR). Output is emitted in the canonical scanline-block layout so the
 * shared scatter step can place it.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

#include <math.h>
#include <stdlib.h>

/* ---- perceptual tables (lazy init) ---------------------------------------- */

static uint16_t g_b44_exp_table[65536];
static uint16_t g_b44_log_table[65536];
static int g_b44_tables_ready = 0;

static float b44_h2f(uint16_t h) {
    union { uint32_t i; float f; } u;
    int s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) { u.i = (uint32_t)s << 31; return u.f; }
        {
            float f = (float)m / 1024.0f * (1.0f / 16384.0f);
            return s ? -f : f;
        }
    } else if (e == 31) {
        u.i = ((uint32_t)s << 31) | 0x7f800000u | ((uint32_t)m << 13);
        return u.f;
    }
    u.i = ((uint32_t)s << 31) | ((uint32_t)(e + 112) << 23) | ((uint32_t)m << 13);
    return u.f;
}

static uint16_t b44_f2h(float f) {
    union { uint32_t i; float f; } u;
    int s, e, m;
    u.f = f;
    s = (int)((u.i >> 31) & 1);
    e = (int)((u.i >> 23) & 0xff);
    m = (int)(u.i & 0x7fffff);
    if (e == 0) return (uint16_t)(s << 15);
    if (e == 255) return (uint16_t)((s << 15) | 0x7c00 | (m >> 13));
    if (e < 113) {
        if (e < 103) return (uint16_t)(s << 15);
        m = (m | 0x800000) >> (114 - e);
        return (uint16_t)((s << 15) | (m >> 13));
    }
    if (e > 142) return (uint16_t)((s << 15) | 0x7c00);
    return (uint16_t)((s << 15) | ((e - 112) << 10) | (m >> 13));
}

static void b44_init_tables(void) {
    int i;
    if (g_b44_tables_ready) return;
    for (i = 0; i < 65536; ++i) {
        uint16_t x = (uint16_t)i;
        /* expTable: convertFromLinear (decode for p_linear channels) */
        if ((x & 0x7c00) == 0x7c00)
            g_b44_exp_table[i] = 0;
        else if (x >= 0x558c && x < 0x8000)
            g_b44_exp_table[i] = 0x7bff;
        else
            g_b44_exp_table[i] = b44_f2h((float)exp((double)b44_h2f(x) / 8.0));
        /* logTable: convertToLinear */
        if ((x & 0x7c00) == 0x7c00) {
            g_b44_log_table[i] = 0;
        } else if (x > 0x8000) {
            g_b44_log_table[i] = 0;
        } else {
            float f = b44_h2f(x);
            if (f <= 0.0f)
                g_b44_log_table[i] = 0;
            else
                g_b44_log_table[i] = b44_f2h((float)(8.0 * log((double)f)));
        }
    }
    g_b44_tables_ready = 1;
}

/* ---- block unpack (matches OpenEXR unpack14 / unpack3) --------------------- */

static void b44_unpack14(uint16_t dst[16], const uint8_t s[14]) {
    uint16_t s0 = (uint16_t)(((uint16_t)s[0] << 8) | s[1]);
    unsigned shift = s[2] >> 2;
    unsigned bias = 0x20u << shift;
    uint16_t v[16];
#define ACC(prev, six) \
    (uint16_t)((uint32_t)(prev) + (uint32_t)((six) & 0x3fu) * (1u << shift) - bias)
    v[0] = s0;
    v[4] = ACC(v[0], (((uint32_t)s[2] << 4) | ((uint32_t)s[3] >> 4)));
    v[8] = ACC(v[4], (((uint32_t)s[3] << 2) | ((uint32_t)s[4] >> 6)));
    v[12] = ACC(v[8], (uint32_t)s[4]);
    v[1] = ACC(v[0], (uint32_t)(s[5] >> 2));
    v[5] = ACC(v[4], (((uint32_t)s[5] << 4) | ((uint32_t)s[6] >> 4)));
    v[9] = ACC(v[8], (((uint32_t)s[6] << 2) | ((uint32_t)s[7] >> 6)));
    v[13] = ACC(v[12], (uint32_t)s[7]);
    v[2] = ACC(v[1], (uint32_t)(s[8] >> 2));
    v[6] = ACC(v[5], (((uint32_t)s[8] << 4) | ((uint32_t)s[9] >> 4)));
    v[10] = ACC(v[9], (((uint32_t)s[9] << 2) | ((uint32_t)s[10] >> 6)));
    v[14] = ACC(v[13], (uint32_t)s[10]);
    v[3] = ACC(v[2], (uint32_t)(s[11] >> 2));
    v[7] = ACC(v[6], (((uint32_t)s[11] << 4) | ((uint32_t)s[12] >> 4)));
    v[11] = ACC(v[10], (((uint32_t)s[12] << 2) | ((uint32_t)s[13] >> 6)));
    v[15] = ACC(v[14], (uint32_t)s[13]);
#undef ACC
    {
        int i;
        for (i = 0; i < 16; ++i) {
            if (v[i] & 0x8000)
                dst[i] = (uint16_t)(v[i] & 0x7fff);
            else
                dst[i] = (uint16_t)(~v[i]);
        }
    }
}

static void b44_unpack3(uint16_t dst[16], const uint8_t s[3]) {
    uint16_t t = (uint16_t)(((uint16_t)s[0] << 8) | s[1]);
    uint16_t h = (t & 0x8000) ? (uint16_t)(t & 0x7fff) : (uint16_t)(~t);
    int i;
    for (i = 0; i < 16; ++i) dst[i] = h;
}

/* ---- decode --------------------------------------------------------------- */

exr_result exr_b44_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size,
                              int optimize_flat) {
    const exr_allocator *a = ctx->alloc;
    int xmin = ctx->x, xmax = ctx->x + ctx->width - 1;
    const uint8_t *in = src, *in_end = src + src_size;
    uint16_t **half = NULL;        /* per-channel decoded scratch (HALF) */
    const uint8_t **nonhalf = NULL; /* per-channel raw ptr (UINT/FLOAT) */
    int *cw = NULL, *ch = NULL, *cpw = NULL;
    int *emitted = NULL;
    int c, line, need_tables = 0;
    exr_result rc = EXR_SUCCESS;
    uint8_t *out, *out_end;

    (void)optimize_flat; /* flat blocks are detected per-block by shift value */

    for (c = 0; c < ctx->num_channels; ++c)
        if (ctx->channels[c].p_linear) need_tables = 1;
    if (need_tables) b44_init_tables();

    half = (uint16_t **)exr_calloc(a, (size_t)ctx->num_channels, sizeof(*half));
    nonhalf = (const uint8_t **)exr_calloc(a, (size_t)ctx->num_channels,
                                           sizeof(*nonhalf));
    cw = (int *)exr_calloc(a, (size_t)ctx->num_channels, sizeof(int));
    ch = (int *)exr_calloc(a, (size_t)ctx->num_channels, sizeof(int));
    cpw = (int *)exr_calloc(a, (size_t)ctx->num_channels, sizeof(int));
    emitted = (int *)exr_calloc(a, (size_t)ctx->num_channels, sizeof(int));
    if (!half || !nonhalf || !cw || !ch || !cpw || !emitted) {
        rc = EXR_ERROR_OUT_OF_MEMORY;
        goto done;
    }

    /* Pass A: decode each channel from the stream into scratch. */
    for (c = 0; c < ctx->num_channels; ++c) {
        int xs = ctx->channels[c].x_sampling, ys = ctx->channels[c].y_sampling;
        int width, height, pw, ph, bx, by, nbx, nby;
        if (xs <= 0 || ys <= 0) { rc = EXR_ERROR_CORRUPT; goto done; }
        width = exr_num_samples(xmin, xmax, xs);
        height = exr_num_samples(ctx->y, ctx->y + ctx->num_lines - 1, ys);
        if (width < 0) width = 0;
        if (height < 0) height = 0;
        cw[c] = width;
        ch[c] = height;

        if (ctx->channels[c].pixel_type != EXR_PIXEL_HALF) {
            size_t bytes = (size_t)width * (size_t)height * 4u;
            if ((size_t)(in_end - in) < bytes) { rc = EXR_ERROR_CORRUPT; goto done; }
            nonhalf[c] = in;
            in += bytes;
            continue;
        }

        pw = ((width + 3) / 4) * 4;
        ph = ((height + 3) / 4) * 4;
        cpw[c] = pw;
        nbx = pw / 4;
        nby = ph / 4;
        if (pw > 0 && ph > 0) {
            size_t n;
            if (exr_mul_ovf((size_t)pw, (size_t)ph, &n)) { rc = EXR_ERROR_CORRUPT; goto done; }
            half[c] = (uint16_t *)exr_calloc(a, n, sizeof(uint16_t));
            if (!half[c]) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
        }
        for (by = 0; by < nby; ++by) {
            for (bx = 0; bx < nbx; ++bx) {
                uint16_t block[16];
                int dy;
                if (in + 3 > in_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                if (in[2] >= (13 << 2)) {
                    b44_unpack3(block, in);
                    in += 3;
                } else {
                    if (in + 14 > in_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                    b44_unpack14(block, in);
                    in += 14;
                }
                if (ctx->channels[c].p_linear) {
                    int i;
                    for (i = 0; i < 16; ++i) block[i] = g_b44_exp_table[block[i]];
                }
                for (dy = 0; dy < 4; ++dy) {
                    int dx;
                    for (dx = 0; dx < 4; ++dx)
                        half[c][(size_t)(by * 4 + dy) * pw + (bx * 4 + dx)] =
                            block[dy * 4 + dx];
                }
            }
        }
    }

    /* Pass B: emit canonical block layout (per line, per sampled channel). */
    out = dst;
    out_end = dst + dst_size;
    for (line = 0; line < ctx->num_lines; ++line) {
        int yy = ctx->y + line;
        for (c = 0; c < ctx->num_channels; ++c) {
            int ys = ctx->channels[c].y_sampling;
            int row, x, width = cw[c];
            if ((yy % ys) != 0) continue;
            row = emitted[c]++;
            if (row >= ch[c]) { rc = EXR_ERROR_CORRUPT; goto done; }
            if (ctx->channels[c].pixel_type == EXR_PIXEL_HALF) {
                const uint16_t *r = half[c] + (size_t)row * cpw[c];
                if (out + (size_t)width * 2 > out_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                for (x = 0; x < width; ++x) {
                    out[0] = (uint8_t)(r[x] & 0xff);
                    out[1] = (uint8_t)(r[x] >> 8);
                    out += 2;
                }
            } else {
                size_t bytes = (size_t)width * 4u;
                const uint8_t *r = nonhalf[c] + (size_t)row * bytes;
                if (out + bytes > out_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                memcpy(out, r, bytes);
                out += bytes;
            }
        }
    }
    if ((size_t)(out - dst) != dst_size) rc = EXR_ERROR_CORRUPT;

done:
    if (half) {
        for (c = 0; c < ctx->num_channels; ++c) exr_free(a, half[c]);
        exr_free(a, half);
    }
    exr_free(a, (void *)nonhalf);
    exr_free(a, cw);
    exr_free(a, ch);
    exr_free(a, cpw);
    exr_free(a, emitted);
    return rc;
}
