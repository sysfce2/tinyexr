/*
 * TinyEXR - PXR24 codec (lossy 24-bit float): zlib inflate + byte-plane delta.
 *
 * Each channel-scanline is stored as byte planes with a running delta across
 * the row; FLOAT keeps the top 3 bytes (low mantissa byte dropped), HALF 2
 * bytes, UINT 4 bytes.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

static size_t pxr_bpc(exr_pixel_type t) {
    switch (t) {
    case EXR_PIXEL_UINT: return 4;
    case EXR_PIXEL_HALF: return 2;
    case EXR_PIXEL_FLOAT: return 3;
    }
    return 2;
}

exr_result exr_pxr24_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                                size_t src_size, uint8_t *dst, size_t dst_size) {
    const exr_allocator *a = ctx->alloc;
    int xmin = ctx->x, xmax = ctx->x + ctx->width - 1;
    int line, c;
    size_t inter = 0;
    uint8_t *buf;
    const uint8_t *in;
    uint8_t *out, *out_end;
    exr_result rc = EXR_SUCCESS;

    /* Pass 1: intermediate (planar, delta-coded) size. */
    for (line = 0; line < ctx->num_lines; ++line) {
        int yy = ctx->y + line;
        for (c = 0; c < ctx->num_channels; ++c) {
            int ys = ctx->channels[c].y_sampling, xs = ctx->channels[c].x_sampling;
            int w;
            size_t add;
            if (ys <= 0 || xs <= 0) return EXR_ERROR_CORRUPT;
            if ((yy % ys) != 0) continue;
            w = exr_num_samples(xmin, xmax, xs);
            if (w < 0) w = 0;
            if (exr_mul_ovf((size_t)w, pxr_bpc(ctx->channels[c].pixel_type), &add))
                return EXR_ERROR_CORRUPT;
            if (exr_add_ovf(inter, add, &inter)) return EXR_ERROR_CORRUPT;
        }
    }

    buf = (uint8_t *)exr_malloc(a, inter ? inter : 1);
    if (!buf) return EXR_ERROR_OUT_OF_MEMORY;

    if (src_size == inter) {
        memcpy(buf, src, inter);
    } else {
        size_t got = 0;
        rc = exr_inflate_zlib(src, src_size, buf, inter, &got);
        if (EXR_OK(rc) && got != inter) rc = EXR_ERROR_CORRUPT;
        if (!EXR_OK(rc)) {
            exr_free(a, buf);
            return rc;
        }
    }

    /* Pass 2: reconstruct pixels into the canonical block layout. */
    in = buf;
    out = dst;
    out_end = dst + dst_size;
    for (line = 0; line < ctx->num_lines; ++line) {
        int yy = ctx->y + line;
        for (c = 0; c < ctx->num_channels; ++c) {
            int ys = ctx->channels[c].y_sampling, xs = ctx->channels[c].x_sampling;
            int w, x;
            uint32_t pixel = 0;
            if ((yy % ys) != 0) continue;
            w = exr_num_samples(xmin, xmax, xs);
            if (w < 0) w = 0;
            switch (ctx->channels[c].pixel_type) {
            case EXR_PIXEL_UINT: {
                const uint8_t *p0 = in, *p1 = in + w, *p2 = in + 2 * w,
                              *p3 = in + 3 * w;
                in += (size_t)w * 4;
                for (x = 0; x < w; ++x) {
                    pixel += ((uint32_t)p0[x] << 24) | ((uint32_t)p1[x] << 16) |
                             ((uint32_t)p2[x] << 8) | (uint32_t)p3[x];
                    if (out + 4 > out_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                    memcpy(out, &pixel, 4);
                    out += 4;
                }
                break;
            }
            case EXR_PIXEL_HALF: {
                const uint8_t *p0 = in, *p1 = in + w;
                in += (size_t)w * 2;
                for (x = 0; x < w; ++x) {
                    uint16_t h;
                    pixel += ((uint32_t)p0[x] << 8) | (uint32_t)p1[x];
                    h = (uint16_t)pixel;
                    if (out + 2 > out_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                    memcpy(out, &h, 2);
                    out += 2;
                }
                break;
            }
            case EXR_PIXEL_FLOAT: {
                const uint8_t *p0 = in, *p1 = in + w, *p2 = in + 2 * w;
                in += (size_t)w * 3;
                for (x = 0; x < w; ++x) {
                    pixel += ((uint32_t)p0[x] << 24) | ((uint32_t)p1[x] << 16) |
                             ((uint32_t)p2[x] << 8);
                    if (out + 4 > out_end) { rc = EXR_ERROR_CORRUPT; goto done; }
                    memcpy(out, &pixel, 4);
                    out += 4;
                }
                break;
            }
            }
        }
    }

    if ((size_t)(out - dst) != dst_size) rc = EXR_ERROR_CORRUPT;

done:
    exr_free(a, buf);
    return rc;
}
