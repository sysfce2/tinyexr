/*
 * TinyEXR - pixel-format conversion helpers + exr_part <-> interleaved-float
 * bridges (util module).
 *
 * Canonical working format is float32. half<->float reuses the existing F16C
 * dispatch; integer<->float goes through the util SIMD kernels (scalar
 * reference here is the source of truth). Everything is libm-free.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

/* Round-to-nearest-even of a value already clamped to [0, ~2^22] via the
 * add/sub-magic trick (uses the default IEEE rounding mode, no libm). Matches
 * the hardware _mm_cvtps_epi32 / vcvtnq used by the SIMD kernels. */
#define EXR_ROUND_MAGIC 12582912.0f /* 2^23 + 2^22 */

/* ============================================================================
 * Scalar reference kernels (vtbl defaults)
 * ========================================================================== */

void exr_u8_to_f32_scalar(float *dst, const uint8_t *src, size_t n, float scale) {
    size_t i;
    for (i = 0; i < n; ++i) dst[i] = (float)src[i] * scale;
}

void exr_u16_to_f32_scalar(float *dst, const uint16_t *src, size_t n,
                           float scale) {
    size_t i;
    for (i = 0; i < n; ++i) dst[i] = (float)src[i] * scale;
}

void exr_f32_to_u8_scalar(uint8_t *dst, const float *src, size_t n, float scale) {
    size_t i;
    for (i = 0; i < n; ++i) {
        float v = src[i] * scale;
        if (!(v > 0.0f)) v = 0.0f; /* clamps negatives and NaN to 0 */
        if (v > 255.0f) v = 255.0f;
        v = (v + EXR_ROUND_MAGIC) - EXR_ROUND_MAGIC; /* round half to even */
        dst[i] = (uint8_t)v;
    }
}

void exr_f32_to_u16_scalar(uint16_t *dst, const float *src, size_t n,
                           float scale) {
    size_t i;
    for (i = 0; i < n; ++i) {
        float v = src[i] * scale;
        if (!(v > 0.0f)) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        v = (v + EXR_ROUND_MAGIC) - EXR_ROUND_MAGIC;
        dst[i] = (uint16_t)v;
    }
}

/* ============================================================================
 * Public integer<->float helpers (runtime SIMD-dispatched)
 * ========================================================================== */

void exr_u8_to_float(const uint8_t *src, float *dst, size_t n, int normalized) {
    exr_simd_init();
    exr_simd.u8_to_f32(dst, src, n, normalized ? (1.0f / 255.0f) : 1.0f);
}
void exr_u16_to_float(const uint16_t *src, float *dst, size_t n, int normalized) {
    exr_simd_init();
    exr_simd.u16_to_f32(dst, src, n, normalized ? (1.0f / 65535.0f) : 1.0f);
}
void exr_float_to_u8(const float *src, uint8_t *dst, size_t n, int normalized) {
    exr_simd_init();
    exr_simd.f32_to_u8(dst, src, n, normalized ? 255.0f : 1.0f);
}
void exr_float_to_u16(const float *src, uint16_t *dst, size_t n, int normalized) {
    exr_simd_init();
    exr_simd.f32_to_u16(dst, src, n, normalized ? 65535.0f : 1.0f);
}

/* ============================================================================
 * uint32 helpers (kept scalar: normalized needs double precision and the
 * float->uint32 convert does not vectorize cleanly past 2^31)
 * ========================================================================== */

static void u32_to_float(const uint32_t *src, float *dst, size_t n, int norm) {
    size_t i;
    if (norm) {
        const double inv = 1.0 / 4294967295.0;
        for (i = 0; i < n; ++i) dst[i] = (float)((double)src[i] * inv);
    } else {
        for (i = 0; i < n; ++i) dst[i] = (float)src[i]; /* lossy > 2^24 */
    }
}

static void float_to_u32(const float *src, uint32_t *dst, size_t n, int norm) {
    size_t i;
    const double scale = norm ? 4294967295.0 : 1.0;
    for (i = 0; i < n; ++i) {
        double v = (double)src[i] * scale;
        if (!(v > 0.0)) v = 0.0; /* negatives + NaN -> 0 */
        if (v > 4294967295.0) v = 4294967295.0;
        /* round half to even in double, then truncate */
        {
            double f = v - (double)(uint32_t)v; /* fractional part [0,1) */
            uint32_t u = (uint32_t)v;
            if (f > 0.5 || (f == 0.5 && (u & 1u))) u += 1u;
            dst[i] = u;
        }
    }
}

/* ============================================================================
 * Generic any-to-any conversion
 * ========================================================================== */

exr_result exr_convert_pixels(void *dst, exr_pixel_type dst_type,
                              const void *src, exr_pixel_type src_type,
                              size_t count, exr_convert_mode mode) {
    int norm = (mode == EXR_CONVERT_NORMALIZED);
    if (!dst || !src) return EXR_ERROR_INVALID_ARGUMENT;
    if (count == 0) return EXR_SUCCESS;

    if (src_type == dst_type) {
        memcpy(dst, src, count * exr_pixel_size(src_type));
        return EXR_SUCCESS;
    }

    /* Fast direct paths that avoid a float scratch buffer. */
    if (src_type == EXR_PIXEL_HALF && dst_type == EXR_PIXEL_FLOAT) {
        exr_half_to_float((const uint16_t *)src, (float *)dst, count);
        return EXR_SUCCESS;
    }
    if (src_type == EXR_PIXEL_FLOAT && dst_type == EXR_PIXEL_HALF) {
        exr_float_to_half((const float *)src, (uint16_t *)dst, count);
        return EXR_SUCCESS;
    }
    if (src_type == EXR_PIXEL_UINT && dst_type == EXR_PIXEL_FLOAT) {
        u32_to_float((const uint32_t *)src, (float *)dst, count, norm);
        return EXR_SUCCESS;
    }
    if (src_type == EXR_PIXEL_FLOAT && dst_type == EXR_PIXEL_UINT) {
        float_to_u32((const float *)src, (uint32_t *)dst, count, norm);
        return EXR_SUCCESS;
    }

    /* HALF <-> UINT: widen to float in modest chunks, then narrow. */
    {
        float tmp[256];
        size_t off = 0;
        while (off < count) {
            size_t k = count - off;
            if (k > 256) k = 256;
            if (src_type == EXR_PIXEL_HALF)
                exr_half_to_float((const uint16_t *)src + off, tmp, k);
            else /* src UINT */
                u32_to_float((const uint32_t *)src + off, tmp, k, norm);

            if (dst_type == EXR_PIXEL_HALF)
                exr_float_to_half(tmp, (uint16_t *)dst + off, k);
            else /* dst UINT */
                float_to_u32(tmp, (uint32_t *)dst + off, k, norm);
            off += k;
        }
    }
    return EXR_SUCCESS;
}

/* ============================================================================
 * exr_part <-> interleaved float bridges
 * ========================================================================== */

/* Widen one source sample of arbitrary EXR type at linear index `idx`. */
static float load_sample(const void *plane, exr_pixel_type t, size_t idx) {
    switch (t) {
        case EXR_PIXEL_HALF: {
            float f;
            exr_half_to_float((const uint16_t *)plane + idx, &f, 1);
            return f;
        }
        case EXR_PIXEL_FLOAT:
            return ((const float *)plane)[idx];
        default: /* UINT */
            return (float)((const uint32_t *)plane)[idx];
    }
}

exr_result exr_part_to_rgba_float(const exr_allocator *a, const exr_part *part,
                                  float **out, int *out_width, int *out_height,
                                  int *out_channels) {
    int w, h, nch, c, x, y;
    size_t total;
    float *buf;
    if (!a) a = exr_default_allocator();
    if (!part || !out || part->is_deep || !part->images)
        return EXR_ERROR_INVALID_ARGUMENT;
    w = part->width;
    h = part->height;
    nch = part->header.num_channels;
    if (w <= 0 || h <= 0 || nch <= 0) return EXR_ERROR_INVALID_ARGUMENT;
    if (exr_mul_ovf((size_t)w, (size_t)h, &total) ||
        exr_mul_ovf(total, (size_t)nch, &total) ||
        exr_mul_ovf(total, sizeof(float), &total))
        return EXR_ERROR_CORRUPT;

    buf = (float *)exr_malloc(a, total);
    if (!buf) return EXR_ERROR_OUT_OF_MEMORY;

    for (c = 0; c < nch; ++c) {
        const exr_channel *ch = &part->header.channels[c];
        int xs = ch->x_sampling < 1 ? 1 : ch->x_sampling;
        int ys = ch->y_sampling < 1 ? 1 : ch->y_sampling;
        int cw = exr_num_samples(0, w - 1, xs);
        const void *plane = part->images[c];
        for (y = 0; y < h; ++y) {
            int cy = (ys == 1) ? y : (y / ys);
            for (x = 0; x < w; ++x) {
                int cx = (xs == 1) ? x : (x / xs);
                size_t sidx = (size_t)cy * (size_t)cw + (size_t)cx;
                buf[((size_t)y * w + x) * nch + c] =
                    load_sample(plane, ch->pixel_type, sidx);
            }
        }
    }
    *out = buf;
    if (out_width) *out_width = w;
    if (out_height) *out_height = h;
    if (out_channels) *out_channels = nch;
    return EXR_SUCCESS;
}

exr_result exr_rgba_float_to_part(const exr_allocator *a, const float *rgba,
                                  int width, int height, int channels,
                                  exr_pixel_type dst_type, exr_part *out) {
    static const char *names1[] = {"Y"};
    static const char *names2[] = {"R", "G"};
    static const char *names3[] = {"R", "G", "B"};
    static const char *names4[] = {"R", "G", "B", "A"};
    const char **names;
    size_t npx, plane_bytes;
    int c, i;
    exr_result rc = EXR_SUCCESS;
    if (!a) a = exr_default_allocator();
    if (!rgba || !out || width <= 0 || height <= 0 || channels < 1 ||
        channels > 4)
        return EXR_ERROR_INVALID_ARGUMENT;
    names = channels == 1 ? names1
                          : channels == 2 ? names2
                                          : channels == 3 ? names3 : names4;

    memset(out, 0, sizeof(*out));
    out->width = width;
    out->height = height;
    out->header.part_type = EXR_PART_SCANLINE;
    out->header.compression = EXR_COMPRESSION_ZIP;
    out->header.num_channels = channels;
    out->header.data_window.min_x = 0;
    out->header.data_window.min_y = 0;
    out->header.data_window.max_x = width - 1;
    out->header.data_window.max_y = height - 1;
    out->header.display_window = out->header.data_window;
    out->header.pixel_aspect_ratio = 1.0f;
    out->header.screen_window_width = 1.0f;

    out->header.channels =
        (exr_channel *)exr_calloc(a, (size_t)channels, sizeof(exr_channel));
    out->images = (void **)exr_calloc(a, (size_t)channels, sizeof(void *));
    if (!out->header.channels || !out->images) {
        rc = EXR_ERROR_OUT_OF_MEMORY;
        goto fail;
    }
    npx = (size_t)width * (size_t)height;
    plane_bytes = npx * exr_pixel_size(dst_type);
    for (c = 0; c < channels; ++c) {
        exr_channel *ch = &out->header.channels[c];
        size_t k = strlen(names[c]);
        memcpy(ch->name, names[c], k + 1);
        ch->pixel_type = dst_type;
        ch->x_sampling = 1;
        ch->y_sampling = 1;
        out->images[c] = exr_malloc(a, plane_bytes ? plane_bytes : 1);
        if (!out->images[c]) {
            rc = EXR_ERROR_OUT_OF_MEMORY;
            goto fail;
        }
    }
    /* Deinterleave + narrow each channel. */
    {
        float *scratch = (float *)exr_malloc(a, npx * sizeof(float));
        if (!scratch) {
            rc = EXR_ERROR_OUT_OF_MEMORY;
            goto fail;
        }
        for (c = 0; c < channels; ++c) {
            for (i = 0; (size_t)i < npx; ++i)
                scratch[i] = rgba[(size_t)i * channels + c];
            rc = exr_convert_pixels(out->images[c], dst_type, scratch,
                                    EXR_PIXEL_FLOAT, npx, EXR_CONVERT_RAW);
            if (!EXR_OK(rc)) break;
        }
        exr_free(a, scratch);
        if (!EXR_OK(rc)) goto fail;
    }
    return EXR_SUCCESS;

fail:
    exr_part_free(a, out);
    return rc;
}
