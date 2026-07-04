/*
 * TinyEXR texcomp CLI.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "texcomp.h"

#ifdef TEXCOMP_HAVE_ASTCENC
/* Vendored Arm astcenc backend (deps/astcenc, Apache-2.0); its public API
 * has C linkage, so the pure-C CLI can drive it directly. */
#include "astcenc.h"

#include <stdio.h>
#include <stdlib.h>

#if !defined(__STDC_NO_THREADS__) && !defined(TC_NO_THREADS)
#include <threads.h>
#define TC_CLI_HAVE_THREADS 1
#endif

#ifdef TC_CLI_HAVE_THREADS
struct tc_cli_astcenc_job {
    struct astcenc_context *ctx;
    const struct astcenc_image *img;
    const struct astcenc_swizzle *swz;
    uint8_t *out;
    size_t out_size;
    unsigned int index;
    int status;
};

static int tc_cli_astcenc_worker(void *arg) {
    struct tc_cli_astcenc_job *job = (struct tc_cli_astcenc_job *)arg;
    job->status = astcenc_compress_image(job->ctx, (struct astcenc_image *)job->img,
                                         job->swz, job->out, job->out_size,
                                         job->index) == ASTCENC_SUCCESS
                      ? 0
                      : 1;
    return 0;
}
#endif

static tc_result tc_cli_astcenc_compress(const uint8_t *rgba, uint32_t w,
                                         uint32_t h,
                                         const tc_astc_options *opt,
                                         uint8_t *out, size_t out_size) {
    static const float presets[3] = {ASTCENC_PRE_FASTEST, ASTCENC_PRE_MEDIUM,
                                     ASTCENC_PRE_THOROUGH};
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = (void *)rgba;
    int q = opt->quality < 0 ? 0 : (opt->quality > 2 ? 2 : opt->quality);
    tc_result tr = TC_SUCCESS;
    if (astcenc_config_init(opt->srgb ? ASTCENC_PRF_LDR_SRGB : ASTCENC_PRF_LDR,
                            opt->block_x, opt->block_y, 1u, presets[q], 0,
                            &cfg) != ASTCENC_SUCCESS)
        return TC_ERROR_UNSUPPORTED;
    /* Error-weighted ASTC: forward per-channel weights to astcenc. */
    if (opt->channel_weights[0] || opt->channel_weights[1] ||
        opt->channel_weights[2] || opt->channel_weights[3]) {
        cfg.cw_r_weight = opt->channel_weights[0];
        cfg.cw_g_weight = opt->channel_weights[1];
        cfg.cw_b_weight = opt->channel_weights[2];
        cfg.cw_a_weight = opt->channel_weights[3];
    }
    {
        unsigned int nthreads = opt->threads > 0 ? (unsigned int)opt->threads : 1u;
#ifndef TC_CLI_HAVE_THREADS
        nthreads = 1u;
#endif
        if (nthreads > 64u) nthreads = 64u;
        if (astcenc_context_alloc(&cfg, nthreads, &ctx, NULL) !=
            ASTCENC_SUCCESS)
            return TC_ERROR_OUT_OF_MEMORY;
        img.dim_x = w;
        img.dim_y = h;
        img.dim_z = 1u;
        img.data_type = ASTCENC_TYPE_U8;
        img.data = &slice;
#ifdef TC_CLI_HAVE_THREADS
        if (nthreads > 1u) {
            struct tc_cli_astcenc_job jobs[64];
            thrd_t tids[64];
            unsigned int t, spawned = 0;
            for (t = 0; t < nthreads; ++t) {
                jobs[t].ctx = ctx;
                jobs[t].img = &img;
                jobs[t].swz = &swz;
                jobs[t].out = out;
                jobs[t].out_size = out_size;
                jobs[t].index = t;
                jobs[t].status = 0;
            }
            for (t = 1; t < nthreads; ++t) {
                if (thrd_create(&tids[t], tc_cli_astcenc_worker, &jobs[t]) !=
                    thrd_success) {
                    /* astcenc requires all workers; fail hard rather than
                     * deadlock its internal barrier. */
                    fprintf(stderr, "texcomp: thread spawn failed\n");
                    exit(1);
                }
                ++spawned;
            }
            tc_cli_astcenc_worker(&jobs[0]);
            for (t = 1; t <= spawned; ++t) {
                int rr;
                thrd_join(tids[t], &rr);
            }
            for (t = 0; t < nthreads; ++t)
                if (jobs[t].status) tr = TC_ERROR_UNSUPPORTED;
        } else
#endif
        if (astcenc_compress_image(ctx, &img, &swz, out, out_size, 0u) !=
            ASTCENC_SUCCESS)
            tr = TC_ERROR_UNSUPPORTED;
        astcenc_context_free(ctx);
        return tr;
    }
}
#endif
#include "exr.h"
#include "tinyexr_zstd.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../../../examples/common/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

static int ends_with(const char *s, const char *suffix) {
    size_t ns, nf;
    if (!s || !suffix) return 0;
    ns = strlen(s);
    nf = strlen(suffix);
    return ns >= nf && strcmp(s + ns - nf, suffix) == 0;
}

static int channel_index(const exr_part *part, const char *name) {
    int32_t i;
    for (i = 0; i < part->header.num_channels; ++i) {
        if (strcmp(part->header.channels[i].name, name) == 0) return (int)i;
    }
    return -1;
}

static float read_exr_sample(const exr_part *part, int ch, size_t idx) {
    exr_pixel_type t = part->header.channels[ch].pixel_type;
    if (t == EXR_PIXEL_UINT) return (float)((const uint32_t *)part->images[ch])[idx] / 255.0f;
    if (t == EXR_PIXEL_HALF) {
        float f;
        exr_half_to_float(&((const uint16_t *)part->images[ch])[idx], &f, 1);
        return f;
    }
    return ((const float *)part->images[ch])[idx];
}

static uint8_t to_u8(float f) {
    int v;
    if (f < 0.0f) f = 0.0f;
    if (f > 1.0f) f = 1.0f;
    v = (int)(f * 255.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (uint8_t)v;
}

static tc_result load_exr_rgba(const char *path, int part_index, uint8_t **rgba,
                               uint32_t *w, uint32_t *h) {
    exr_image img;
    exr_part *part;
    int r, g, b, a;
    size_t i, n;
    exr_result er;
    memset(&img, 0, sizeof(img));
    er = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(er)) return TC_ERROR_CORRUPT;
    if (part_index < 0) part_index = 0;
    if (part_index >= img.num_parts) {
        exr_image_free(&img);
        return TC_ERROR_INVALID_ARGUMENT;
    }
    part = &img.parts[part_index];
    if (part->is_deep || !part->images || part->width <= 0 || part->height <= 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    r = channel_index(part, "R");
    g = channel_index(part, "G");
    b = channel_index(part, "B");
    a = channel_index(part, "A");
    if (r < 0 || g < 0 || b < 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    n = (size_t)part->width * (size_t)part->height;
    *rgba = (uint8_t *)malloc(n * 4u);
    if (!*rgba) {
        exr_image_free(&img);
        return TC_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < n; ++i) {
        (*rgba)[i * 4u + 0u] = to_u8(read_exr_sample(part, r, i));
        (*rgba)[i * 4u + 1u] = to_u8(read_exr_sample(part, g, i));
        (*rgba)[i * 4u + 2u] = to_u8(read_exr_sample(part, b, i));
        (*rgba)[i * 4u + 3u] = a >= 0 ? to_u8(read_exr_sample(part, a, i)) : 255u;
    }
    *w = (uint32_t)part->width;
    *h = (uint32_t)part->height;
    exr_image_free(&img);
    return TC_SUCCESS;
}

static tc_result load_png_rgba(const char *path, uint8_t **rgba, uint32_t *w,
                               uint32_t *h) {
    int x, y, n;
    unsigned char *p = stbi_load(path, &x, &y, &n, 4);
    (void)n;
    if (!p) return TC_ERROR_CORRUPT;
    if (x <= 0 || y <= 0) {
        stbi_image_free(p);
        return TC_ERROR_CORRUPT;
    }
    *rgba = p;
    *w = (uint32_t)x;
    *h = (uint32_t)y;
    return TC_SUCCESS;
}

static tc_result load_exr_rgbf(const char *path, int part_index, float **rgb,
                               uint32_t *w, uint32_t *h) {
    exr_image img;
    exr_part *part;
    int r, g, b;
    size_t i, n;
    exr_result er;
    memset(&img, 0, sizeof(img));
    er = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(er)) return TC_ERROR_CORRUPT;
    if (part_index < 0) part_index = 0;
    if (part_index >= img.num_parts) {
        exr_image_free(&img);
        return TC_ERROR_INVALID_ARGUMENT;
    }
    part = &img.parts[part_index];
    if (part->is_deep || !part->images || part->width <= 0 || part->height <= 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    r = channel_index(part, "R");
    g = channel_index(part, "G");
    b = channel_index(part, "B");
    if (r < 0 || g < 0 || b < 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    n = (size_t)part->width * (size_t)part->height;
    *rgb = (float *)malloc(n * 3u * sizeof(float));
    if (!*rgb) {
        exr_image_free(&img);
        return TC_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < n; ++i) {
        (*rgb)[i * 3u + 0u] = read_exr_sample(part, r, i);
        (*rgb)[i * 3u + 1u] = read_exr_sample(part, g, i);
        (*rgb)[i * 3u + 2u] = read_exr_sample(part, b, i);
    }
    *w = (uint32_t)part->width;
    *h = (uint32_t)part->height;
    exr_image_free(&img);
    return TC_SUCCESS;
}

/* Load an EXR as float RGBA (for HDR-alpha / CEM 15). A defaults to 1.0 when the
 * part has no alpha channel. */
static tc_result load_exr_rgbaf(const char *path, int part_index, float **rgba,
                                uint32_t *w, uint32_t *h) {
    exr_image img;
    exr_part *part;
    int r, g, b, a;
    size_t i, n;
    exr_result er;
    memset(&img, 0, sizeof(img));
    er = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(er)) return TC_ERROR_CORRUPT;
    if (part_index < 0) part_index = 0;
    if (part_index >= img.num_parts) {
        exr_image_free(&img);
        return TC_ERROR_INVALID_ARGUMENT;
    }
    part = &img.parts[part_index];
    if (part->is_deep || !part->images || part->width <= 0 || part->height <= 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    r = channel_index(part, "R");
    g = channel_index(part, "G");
    b = channel_index(part, "B");
    a = channel_index(part, "A");
    if (r < 0 || g < 0 || b < 0) {
        exr_image_free(&img);
        return TC_ERROR_UNSUPPORTED;
    }
    n = (size_t)part->width * (size_t)part->height;
    *rgba = (float *)malloc(n * 4u * sizeof(float));
    if (!*rgba) {
        exr_image_free(&img);
        return TC_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < n; ++i) {
        (*rgba)[i * 4u + 0u] = read_exr_sample(part, r, i);
        (*rgba)[i * 4u + 1u] = read_exr_sample(part, g, i);
        (*rgba)[i * 4u + 2u] = read_exr_sample(part, b, i);
        (*rgba)[i * 4u + 3u] = (a >= 0) ? read_exr_sample(part, a, i) : 1.0f;
    }
    *w = (uint32_t)part->width;
    *h = (uint32_t)part->height;
    exr_image_free(&img);
    return TC_SUCCESS;
}

static tc_result rgba_to_rgbf(const uint8_t *rgba, uint32_t w, uint32_t h,
                              float **rgb) {
    size_t i, n = (size_t)w * (size_t)h;
    *rgb = (float *)malloc(n * 3u * sizeof(float));
    if (!*rgb) return TC_ERROR_OUT_OF_MEMORY;
    for (i = 0; i < n; ++i) {
        (*rgb)[i * 3u + 0u] = (float)rgba[i * 4u + 0u] / 255.0f;
        (*rgb)[i * 3u + 1u] = (float)rgba[i * 4u + 1u] / 255.0f;
        (*rgb)[i * 3u + 2u] = (float)rgba[i * 4u + 2u] / 255.0f;
    }
    return TC_SUCCESS;
}

static tc_result write_file(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return TC_ERROR_IO;
    if (fwrite(data, 1, size, f) != size) {
        fclose(f);
        return TC_ERROR_IO;
    }
    if (fclose(f) != 0) return TC_ERROR_IO;
    return TC_SUCCESS;
}

/* --- xbc7: supercompressed BC7 (windowed RDO + zstd) ----------------------
 * Container: "XBC7" magic, then a 24-byte header, then a zstd frame holding the
 * raw RDO'd BC7 block stream. Transcoding is just a zstd decode back to a
 * standard BC7 stream, which any BC7 device/decoder reads directly. This is a
 * texcomp-native format, not Basis's XBC7 bitstream. */
#define XBC7_HDR 24u
static void xbc7_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static uint32_t xbc7_get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* Encode RGBA -> RDO'd BC7 -> zstd -> .xbc7 file. */
static tc_result xbc7_encode(const uint8_t *rgba, uint32_t w, uint32_t h,
                             const tc_bc7_options *opt, const char *out_path,
                             const char *raw_path) {
    size_t bc7_size = tc_bc7_compressed_size(w, h), zbound, zc;
    uint8_t *bc7, *xf;
    tc_result tr;
    if (!bc7_size) return TC_ERROR_INVALID_ARGUMENT;
    bc7 = (uint8_t *)malloc(bc7_size);
    if (!bc7) return TC_ERROR_OUT_OF_MEMORY;
    tr = tc_bc7_compress_rgba8(rgba, w, h, (size_t)w * 4u, opt, bc7, bc7_size);
    if (tr != TC_SUCCESS) {
        free(bc7);
        return tr;
    }
    zbound = tinyexr_zstd_compress_bound(bc7_size);
    xf = (uint8_t *)malloc(XBC7_HDR + zbound);
    if (!xf) {
        free(bc7);
        return TC_ERROR_OUT_OF_MEMORY;
    }
    zc = tinyexr_zstd_compress(xf + XBC7_HDR, zbound, bc7, bc7_size, 19);
    if (tinyexr_zstd_is_error(zc)) {
        free(bc7);
        free(xf);
        return TC_ERROR_CORRUPT;
    }
    memcpy(xf, "XBC7", 4u);
    xf[4] = 1u;      /* version */
    xf[5] = 1u;      /* flags: zstd */
    xf[6] = 4u;      /* block_x */
    xf[7] = 4u;      /* block_y */
    xbc7_put32(xf + 8, w);
    xbc7_put32(xf + 12, h);
    xbc7_put32(xf + 16, (uint32_t)bc7_size);
    xbc7_put32(xf + 20, (uint32_t)zc);
    tr = write_file(out_path, xf, XBC7_HDR + zc);
    if (tr == TC_SUCCESS && raw_path)
        tr = write_file(raw_path, bc7, bc7_size);
    if (tr == TC_SUCCESS)
        printf("xbc7: %ux%u  rdo=%d  BC7=%zu -> xbc7=%zu (%.1f%%, %.3f bpp)\n", w,
               h, opt->rdo, bc7_size, (size_t)(XBC7_HDR + zc),
               100.0 * (double)(XBC7_HDR + zc) / (double)bc7_size,
               (double)(XBC7_HDR + zc) * 8.0 / ((double)w * h));
    free(bc7);
    free(xf);
    return tr;
}

/* Transcode a .xbc7 file back to a standard BC7 DDS (or raw .bc7). */
static tc_result xbc7_transcode(const char *in_path, const char *out_path,
                                const char *raw_path) {
    FILE *f = fopen(in_path, "rb");
    uint8_t hdr[XBC7_HDR];
    uint8_t *comp = NULL, *bc7 = NULL, *dds = NULL;
    uint32_t w, h, raw_size, comp_size;
    size_t got, dds_size;
    tc_result tr = TC_ERROR_CORRUPT;
    tc_bc7_options bopt;
    if (!f) return TC_ERROR_IO;
    if (fread(hdr, 1u, XBC7_HDR, f) != XBC7_HDR || memcmp(hdr, "XBC7", 4u) != 0) {
        fclose(f);
        return TC_ERROR_CORRUPT;
    }
    w = xbc7_get32(hdr + 8);
    h = xbc7_get32(hdr + 12);
    raw_size = xbc7_get32(hdr + 16);
    comp_size = xbc7_get32(hdr + 20);
    comp = (uint8_t *)malloc(comp_size);
    bc7 = (uint8_t *)malloc(raw_size);
    if (!comp || !bc7) {
        fclose(f);
        free(comp);
        free(bc7);
        return TC_ERROR_OUT_OF_MEMORY;
    }
    got = fread(comp, 1u, comp_size, f);
    fclose(f);
    if (got != comp_size) goto done;
    if (tinyexr_zstd_decompress(bc7, raw_size, comp, comp_size) != raw_size)
        goto done;
    tc_bc7_options_init(&bopt);
    dds_size = tc_dds_bc7_size(w, h);
    dds = (uint8_t *)malloc(dds_size);
    if (!dds) {
        tr = TC_ERROR_OUT_OF_MEMORY;
        goto done;
    }
    tr = tc_dds_write_bc7_memory(bc7, w, h, &bopt, dds, dds_size);
    if (tr == TC_SUCCESS) tr = write_file(out_path, dds, dds_size);
    if (tr == TC_SUCCESS && raw_path) tr = write_file(raw_path, bc7, raw_size);
    if (tr == TC_SUCCESS)
        printf("xbc7 transcode: %ux%u -> BC7 DDS %zu bytes\n", w, h, dds_size);
done:
    free(comp);
    free(bc7);
    free(dds);
    return tr;
}

/* --- uni universal intermediate: KTX2 (Zstd-supercompressed) container ---- */

static void uni_put64(uint8_t *p, uint64_t v) {
    xbc7_put32(p, (uint32_t)v);
    xbc7_put32(p + 4, (uint32_t)(v >> 32));
}
static uint64_t uni_get64(const uint8_t *p) {
    return (uint64_t)xbc7_get32(p) | ((uint64_t)xbc7_get32(p + 4) << 32);
}
static uint8_t *read_whole_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    long sz;
    uint8_t *buf;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    buf = (uint8_t *)malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

#define UNI_MAX_LEVELS 16

static void uni_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static uint16_t uni_get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static const uint32_t *g_uni_cnt; /* for the frequency sort below */
static int uni_cnt_cmp(const void *a, const void *b) {
    uint32_t ca = g_uni_cnt[*(const uint32_t *)a], cb = g_uni_cnt[*(const uint32_t *)b];
    return ca < cb ? 1 : (ca > cb ? -1 : 0);
}
static uint32_t uni_sse8(const uint8_t *a, const uint8_t *b) {
    uint32_t s = 0;
    int k;
    for (k = 0; k < 8; ++k) { int d = (int)a[k] - (int)b[k]; s += (uint32_t)(d * d); }
    return s;
}
/* Selectors are 16 4-bit weights packed in 8 bytes; compare per-texel. */
static uint32_t uni_sse_sel(const uint8_t *a, const uint8_t *b) {
    uint32_t s = 0;
    int k;
    for (k = 0; k < 16; ++k) {
        int wa = (a[k >> 1] >> ((k & 1) * 4)) & 0xf;
        int wb = (b[k >> 1] >> ((k & 1) * 4)) & 0xf;
        int d = wa - wb;
        s += (uint32_t)(d * d);
    }
    return s;
}

#define UNI_LLOYD_ITERS 8

/* Nearest codebook entry to `pp` under the field metric, plus an optional
 * rate penalty lambda*rate[c] (bits) for rate-distortion assignment. */
static uint32_t uni_assign(const uint8_t *pp, const uint8_t *cb, uint32_t cap,
                           int is_sel, const float *rate, float lambda) {
    uint32_t best = 0, c;
    double bestc = 1e300;
    for (c = 0; c < cap; ++c) {
        double d = is_sel ? uni_sse_sel(pp, cb + (size_t)c * 8u)
                          : uni_sse8(pp, cb + (size_t)c * 8u);
        double cost = rate ? d + (double)lambda * rate[c] : d;
        if (cost < bestc) { bestc = cost; best = c; if (d == 0.0 && !rate) break; }
    }
    return best;
}

/* Build an RD-optimized codebook of the 8-byte field at `off` of each uni block,
 * plus per-block indices. cap == 0 or distinct <= cap is lossless. Otherwise:
 * seed with the cap most-frequent patterns, refine with Lloyd (k-means) so the
 * entries minimize distortion, then a final rate-distortion assignment
 * (distortion + lambda*bits) that biases toward popular entries for a smaller
 * post-Zstd index stream. Returns the code count, or 0xffffffff on OOM. */
static uint32_t uni_build_codebook(const uint8_t *uni, size_t nblocks, size_t off,
                                   uint32_t cap, float lambda, uint8_t *cb,
                                   uint32_t *idx) {
    size_t hcap = 16, i;
    uint64_t *hk, *upat;
    int32_t *hv;
    uint32_t *ucnt, *bidx, nu = 0, ncodes;
    int is_sel = (off == 8u);
    while (hcap < nblocks * 2u) hcap <<= 1;
    hk = (uint64_t *)malloc(hcap * 8u);
    hv = (int32_t *)malloc(hcap * 4u);
    upat = (uint64_t *)malloc(nblocks * 8u);
    ucnt = (uint32_t *)malloc(nblocks * 4u);
    bidx = (uint32_t *)malloc(nblocks * 4u);
    if (!hk || !hv || !upat || !ucnt || !bidx) { free(hk); free(hv); free(upat); free(ucnt); free(bidx); return 0xffffffffu; }
    for (i = 0; i < hcap; ++i) hv[i] = -1;
    for (i = 0; i < nblocks; ++i) {
        uint64_t k;
        size_t h;
        memcpy(&k, uni + i * 16u + off, 8u);
        h = (size_t)((k * 0x9E3779B97F4A7C15ULL) & (uint64_t)(hcap - 1u));
        while (hv[h] != -1 && hk[h] != k) h = (h + 1u) & (hcap - 1u);
        if (hv[h] == -1) { hk[h] = k; hv[h] = (int32_t)nu; upat[nu] = k; ucnt[nu] = 1u; ++nu; }
        else ucnt[hv[h]]++;
        bidx[i] = (uint32_t)hv[h];
    }
    if (cap == 0u || nu <= cap) {
        for (i = 0; i < nu; ++i) memcpy(cb + i * 8u, &upat[i], 8u);
        for (i = 0; i < nblocks; ++i) idx[i] = bidx[i];
        ncodes = nu;
    } else {
        uint32_t *order = (uint32_t *)malloc(nu * 4u), c;
        int vlen = is_sel ? 16 : 8, iter, k;
        uint32_t *csum = (uint32_t *)malloc((size_t)cap * vlen * 4u);
        uint32_t *ccnt = (uint32_t *)malloc((size_t)cap * 4u);
        float *rate = (float *)malloc((size_t)cap * sizeof(float));
        if (!order || !csum || !ccnt || !rate) { free(order); free(csum); free(ccnt); free(rate); free(hk); free(hv); free(upat); free(ucnt); free(bidx); return 0xffffffffu; }
        for (c = 0; c < nu; ++c) order[c] = c;
        g_uni_cnt = ucnt;
        qsort(order, nu, sizeof(uint32_t), uni_cnt_cmp);
        for (c = 0; c < cap; ++c) memcpy(cb + (size_t)c * 8u, &upat[order[c]], 8u);
        ncodes = cap;
        /* Lloyd refinement: assign to nearest, recompute centroids as the mean
         * of assigned blocks (per texel for selectors, per byte for endpoints). */
        for (iter = 0; iter < UNI_LLOYD_ITERS; ++iter) {
            memset(csum, 0, (size_t)cap * vlen * 4u);
            memset(ccnt, 0, (size_t)cap * 4u);
            for (i = 0; i < nblocks; ++i) {
                const uint8_t *pp = uni + i * 16u + off;
                uint32_t b = uni_assign(pp, cb, cap, is_sel, NULL, 0.0f);
                idx[i] = b;
                ccnt[b]++;
                if (is_sel) { for (k = 0; k < 16; ++k) csum[b * 16 + k] += (pp[k >> 1] >> ((k & 1) * 4)) & 0xf; }
                else { for (k = 0; k < 8; ++k) csum[b * 8 + k] += pp[k]; }
            }
            for (c = 0; c < cap; ++c) {
                if (!ccnt[c]) continue;
                if (is_sel) {
                    uint8_t nw[8] = {0};
                    for (k = 0; k < 16; ++k) { uint32_t w = (csum[c * 16 + k] + ccnt[c] / 2) / ccnt[c]; if (w > 15u) w = 15u; nw[k >> 1] |= (uint8_t)(w << ((k & 1) * 4)); }
                    memcpy(cb + (size_t)c * 8u, nw, 8u);
                } else {
                    for (k = 0; k < 8; ++k) { uint32_t v = (csum[c * 8 + k] + ccnt[c] / 2) / ccnt[c]; cb[(size_t)c * 8u + k] = (uint8_t)(v > 255u ? 255u : v); }
                }
            }
        }
        /* rate-distortion final assignment: bits(c) ~= -log2(freq(c)). */
        for (c = 0; c < cap; ++c) {
            double p = ((double)ccnt[c] + 1.0) / ((double)nblocks + (double)cap);
            rate[c] = (float)(-log(p) / log(2.0));
        }
        for (i = 0; i < nblocks; ++i)
            idx[i] = uni_assign(uni + i * 16u + off, cb, cap, is_sel,
                                lambda > 0.0f ? rate : NULL, lambda);
        free(order); free(csum); free(ccnt); free(rate);
    }
    free(hk); free(hv); free(upat); free(ucnt); free(bidx);
    return ncodes;
}

/* Basis-style codebook pre-transform of a uni level: split into deduplicated
 * endpoint + selector codebooks with per-block indices. This is the size lever
 * behind BasisLZ (shared endpoint/selector codebooks), entropy-coded afterwards
 * by the outer Zstd. NOTE: this is NOT the wire-format BasisLZ scheme (that is
 * the Basis ETC1S codec); the KTX2 supercompressionScheme stays 2 (Zstd). */
static tc_result encode_uni_basis(const uint8_t *uni, size_t uni_size,
                                  uint32_t cap, float lambda, uint8_t **out,
                                  size_t *out_size) {
    size_t nblocks = uni_size / 16u, i;
    uint8_t *epcb = (uint8_t *)malloc(nblocks * 8u), *selcb = (uint8_t *)malloc(nblocks * 8u), *o, *p;
    uint32_t *epidx = (uint32_t *)malloc(nblocks * 4u), *selidx = (uint32_t *)malloc(nblocks * 4u);
    uint32_t nep, nsel;
    int epb, selb;
    size_t sz;
    tc_result r = TC_ERROR_OUT_OF_MEMORY;
    if (!epcb || !selcb || !epidx || !selidx) goto done;
    nep = uni_build_codebook(uni, nblocks, 0u, cap, lambda, epcb, epidx);
    nsel = uni_build_codebook(uni, nblocks, 8u, cap, lambda, selcb, selidx);
    if (nep == 0xffffffffu || nsel == 0xffffffffu) goto done;
    epb = nep <= 256u ? 1 : 2;
    selb = nsel <= 256u ? 1 : 2;
    sz = 18u + (size_t)nep * 8u + (size_t)nsel * 8u + nblocks * (size_t)epb + nblocks * (size_t)selb;
    o = (uint8_t *)malloc(sz);
    if (!o) goto done;
    p = o;
    memcpy(p, "UBAS", 4); p += 4;
    xbc7_put32(p, (uint32_t)nblocks); p += 4;
    xbc7_put32(p, nep); p += 4;
    xbc7_put32(p, nsel); p += 4;
    *p++ = (uint8_t)epb; *p++ = (uint8_t)selb;
    memcpy(p, epcb, (size_t)nep * 8u); p += (size_t)nep * 8u;
    memcpy(p, selcb, (size_t)nsel * 8u); p += (size_t)nsel * 8u;
    for (i = 0; i < nblocks; ++i) { if (epb == 1) *p++ = (uint8_t)epidx[i]; else { uni_put16(p, (uint16_t)epidx[i]); p += 2; } }
    for (i = 0; i < nblocks; ++i) { if (selb == 1) *p++ = (uint8_t)selidx[i]; else { uni_put16(p, (uint16_t)selidx[i]); p += 2; } }
    *out = o; *out_size = sz;
    r = TC_SUCCESS;
done:
    free(epcb); free(selcb); free(epidx); free(selidx);
    return r;
}

/* Reverse encode_uni_basis: reconstruct the raw uni block stream. */
static tc_result decode_uni_basis(const uint8_t *buf, size_t size,
                                  uint8_t **out_uni, size_t *out_size) {
    size_t nblocks, i;
    uint32_t nep, nsel;
    int epb, selb;
    const uint8_t *p = buf, *epcb, *selcb, *epidx, *selidx;
    uint8_t *uni;
    if (size < 18u || memcmp(buf, "UBAS", 4) != 0) return TC_ERROR_CORRUPT;
    p += 4; nblocks = xbc7_get32(p); p += 4; nep = xbc7_get32(p); p += 4; nsel = xbc7_get32(p); p += 4;
    epb = *p++; selb = *p++;
    epcb = p; p += (size_t)nep * 8u;
    selcb = p; p += (size_t)nsel * 8u;
    epidx = p; p += nblocks * (size_t)epb;
    selidx = p; p += nblocks * (size_t)selb;
    if ((size_t)(p - buf) > size) return TC_ERROR_CORRUPT;
    uni = (uint8_t *)malloc(nblocks * 16u);
    if (!uni) return TC_ERROR_OUT_OF_MEMORY;
    for (i = 0; i < nblocks; ++i) {
        uint32_t ei = epb == 1 ? epidx[i] : uni_get16(epidx + i * 2u);
        uint32_t si = selb == 1 ? selidx[i] : uni_get16(selidx + i * 2u);
        if (ei >= nep || si >= nsel) { free(uni); return TC_ERROR_CORRUPT; }
        memcpy(uni + i * 16u, epcb + (size_t)ei * 8u, 8u);
        memcpy(uni + i * 16u + 8u, selcb + (size_t)si * 8u, 8u);
    }
    *out_uni = uni; *out_size = nblocks * 16u;
    return TC_SUCCESS;
}

/* Box-filter (2x2, from previous level) mip chain of an RGBA8 image. Fills
 * levels[]/lw[]/lh[] (each level malloc'd) and returns the level count. */
static int build_mips(const uint8_t *base, uint32_t w, uint32_t h,
                      uint8_t *levels[], uint32_t lw[], uint32_t lh[], int cap) {
    int n = 1, l;
    uint32_t x, y, c, m = w > h ? w : h;
    while (m > 1u) { m >>= 1; ++n; }
    if (n > cap) n = cap;
    lw[0] = w; lh[0] = h;
    levels[0] = (uint8_t *)malloc((size_t)w * h * 4u);
    if (!levels[0]) return 0;
    memcpy(levels[0], base, (size_t)w * h * 4u);
    for (l = 1; l < n; ++l) {
        uint32_t pw = lw[l - 1], ph = lh[l - 1];
        uint32_t cw = pw > 1u ? pw >> 1 : 1u, ch = ph > 1u ? ph >> 1 : 1u;
        const uint8_t *p = levels[l - 1];
        lw[l] = cw; lh[l] = ch;
        levels[l] = (uint8_t *)malloc((size_t)cw * ch * 4u);
        if (!levels[l]) { while (--l >= 0) { free(levels[l]); } return 0; }
        for (y = 0; y < ch; ++y)
            for (x = 0; x < cw; ++x) {
                uint32_t x0 = 2u * x, x1 = x0 + 1u < pw ? x0 + 1u : pw - 1u;
                uint32_t y0 = 2u * y, y1 = y0 + 1u < ph ? y0 + 1u : ph - 1u;
                uint8_t *o = levels[l] + ((size_t)y * cw + x) * 4u;
                for (c = 0; c < 4u; ++c)
                    o[c] = (uint8_t)((p[((size_t)y0 * pw + x0) * 4u + c] +
                                      p[((size_t)y0 * pw + x1) * 4u + c] +
                                      p[((size_t)y1 * pw + x0) * 4u + c] +
                                      p[((size_t)y1 * pw + x1) * 4u + c] + 2u) / 4u);
            }
    }
    return n;
}

/* Write uni levels as a KTX2 (vkFormat=UNDEFINED, supercompressionScheme=2/Zstd).
 * Each mip level is Zstd-compressed independently; data is packed smallest-first
 * (KTX2 convention). Our loader identifies the file by vkFormat==0 && scheme==2. */
static tc_result write_uni_ktx2(const char *path, uint8_t *const uni[],
                                const size_t uni_sizes[], const uint32_t lw[],
                                const uint32_t lh[], int nlevels, int basis,
                                float rdo) {
    static const uint8_t id[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                   0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    uint8_t *comp[UNI_MAX_LEVELS] = {0}, *bbuf[UNI_MAX_LEVELS] = {0}, *ktx = NULL;
    size_t clen[UNI_MAX_LEVELS], ulen[UNI_MAX_LEVELS];
    size_t dfd_off, dfd_len = 44u, data_base, cursor, total;
    uint64_t loff[UNI_MAX_LEVELS];
    int l;
    tc_result r = TC_SUCCESS;
    (void)lh;
    for (l = 0; l < nlevels; ++l) {
        const uint8_t *src = uni[l];
        size_t src_size = uni_sizes[l], bound;
        if (basis) {
            /* `basis` carries the codebook cap (0 = plain zstd path, not here). */
            if ((r = encode_uni_basis(uni[l], uni_sizes[l], (uint32_t)basis, rdo, &bbuf[l], &src_size)) != TC_SUCCESS) goto done;
            src = bbuf[l];
        }
        ulen[l] = src_size; /* KTX2 uncompressedByteLength = post-zstd-decode size */
        bound = tinyexr_zstd_compress_bound(src_size);
        comp[l] = (uint8_t *)malloc(bound);
        if (!comp[l]) { r = TC_ERROR_OUT_OF_MEMORY; goto done; }
        clen[l] = tinyexr_zstd_compress(comp[l], bound, src, src_size, 19);
        if (tinyexr_zstd_is_error(clen[l])) { r = TC_ERROR_UNSUPPORTED; goto done; }
    }
    dfd_off = 80u + (size_t)nlevels * 24u;
    data_base = dfd_off + dfd_len;
    cursor = data_base; /* supercompressed: tightly packed, smallest-first */
    for (l = nlevels - 1; l >= 0; --l) { loff[l] = cursor; cursor += clen[l]; }
    total = cursor;
    ktx = (uint8_t *)calloc(1, total);
    if (!ktx) { r = TC_ERROR_OUT_OF_MEMORY; goto done; }
    memcpy(ktx, id, 12);
    xbc7_put32(ktx + 12, 0u);              /* vkFormat = UNDEFINED */
    xbc7_put32(ktx + 16, 1u);              /* typeSize */
    xbc7_put32(ktx + 20, lw[0]);
    xbc7_put32(ktx + 24, lh[0]);
    xbc7_put32(ktx + 36, 1u);              /* faceCount */
    xbc7_put32(ktx + 40, (uint32_t)nlevels);
    xbc7_put32(ktx + 44, 2u);              /* Zstandard */
    xbc7_put32(ktx + 48, (uint32_t)dfd_off);
    xbc7_put32(ktx + 52, dfd_len);
    for (l = 0; l < nlevels; ++l) {
        uint8_t *e = ktx + 80u + (size_t)l * 24u;
        uni_put64(e + 0, loff[l]);
        uni_put64(e + 8, clen[l]);           /* byteLength (compressed) */
        uni_put64(e + 16, ulen[l]);          /* uncompressedByteLength */
    }
    xbc7_put32(ktx + dfd_off, 44u);
    xbc7_put32(ktx + dfd_off + 8, 2u | (40u << 16));
    ktx[dfd_off + 12] = 166u; ktx[dfd_off + 13] = 1u; ktx[dfd_off + 14] = 1u;
    ktx[dfd_off + 16] = 3u; ktx[dfd_off + 17] = 3u; ktx[dfd_off + 20] = 16u;
    ktx[dfd_off + 30] = 127u; xbc7_put32(ktx + dfd_off + 40, 0xffffffffu);
    for (l = 0; l < nlevels; ++l) memcpy(ktx + loff[l], comp[l], clen[l]);
    r = write_file(path, ktx, total);
    if (r == TC_SUCCESS)
        printf("wrote %s (uni KTX2, %d mip levels, %s, %zu bytes)\n", path, nlevels,
               basis ? "Basis-style codebook + Zstd" : "Zstd-supercompressed", total);
done:
    for (l = 0; l < nlevels; ++l) { free(comp[l]); free(bbuf[l]); }
    free(ktx);
    return r;
}

/* Read all uni levels back from a KTX2 (Zstd-decompressed). Returns level count
 * or -1; fills uni[]/uni_sizes[]/lw[]/lh[] (each level malloc'd). */
static int read_uni_ktx2(const char *path, uint8_t *uni[], size_t uni_sizes[],
                         uint32_t lw[], uint32_t lh[], int cap) {
    size_t fsz;
    uint8_t *buf = read_whole_file(path, &fsz);
    uint32_t pw, ph;
    int nlevels, l;
    if (!buf) return -1;
    if (fsz < 80u || xbc7_get32(buf + 12) != 0u || xbc7_get32(buf + 44) != 2u) {
        free(buf); return -1;
    }
    nlevels = (int)xbc7_get32(buf + 40);
    if (nlevels < 1 || nlevels > cap || fsz < 80u + (size_t)nlevels * 24u) { free(buf); return -1; }
    pw = xbc7_get32(buf + 20); ph = xbc7_get32(buf + 24);
    for (l = 0; l < nlevels; ++l) {
        const uint8_t *e = buf + 80u + (size_t)l * 24u;
        uint64_t off = uni_get64(e), clen = uni_get64(e + 8), ulen = uni_get64(e + 16);
        size_t dec;
        uint8_t *dbuf;
        lw[l] = pw >> l ? pw >> l : 1u;
        lh[l] = ph >> l ? ph >> l : 1u;
        if (off + clen > fsz || ulen == 0 || !(dbuf = (uint8_t *)malloc((size_t)ulen))) {
            while (--l >= 0) free(uni[l]);
            free(buf);
            return -1;
        }
        dec = tinyexr_zstd_decompress(dbuf, (size_t)ulen, buf + off, (size_t)clen);
        if (tinyexr_zstd_is_error(dec) || dec != ulen) {
            free(dbuf);
            while (--l >= 0) free(uni[l]);
            free(buf);
            return -1;
        }
        /* codebook layer (Basis-style) is nested inside the Zstd frame. */
        if (ulen >= 4u && memcmp(dbuf, "UBAS", 4) == 0) {
            tc_result dr = decode_uni_basis(dbuf, (size_t)ulen, &uni[l], &uni_sizes[l]);
            free(dbuf);
            if (dr != TC_SUCCESS) { while (--l >= 0) free(uni[l]); free(buf); return -1; }
        } else {
            uni[l] = dbuf;
            uni_sizes[l] = (size_t)ulen;
        }
    }
    free(buf);
    return nlevels;
}

/* Transcode one uni level; returns malloc'd target blocks via *out (+ *osz). */
static tc_result transcode_uni_level(const uint8_t *ubuf, uint32_t uw, uint32_t uh,
                                     const char *format, uint8_t **out, size_t *osz) {
    uint8_t *ob = NULL;
    size_t sz = 0;
    tc_result tr;
    if (!strcmp(format, "bc7")) { sz = tc_bc7_compressed_size(uw, uh); ob = malloc(sz); tr = tc_uni_transcode_bc7(ubuf, uw, uh, ob, sz); }
    else if (!strcmp(format, "bc1") || !strcmp(format, "dxt1")) { sz = tc_bc1_compressed_size(uw, uh); ob = malloc(sz); tr = tc_uni_transcode_bc1(ubuf, uw, uh, ob, sz); }
    else if (!strcmp(format, "astc")) { tc_astc_options ao; tc_astc_options_init(&ao); ao.block_x = 4; ao.block_y = 4; ao.uastc = 1; sz = tc_astc_compressed_size(uw, uh, &ao); ob = malloc(sz); tr = tc_uni_transcode_astc(ubuf, uw, uh, ob, sz); }
    else if (!strcmp(format, "etc2") || !strcmp(format, "etc2_rgba")) { sz = tc_etc2_rgba_compressed_size(uw, uh); ob = malloc(sz); tr = tc_uni_transcode_etc2(ubuf, uw, uh, 1, ob, sz); }
    else { fprintf(stderr, "texcomp: --format for uni transcode must be bc7|bc1|astc|etc2\n"); return TC_ERROR_INVALID_ARGUMENT; }
    if (tr != TC_SUCCESS) { free(ob); return tr; }
    *out = ob; *osz = sz;
    return TC_SUCCESS;
}

static void usage(void) {
    fprintf(stderr,
            "usage: texcomp -i in.{png,exr} -o out [--format bc1|bc3|bc7|bc5|bc6h|etc2|etc2_rgb|eac_r11|eac_rg11|astc|astc_hdr|uastc_ldr|xbc7|uni] "
            "[--raw out.bin] [--raw-bc7 out.bc7] [--part N] [--srgb] "
            "[--signed] [--astc-block WxH] [--quality fast|medium|normal] [--encoder tc|arm] [--threads N] "
            "[--quick on|off] [--mode-mask HEX] [--rdo N] "
            "[--channel-weights R,G,B,A] "
            "[--linear|--perceptual]\n"
            "  --channel-weights: per-channel error weights for BC7 (pure-C) and\n"
            "        ASTC (astcenc/--encoder arm); e.g. 2,2,1,1 favours R,G.\n"
            "  --bc7-pca: also seed BC7 endpoints from the weighted principal\n"
            "        color axis, keep-best (off by default; ~+0.09 dB, ~1.7x slower).\n"
            "  --hdr-alpha: for --format astc_hdr on an EXR, encode the alpha\n"
            "        channel too (ASTC HDR CEM 15); default is RGB-only.\n"
            "  uni: universal transcodable intermediate. `--format uni -o x.uni`\n"
            "        (raw, level 0) or `-o x.ktx2` (KTX2, full mip chain, Zstd-\n"
            "        supercompressed; add --basis[/--basis-cap N] for a BasisLZ-\n"
            "        style bounded endpoint/selector codebook before Zstd -- a lossy\n"
            "        rate knob, smaller N = smaller+lossier). Transcode all levels\n"
            "        with `-i x.{uni,ktx2} -o out.bin --format bc7|bc1|astc|etc2`.\n"
            "  xbc7: supercompressed BC7 (windowed RDO + zstd); --rdo N sets the\n"
            "        max per-channel RMS reuse budget. Transcode back with\n"
            "        `-i in.xbc7 -o out.dds` (standard BC7, any device reads it).\n");
}

static int parse_astc_block(const char *s, uint32_t *bx, uint32_t *by) {
    char *endp = NULL;
    unsigned long x = strtoul(s, &endp, 10);
    unsigned long y;
    if (!endp || (*endp != 'x' && *endp != 'X')) return 0;
    y = strtoul(endp + 1, &endp, 10);
    if (!endp || *endp != '\0' || x > UINT32_MAX || y > UINT32_MAX) return 0;
    *bx = (uint32_t)x;
    *by = (uint32_t)y;
    return 1;
}

int main(int argc, char **argv) {
    const char *in = NULL, *out = NULL, *raw = NULL;
    const char *format = "bc7";
    int part = 0;
    int uni_basis = 0;
    float uni_rdo = 0.0f;
    uint8_t *rgba = NULL, *compressed = NULL, *container = NULL;
    float *rgbf = NULL, *rgbaf = NULL;
    int hdr_alpha = 0;
    uint32_t w = 0, h = 0;
    size_t compressed_size, container_size;
    tc_bc7_options bc7_opt;
    tc_bc1_options bc1_opt;
    tc_bc3_options bc3_opt;
    tc_bc5_options bc5_opt;
    tc_bc6h_options bc6h_opt;
    tc_etc2_options etc2_opt;
    tc_astc_options astc_opt;
    int use_arm_encoder = 0;
    tc_result tr;
    int i;

    tc_bc7_options_init(&bc7_opt);
    tc_bc1_options_init(&bc1_opt);
    tc_bc3_options_init(&bc3_opt);
    tc_bc5_options_init(&bc5_opt);
    tc_bc6h_options_init(&bc6h_opt);
    tc_etc2_options_init(&etc2_opt);
    tc_astc_options_init(&astc_opt);
    (void)use_arm_encoder;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) in = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) format = argv[++i];
        else if (strcmp(argv[i], "--raw") == 0 && i + 1 < argc) raw = argv[++i];
        else if (strcmp(argv[i], "--raw-bc7") == 0 && i + 1 < argc) raw = argv[++i];
        else if (strcmp(argv[i], "--part") == 0 && i + 1 < argc) part = atoi(argv[++i]);
        else if (strcmp(argv[i], "--srgb") == 0) {
            bc7_opt.srgb = 1;
            bc1_opt.srgb = 1;
            bc3_opt.srgb = 1;
            etc2_opt.srgb = 1;
            astc_opt.srgb = 1;
        }
        else if (strcmp(argv[i], "--signed") == 0) bc6h_opt.signed_float = 1;
        else if (strcmp(argv[i], "--encoder") == 0 && i + 1 < argc) {
            const char *e = argv[++i];
            if (strcmp(e, "arm") == 0) {
#ifdef TEXCOMP_HAVE_ASTCENC
                use_arm_encoder = 1;
#else
                fprintf(stderr,
                        "texcomp: built without the Arm astcenc backend "
                        "(use `make texcomp-arm`)\n");
                return 1;
#endif
            } else if (strcmp(e, "tc") != 0) {
                fprintf(stderr, "texcomp: unknown encoder '%s'\n", e);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            astc_opt.threads = atoi(argv[++i]);
            if (astc_opt.threads < 1) astc_opt.threads = 1;
        }
        else if (strcmp(argv[i], "--astc-block") == 0 && i + 1 < argc) {
            if (!parse_astc_block(argv[++i], &astc_opt.block_x, &astc_opt.block_y)) {
                usage();
                return 2;
            }
        }
        else if (strcmp(argv[i], "--quality") == 0 && i + 1 < argc) {
            const char *q = argv[++i];
            if (strcmp(q, "fast") == 0) astc_opt.quality = 0;
            else if (strcmp(q, "medium") == 0) astc_opt.quality = 1;
            else if (strcmp(q, "normal") == 0) astc_opt.quality = 2;
            else {
                usage();
                return 2;
            }
        }
        else if (strcmp(argv[i], "--linear") == 0) bc7_opt.perceptual = 0;
        else if (strcmp(argv[i], "--perceptual") == 0) bc7_opt.perceptual = 1;
        else if (strcmp(argv[i], "--quick") == 0 && i + 1 < argc) bc7_opt.quick = strcmp(argv[++i], "off") != 0;
        else if (strcmp(argv[i], "--mode-mask") == 0 && i + 1 < argc)
            bc7_opt.mode_mask = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--rdo") == 0 && i + 1 < argc)
            bc7_opt.rdo = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bc7-pca") == 0) bc7_opt.pca_endpoints = 1;
        else if (strcmp(argv[i], "--hdr-alpha") == 0) hdr_alpha = 1;
        else if (strcmp(argv[i], "--basis") == 0) uni_basis = 2048; /* codebook cap */
        else if (strcmp(argv[i], "--basis-cap") == 0 && i + 1 < argc) uni_basis = atoi(argv[++i]);
        else if (strcmp(argv[i], "--basis-rdo") == 0 && i + 1 < argc) uni_rdo = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--channel-weights") == 0 && i + 1 < argc) {
            /* R,G,B,A error weights for BC7 (pure-C) and ASTC (astcenc). */
            float wr = 1, wg = 1, wb = 1, wa = 1;
            int k;
            sscanf(argv[++i], "%f,%f,%f,%f", &wr, &wg, &wb, &wa);
            astc_opt.channel_weights[0] = wr; astc_opt.channel_weights[1] = wg;
            astc_opt.channel_weights[2] = wb; astc_opt.channel_weights[3] = wa;
            {
                float wv[4] = {wr, wg, wb, wa};
                for (k = 0; k < 4; ++k) {
                    int iw = (int)(wv[k] + 0.5f);
                    if (iw < 0) iw = 0;
                    if (iw > 255) iw = 255;
                    bc7_opt.channel_weights[k] = (uint8_t)iw;
                }
            }
        }
        else {
            usage();
            return 2;
        }
    }
    if (!in || !out) {
        usage();
        return 2;
    }

    /* Transcode mode: a .xbc7 input is decoded back to a standard BC7 DDS. */
    if (ends_with(in, ".xbc7")) {
        tr = xbc7_transcode(in, out, raw);
        if (tr != TC_SUCCESS)
            fprintf(stderr, "texcomp: xbc7 transcode failed: %s\n",
                    tc_result_string(tr));
        return tr == TC_SUCCESS ? 0 : 1;
    }

    /* Transcode mode: a uni intermediate (.uni raw single level, or .ktx2
     * Zstd-supercompressed with a mip chain) -> the --format target. Every level
     * is transcoded and the block streams are concatenated (largest mip first).
     * BC7/BC1 are cheap repacks; astc/etc2 re-encode. */
    if (ends_with(in, ".uni") || ends_with(in, ".ktx2")) {
        uint8_t *ulev[UNI_MAX_LEVELS] = {0}, *blocks[UNI_MAX_LEVELS] = {0}, *cat = NULL, *p;
        size_t usz[UNI_MAX_LEVELS], bsz[UNI_MAX_LEVELS], total = 0;
        uint32_t lw[UNI_MAX_LEVELS], lh[UNI_MAX_LEVELS];
        int nl = 0, l;
        if (ends_with(in, ".ktx2")) {
            nl = read_uni_ktx2(in, ulev, usz, lw, lh, UNI_MAX_LEVELS);
            if (nl < 1) { fprintf(stderr, "texcomp: %s is not a uni KTX2\n", in); return 1; }
        } else {
            size_t fsz;
            uint8_t *raw = read_whole_file(in, &fsz);
            if (!raw || fsz < 12u || memcmp(raw, "TUNI", 4) != 0) { free(raw); fprintf(stderr, "texcomp: not a .uni file\n"); return 1; }
            lw[0] = xbc7_get32(raw + 4); lh[0] = xbc7_get32(raw + 8); usz[0] = fsz - 12u;
            ulev[0] = (uint8_t *)malloc(usz[0]);
            if (ulev[0]) memcpy(ulev[0], raw + 12u, usz[0]);
            free(raw);
            if (!ulev[0]) return 1;
            nl = 1;
        }
        tr = TC_SUCCESS;
        for (l = 0; l < nl; ++l) {
            tr = transcode_uni_level(ulev[l], lw[l], lh[l], format, &blocks[l], &bsz[l]);
            if (tr != TC_SUCCESS) break;
            total += bsz[l];
        }
        if (tr == TC_SUCCESS) {
            cat = (uint8_t *)malloc(total ? total : 1u);
            if (cat) { p = cat; for (l = 0; l < nl; ++l) { memcpy(p, blocks[l], bsz[l]); p += bsz[l]; } tr = write_file(out, cat, total); }
            else tr = TC_ERROR_OUT_OF_MEMORY;
        }
        if (tr == TC_SUCCESS) printf("transcoded %s -> %s (%s, %d level%s, %zu bytes)\n", in, out, format, nl, nl == 1 ? "" : "s", total);
        else fprintf(stderr, "texcomp: uni transcode failed\n");
        for (l = 0; l < nl; ++l) { free(blocks[l]); free(ulev[l]); }
        free(cat);
        return tr == TC_SUCCESS ? 0 : 1;
    }

    if (strcmp(format, "bc6") == 0) format = "bc6h";
    if (strcmp(format, "dxt1") == 0) format = "bc1";
    if (strcmp(format, "dxt5") == 0) format = "bc3";
    if (strcmp(format, "uastc_hdr") == 0) format = "astc_hdr";
    /* UASTC LDR is a subset of standard ASTC LDR 4x4; our ASTC encoder emits
     * the (superset) standard format, decodable by any ASTC device. This is
     * not the constrained 19-mode transcodable subset (see docs). */
    if (strcmp(format, "uastc_ldr") == 0 || strcmp(format, "uastc") == 0) {
        format = "astc";
        astc_opt.block_x = 4;
        astc_opt.block_y = 4;
        astc_opt.uastc = 1;
    }
    if (strcmp(format, "etc2") == 0 || strcmp(format, "etc2_rgba") == 0) {
        etc2_opt.alpha = 1;
        format = "etc2_rgba";
    } else if (strcmp(format, "etc2_rgb") == 0) {
        etc2_opt.alpha = 0;
    } else if (strcmp(format, "etc2_r11") == 0) {
        format = "eac_r11";
    } else if (strcmp(format, "etc2_rg11") == 0) {
        format = "eac_rg11";
    }
    if (strcmp(format, "astc_hdr") == 0 && hdr_alpha && ends_with(in, ".exr")) {
        tr = load_exr_rgbaf(in, part, &rgbaf, &w, &h);
    } else if ((strcmp(format, "bc6h") == 0 || strcmp(format, "astc_hdr") == 0) &&
               ends_with(in, ".exr")) {
        tr = load_exr_rgbf(in, part, &rgbf, &w, &h);
    } else {
        if (ends_with(in, ".png")) tr = load_png_rgba(in, &rgba, &w, &h);
        else if (ends_with(in, ".exr")) tr = load_exr_rgba(in, part, &rgba, &w, &h);
        else tr = TC_ERROR_UNSUPPORTED;
    }
    if (tr != TC_SUCCESS) {
        fprintf(stderr, "texcomp: load failed: %s\n", tc_result_string(tr));
        return 1;
    }

    /* xbc7: supercompressed BC7 (windowed RDO + zstd). Default to a moderate
     * RDO budget when the user didn't request one. */
    if (strcmp(format, "xbc7") == 0) {
        if (bc7_opt.rdo <= 0) bc7_opt.rdo = 16;
        tr = xbc7_encode(rgba, w, h, &bc7_opt, out, raw);
        if (tr != TC_SUCCESS)
            fprintf(stderr, "texcomp: xbc7 encode failed: %s\n",
                    tc_result_string(tr));
        free(rgba);
        free(rgbf); free(rgbaf);
        return tr == TC_SUCCESS ? 0 : 1;
    }

    /* uni: universal transcodable intermediate. Write a tiny container
     * ("TUNI" + w,h + blocks); transcode at load with tc_uni_transcode_*. */
    if (strcmp(format, "uni") == 0) {
        if (ends_with(out, ".ktx2")) {
            /* Zstd-supercompressed KTX2 with a full mip chain. */
            uint8_t *rlev[UNI_MAX_LEVELS] = {0}, *ulev[UNI_MAX_LEVELS] = {0};
            uint32_t lw[UNI_MAX_LEVELS], lh[UNI_MAX_LEVELS];
            size_t usz[UNI_MAX_LEVELS];
            int nl = build_mips(rgba, w, h, rlev, lw, lh, UNI_MAX_LEVELS), l;
            tr = nl >= 1 ? TC_SUCCESS : TC_ERROR_OUT_OF_MEMORY;
            for (l = 0; l < nl && tr == TC_SUCCESS; ++l) {
                usz[l] = tc_uni_compressed_size(lw[l], lh[l]);
                ulev[l] = (uint8_t *)malloc(usz[l]);
                if (!ulev[l]) { tr = TC_ERROR_OUT_OF_MEMORY; break; }
                tr = tc_uni_compress_rgba8(rlev[l], lw[l], lh[l], (size_t)lw[l] * 4u, ulev[l], usz[l]);
            }
            if (tr == TC_SUCCESS) tr = write_uni_ktx2(out, ulev, usz, lw, lh, nl, uni_basis, uni_rdo);
            for (l = 0; l < nl; ++l) { free(rlev[l]); free(ulev[l]); }
        } else {
            /* raw TUNI container: "TUNI" + w,h + level-0 blocks */
            size_t usz = tc_uni_compressed_size(w, h);
            uint8_t *u = (uint8_t *)malloc(usz), *f;
            tr = u ? tc_uni_compress_rgba8(rgba, w, h, (size_t)w * 4u, u, usz) : TC_ERROR_OUT_OF_MEMORY;
            if (tr == TC_SUCCESS && (f = (uint8_t *)malloc(12u + usz)) != NULL) {
                memcpy(f, "TUNI", 4); xbc7_put32(f + 4, w); xbc7_put32(f + 8, h);
                memcpy(f + 12, u, usz);
                tr = write_file(out, f, 12u + usz);
                if (tr == TC_SUCCESS) printf("wrote %s (universal intermediate, %ux%u, %zu bytes)\n", out, w, h, usz);
                free(f);
            } else if (tr == TC_SUCCESS) tr = TC_ERROR_OUT_OF_MEMORY;
            free(u);
        }
        if (tr != TC_SUCCESS) fprintf(stderr, "texcomp: uni encode failed\n");
        free(rgba); free(rgbf); free(rgbaf);
        return tr == TC_SUCCESS ? 0 : 1;
    }

    if (strcmp(format, "bc7") == 0) {
        compressed_size = tc_bc7_compressed_size(w, h);
        container_size = tc_dds_bc7_size(w, h);
    } else if (strcmp(format, "bc1") == 0) {
        compressed_size = tc_bc1_compressed_size(w, h);
        container_size = tc_dds_bc1_size(w, h);
    } else if (strcmp(format, "bc3") == 0) {
        compressed_size = tc_bc3_compressed_size(w, h);
        container_size = tc_dds_bc3_size(w, h);
    } else if (strcmp(format, "bc5") == 0) {
        compressed_size = tc_bc5_compressed_size(w, h);
        container_size = tc_dds_bc5_size(w, h);
    } else if (strcmp(format, "bc6h") == 0) {
        compressed_size = tc_bc6h_compressed_size(w, h);
        container_size = tc_dds_bc6h_size(w, h);
    } else if (strcmp(format, "astc_hdr") == 0) {
        tc_astc_options hdr4;
        tc_astc_options_init(&hdr4);
        hdr4.block_x = 4;
        hdr4.block_y = 4;
        compressed_size = tc_astc_hdr_compressed_size(w, h);
        container_size = tc_astc_file_size(w, h, &hdr4);
    } else if (strcmp(format, "etc2_rgba") == 0 || strcmp(format, "etc2_rgb") == 0) {
        compressed_size = etc2_opt.alpha ? tc_etc2_rgba_compressed_size(w, h)
                                         : tc_etc2_rgb_compressed_size(w, h);
        container_size = tc_ktx_etc2_size(w, h, &etc2_opt);
    } else if (strcmp(format, "eac_r11") == 0) {
        compressed_size = tc_eac_r11_compressed_size(w, h);
        container_size = 68u + compressed_size;
    } else if (strcmp(format, "eac_rg11") == 0) {
        compressed_size = tc_eac_rg11_compressed_size(w, h);
        container_size = 68u + compressed_size;
    } else if (strcmp(format, "astc") == 0) {
        compressed_size = tc_astc_compressed_size(w, h, &astc_opt);
        container_size = tc_astc_file_size(w, h, &astc_opt);
    } else {
        fprintf(stderr, "texcomp: unsupported format: %s\n", format);
        free(rgba);
        free(rgbf); free(rgbaf);
        return 1;
    }
    if (!compressed_size || !container_size) {
        fprintf(stderr, "texcomp: invalid output size for format: %s\n", format);
        free(rgba);
        free(rgbf); free(rgbaf);
        return 1;
    }
    compressed = (uint8_t *)malloc(compressed_size);
    container = (uint8_t *)malloc(container_size);
    if (!compressed || !container) {
        fprintf(stderr, "texcomp: out of memory\n");
        free(rgba);
        free(rgbf); free(rgbaf);
        free(compressed);
        free(container);
        return 1;
    }

    if (strcmp(format, "bc7") == 0) {
        tr = tc_bc7_compress_rgba8(rgba, w, h, (size_t)w * 4u, &bc7_opt,
                                   compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_dds_write_bc7_memory(compressed, w, h, &bc7_opt, container,
                                         container_size);
    } else if (strcmp(format, "bc1") == 0) {
        tr = tc_bc1_compress_rgba8(rgba, w, h, (size_t)w * 4u, &bc1_opt,
                                   compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_dds_write_bc1_memory(compressed, w, h, &bc1_opt, container,
                                         container_size);
    } else if (strcmp(format, "bc3") == 0) {
        tr = tc_bc3_compress_rgba8(rgba, w, h, (size_t)w * 4u, &bc3_opt,
                                   compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_dds_write_bc3_memory(compressed, w, h, &bc3_opt, container,
                                         container_size);
    } else if (strcmp(format, "bc5") == 0) {
        tr = tc_bc5_compress_rgba8(rgba, w, h, (size_t)w * 4u, &bc5_opt,
                                   compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_dds_write_bc5_memory(compressed, w, h, &bc5_opt, container,
                                         container_size);
    } else if (strcmp(format, "bc6h") == 0) {
        if (!rgbf) tr = rgba_to_rgbf(rgba, w, h, &rgbf);
        if (tr == TC_SUCCESS)
            tr = tc_bc6h_compress_rgb32f(rgbf, w, h, (size_t)w * 3u * sizeof(float),
                                         &bc6h_opt, compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_dds_write_bc6h_memory(compressed, w, h, &bc6h_opt, container,
                                          container_size);
    } else if (strcmp(format, "astc_hdr") == 0) {
        tc_astc_options hdr4;
        tc_astc_hdr_options hdr_opt;
        tc_astc_hdr_options_init(&hdr_opt);
        tc_astc_options_init(&hdr4);
        hdr4.block_x = 4;
        hdr4.block_y = 4;
        if (rgbaf) {
            tr = tc_astc_hdr_compress_rgbaf(rgbaf, w, h,
                                            (size_t)w * 4u * sizeof(float),
                                            &hdr_opt, compressed, compressed_size);
        } else {
            if (!rgbf) tr = rgba_to_rgbf(rgba, w, h, &rgbf);
            if (tr == TC_SUCCESS)
                tr = tc_astc_hdr_compress_rgbf(rgbf, w, h,
                                               (size_t)w * 3u * sizeof(float),
                                               &hdr_opt, compressed,
                                               compressed_size);
        }
        if (tr == TC_SUCCESS)
            tr = tc_astc_write_file_memory(compressed, w, h, &hdr4, container,
                                           container_size);
    } else if (strcmp(format, "etc2_rgba") == 0 || strcmp(format, "etc2_rgb") == 0) {
        tr = tc_etc2_compress_rgba8(rgba, w, h, (size_t)w * 4u, &etc2_opt,
                                    compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_ktx_write_etc2_memory(compressed, w, h, &etc2_opt, container,
                                          container_size);
    } else if (strcmp(format, "eac_r11") == 0 || strcmp(format, "eac_rg11") == 0) {
        int rg11 = strcmp(format, "eac_rg11") == 0;
        tr = tc_eac_compress_rgba8(rgba, w, h, (size_t)w * 4u, rg11,
                                   compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_ktx_write_eac_memory(compressed, w, h, rg11, container,
                                         container_size);
    } else {
#ifdef TEXCOMP_HAVE_ASTCENC
        if (use_arm_encoder)
            tr = tc_cli_astcenc_compress(rgba, w, h, &astc_opt, compressed,
                                         compressed_size);
        else
#endif
        tr = tc_astc_compress_rgba8(rgba, w, h, (size_t)w * 4u, &astc_opt,
                                    compressed, compressed_size);
        if (tr == TC_SUCCESS)
            tr = tc_astc_write_file_memory(compressed, w, h, &astc_opt, container,
                                           container_size);
    }
    if (tr == TC_SUCCESS) tr = write_file(out, container, container_size);
    if (tr == TC_SUCCESS && raw) tr = write_file(raw, compressed, compressed_size);
    if (tr != TC_SUCCESS) fprintf(stderr, "texcomp: write failed: %s\n", tc_result_string(tr));
    free(rgba);
    free(rgbf); free(rgbaf);
    free(compressed);
    free(container);
    return tr == TC_SUCCESS ? 0 : 1;
}
