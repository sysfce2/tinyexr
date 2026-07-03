/*
 * TinyEXR texcomp CLI.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
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

static void usage(void) {
    fprintf(stderr,
            "usage: texcomp -i in.{png,exr} -o out [--format bc1|bc3|bc7|bc5|bc6h|etc2|etc2_rgb|eac_r11|eac_rg11|astc|astc_hdr|uastc_ldr|xbc7] "
            "[--raw out.bin] [--raw-bc7 out.bc7] [--part N] [--srgb] "
            "[--signed] [--astc-block WxH] [--quality fast|medium|normal] [--encoder tc|arm] [--threads N] "
            "[--quick on|off] [--mode-mask HEX] "
            "[--linear|--perceptual]\n");
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
    uint8_t *rgba = NULL, *compressed = NULL, *container = NULL;
    float *rgbf = NULL;
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
    if ((strcmp(format, "bc6h") == 0 || strcmp(format, "astc_hdr") == 0) &&
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
        free(rgbf);
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
        free(rgbf);
        return 1;
    }
    if (!compressed_size || !container_size) {
        fprintf(stderr, "texcomp: invalid output size for format: %s\n", format);
        free(rgba);
        free(rgbf);
        return 1;
    }
    compressed = (uint8_t *)malloc(compressed_size);
    container = (uint8_t *)malloc(container_size);
    if (!compressed || !container) {
        fprintf(stderr, "texcomp: out of memory\n");
        free(rgba);
        free(rgbf);
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
        if (!rgbf) tr = rgba_to_rgbf(rgba, w, h, &rgbf);
        if (tr == TC_SUCCESS)
            tr = tc_astc_hdr_compress_rgbf(rgbf, w, h,
                                           (size_t)w * 3u * sizeof(float),
                                           &hdr_opt, compressed, compressed_size);
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
    free(rgbf);
    free(compressed);
    free(container);
    return tr == TC_SUCCESS ? 0 : 1;
}
