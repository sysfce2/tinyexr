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
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS)
        return TC_ERROR_OUT_OF_MEMORY;
    img.dim_x = w;
    img.dim_y = h;
    img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_U8;
    img.data = &slice;
    if (astcenc_compress_image(ctx, &img, &swz, out, out_size, 0u) !=
        ASTCENC_SUCCESS)
        tr = TC_ERROR_UNSUPPORTED;
    astcenc_context_free(ctx);
    return tr;
}
#endif
#include "exr.h"

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

static void usage(void) {
    fprintf(stderr,
            "usage: texcomp -i in.{png,exr} -o out [--format bc7|bc5|bc6h|etc2|etc2_rgb|eac_r11|eac_rg11|astc] "
            "[--raw out.bin] [--raw-bc7 out.bc7] [--part N] [--srgb] "
            "[--signed] [--astc-block WxH] [--quality fast|medium|normal] [--encoder tc|arm] "
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
    tc_bc5_options bc5_opt;
    tc_bc6h_options bc6h_opt;
    tc_etc2_options etc2_opt;
    tc_astc_options astc_opt;
    int use_arm_encoder = 0;
    tc_result tr;
    int i;

    tc_bc7_options_init(&bc7_opt);
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
        else {
            usage();
            return 2;
        }
    }
    if (!in || !out) {
        usage();
        return 2;
    }

    if (strcmp(format, "bc6") == 0) format = "bc6h";
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
    if (strcmp(format, "bc6h") == 0 && ends_with(in, ".exr")) {
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

    if (strcmp(format, "bc7") == 0) {
        compressed_size = tc_bc7_compressed_size(w, h);
        container_size = tc_dds_bc7_size(w, h);
    } else if (strcmp(format, "bc5") == 0) {
        compressed_size = tc_bc5_compressed_size(w, h);
        container_size = tc_dds_bc5_size(w, h);
    } else if (strcmp(format, "bc6h") == 0) {
        compressed_size = tc_bc6h_compressed_size(w, h);
        container_size = tc_dds_bc6h_size(w, h);
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
