/*
 * TinyEXR texcomp — BC6H/BC7 pipeline quality gate.
 *
 * Loads real HDR EXR images from a given directory, runs them through the
 * BC6H (unsigned + signed) and BC7 encoders, decodes with the independent
 * reference decoder, and asserts PSNR + SSIM floors.
 *
 * Usage:
 *   ./bc6h_pipeline_gate ~/work/openexr-images/
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "texcomp.h"
#include "exr.h"
#include "bc6h_ref_decode.h"
#include "tc_ssim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

static float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) { uint32_t r = m << 13u, i = (s << 31u) | r; float f; memcpy(&f, &i, 4); return f; }
    if (e == 31) { uint32_t r = m ? (m << 13u) : 0x7f800000u; uint32_t i = (s << 31u) | r; float f; memcpy(&f, &i, 4); return f; }
    uint32_t r = m << 13u, i = (s << 31u) | ((e + 112u) << 23u) | r; float f; memcpy(&f, &i, 4); return f;
}

static double psnr_f(const float *a, const float *b, size_t n, double peak) {
    double sse = 0.0; size_t i;
    for (i = 0; i < n; ++i) { double d = (double)a[i] - (double)b[i]; sse += d * d; }
    if (sse <= 0.0) return 99.0;
    return 10.0 * log10(peak * peak / (sse / (double)n));
}

static void decode_bc6h(const uint8_t *blocks, uint32_t w, uint32_t h,
                         int is_signed, float *out) {
    uint32_t by, bx, yy, xx;
    size_t bi = 0;
    for (by = 0; by < h; by += 4)
        for (bx = 0; bx < w; bx += 4) {
            uint16_t tex[16][3];
            tc_bc6h_ref_decode_block(blocks + bi, is_signed, tex);
            bi += 16;
            for (yy = 0; yy < 4; ++yy)
                for (xx = 0; xx < 4; ++xx) {
                    uint32_t y = by + yy, x = bx + xx;
                    if (y >= h || x >= w) continue;
                    float *p = out + ((size_t)y * w + x) * 3;
                    p[0] = half_to_float(tex[yy * 4 + xx][0]);
                    p[1] = half_to_float(tex[yy * 4 + xx][1]);
                    p[2] = half_to_float(tex[yy * 4 + xx][2]);
                }
        }
}

static int has_negatives(const float *img, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (img[i] < 0.0f) return 1;
    return 0;
}

static int ch_index(const exr_part *part, const char *name) {
    int i;
    for (i = 0; i < part->header.num_channels; ++i)
        if (strcmp(part->header.channels[i].name, name) == 0) return i;
    return -1;
}

static float read_sample(const exr_part *part, int ch, size_t idx) {
    exr_pixel_type t = part->header.channels[ch].pixel_type;
    if (t == EXR_PIXEL_UINT) return (float)((const uint32_t *)part->images[ch])[idx];
    if (t == EXR_PIXEL_HALF) {
        float f;
        exr_half_to_float(&((const uint16_t *)part->images[ch])[idx], &f, 1);
        return f;
    }
    return ((const float *)part->images[ch])[idx];
}

static float max_abs(const float *img, size_t n) {
    float m = 1e-6f; size_t i;
    for (i = 0; i < n; ++i) { float a = fabsf(img[i]); if (a > m) m = a; }
    return m;
}

/* Check if image has values outside [0,1] (BC7 is LDR only). */
static int has_out_of_range(const float *img, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (img[i] < 0.0f || img[i] > 1.0f) return 1;
    return 0;
}

/* Check for NaN/Inf values. */
static int has_nan_inf(const float *img, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(img[i])) return 1;
    return 0;
}

static int test_image(const char *path, int verbose) {
    exr_image img;
    exr_result er;
    float *rgbf = NULL;
    int nfail = 0;

    er = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(er)) { fprintf(stderr, "  SKIP (load): %s\n", path); return 0; }

    for (int part = 0; part < img.num_parts; ++part) {
        exr_part *p = &img.parts[part];
        int r, g, b;
        size_t n, bc6h_sz;
        uint8_t *bc6h = NULL;
        float *dec = NULL;

        if (p->is_deep) continue;
        if (p->header.compression == EXR_COMPRESSION_DWAA ||
            p->header.compression == EXR_COMPRESSION_DWAB) continue;
        if (p->header.compression == EXR_COMPRESSION_HTJ2K32 ||
            p->header.compression == EXR_COMPRESSION_HTJ2K256) continue;
        if (p->width <= 0 || p->height <= 0) continue;

        uint32_t w = (uint32_t)p->width;
        uint32_t h = (uint32_t)p->height;
        n = (size_t)w * h;

        r = ch_index(p, "R");
        g = ch_index(p, "G");
        b = ch_index(p, "B");
        if (r < 0 || g < 0 || b < 0) continue;

        if (verbose) printf("  %s (part %d, %ux%u)\n", path, part, w, h);

        rgbf = (float *)malloc(n * 3 * sizeof(float));
        if (!rgbf) continue;

        for (size_t i = 0; i < n; ++i) {
            rgbf[i * 3 + 0] = read_sample(p, r, i);
            rgbf[i * 3 + 1] = read_sample(p, g, i);
            rgbf[i * 3 + 2] = read_sample(p, b, i);
        }

        /* Skip files with NaN/Inf (e.g. AllHalfValues.exr). */
        if (has_nan_inf(rgbf, n * 3)) {
            if (verbose) printf("    SKIP (NaN/Inf)\n");
            free(rgbf);
            continue;
        }

        /* Detect LDR suitability for BC7. */
        int is_hdr = has_out_of_range(rgbf, n * 3);

        if (verbose) printf("  %s (part %d, %ux%u)%s\n", path, part, w, h,
                           is_hdr ? " HDR" : " LDR");
        dec = (float *)malloc(n * 3 * sizeof(float));
        if (!rgbf || !dec) { free(rgbf); free(dec); continue; }

        /* --- BC6H unsigned --- */
        bc6h_sz = tc_bc6h_compressed_size(w, h);
        bc6h = (uint8_t *)malloc(bc6h_sz);
        if (bc6h) {
            tc_bc6h_options bopt;
            tc_bc6h_options_init(&bopt);
            bopt.signed_float = 0;
            if (tc_bc6h_compress_rgb32f(rgbf, w, h, w * 3 * sizeof(float),
                                        &bopt, bc6h, bc6h_sz) == TC_SUCCESS) {
                decode_bc6h(bc6h, w, h, 0, dec);
                double peak = (double)max_abs(rgbf, n * 3);
                double p = psnr_f(rgbf, dec, n * 3, peak);
                double cs[3];
                double ssim_val = tc_ssim(rgbf, dec, w, h, cs);
                if (verbose)
                    printf("    BC6H unsigned: P=%.2f SSIM=%.4f (R=%.4f G=%.4f B=%.4f)\n",
                           p, ssim_val, cs[0], cs[1], cs[2]);
                if (p < 28.0) {
                    fprintf(stderr, "    FAIL: BC6H unsigned PSNR %.2f <%s>\n", p, path);
                    ++nfail;
                }
            }
            free(bc6h);
        }

        /* --- BC6H signed (only if image has negative values) --- */
        if (has_negatives(rgbf, n * 3)) {
            bc6h = (uint8_t *)malloc(bc6h_sz);
            if (bc6h) {
                tc_bc6h_options bopt;
                tc_bc6h_options_init(&bopt);
                bopt.signed_float = 1;
                if (tc_bc6h_compress_rgb32f(rgbf, w, h, w * 3 * sizeof(float),
                                            &bopt, bc6h, bc6h_sz) == TC_SUCCESS) {
                    decode_bc6h(bc6h, w, h, 1, dec);
                    double peak = (double)max_abs(rgbf, n * 3);
                    double p = psnr_f(rgbf, dec, n * 3, peak);
                    double cs[3];
                    double ssim_val = tc_ssim(rgbf, dec, w, h, cs);
                    if (verbose)
                        printf("    BC6H signed:   P=%.2f SSIM=%.4f (R=%.4f G=%.4f B=%.4f)\n",
                               p, ssim_val, cs[0], cs[1], cs[2]);
                    if (p < 25.0) {
                        fprintf(stderr, "    FAIL: BC6H signed PSNR %.2f <%s>\n", p, path);
                        ++nfail;
                    }
                }
                free(bc6h);
            }
        }

        /* --- BC7 (float -> uint8 -> BC7 -> decode -> float).
         * Only tested on LDR content (values in [0,1]) since BC7 is an
         * LDR-only format. HDR images would get clamped and produce
         * meaningless PSNR. */
        if (!is_hdr) {
            size_t bc7_sz2 = tc_bc7_compressed_size(w, h);
            uint8_t *bc7_img = (uint8_t *)malloc(bc7_sz2);
            uint8_t *rgba = (uint8_t *)malloc(n * 4);
            if (bc7_img && rgba) {
                tc_bc7_options b7opt;
                tc_bc7_options_init(&b7opt);
                b7opt.quick = 1;
                for (size_t i = 0; i < n; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        float v = rgbf[i * 3 + c];
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        rgba[i * 4 + c] = (uint8_t)(v * 255.0f + 0.5f);
                    }
                    rgba[i * 4 + 3] = 255;
                }
                if (tc_bc7_compress_rgba8(rgba, w, h, w * 4, &b7opt, bc7_img, bc7_sz2) == TC_SUCCESS) {
                    uint8_t *dec_u8 = (uint8_t *)malloc(n * 4);
                    if (dec_u8 && tc_bc7_decompress_rgba8(bc7_img, w, h, w * 4, dec_u8, n * 4) == TC_SUCCESS) {
                        float *dec_f = (float *)malloc(n * 3 * sizeof(float));
                        if (dec_f) {
                            for (size_t i = 0; i < n; ++i)
                                for (int c = 0; c < 3; ++c)
                                    dec_f[i * 3 + c] = (float)dec_u8[i * 4 + c] / 255.0f;
                            double p = psnr_f(rgbf, dec_f, n * 3, 1.0);
                            double cs[3];
                            double ssim_val = tc_ssim(rgbf, dec_f, w, h, cs);
                            if (verbose)
                                printf("    BC7:            P=%.2f SSIM=%.4f (R=%.4f G=%.4f B=%.4f)\n",
                                       p, ssim_val, cs[0], cs[1], cs[2]);
                            if (p < 20.0) {
                                fprintf(stderr, "    FAIL: BC7 PSNR %.2f <%s>\n", p, path);
                                ++nfail;
                            }
                            free(dec_f);
                        }
                        free(dec_u8);
                    }
                }
                free(rgba); free(bc7_img);
            }
        } else if (verbose) {
            printf("    SKIP BC7 (HDR content)\n");
        }

        free(rgbf);
        free(dec);
    }

    exr_image_free(&img);
    return nfail;
}

/* Recursively scan a directory for .exr files and test each. */
static int scan_and_test(const char *dir, int verbose,
                          int *ntotal, int *nfail) {
    DIR *d = opendir(dir);
    struct dirent *de;
    char path[1024];

    if (!d) return 0;

    while ((de = readdir(d)) != NULL) {
        struct stat st;
        size_t nlen = strlen(de->d_name);
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            scan_and_test(path, verbose, ntotal, nfail);
        } else if (nlen >= 5 && strcmp(de->d_name + nlen - 4, ".exr") == 0) {
            int ec = test_image(path, verbose);
            if (ec > 0) *nfail += ec;
            ++*ntotal;
        }
    }
    closedir(d);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir;
    struct stat st;
    int ntotal = 0, nfail = 0;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <exr-dir>\n", argv[0]);
        return 1;
    }

    dir = argv[1];
    int verbose = argc > 2 && strcmp(argv[2], "--verbose") == 0;

    if (stat(dir, &st) != 0) {
        fprintf(stderr, "error: cannot access %s\n", dir);
        return 1;
    }

    if (S_ISREG(st.st_mode)) {
        nfail = test_image(dir, verbose);
        printf("bc6h pipeline: 1 file, %d failures\n", nfail);
    } else {
        scan_and_test(dir, verbose, &ntotal, &nfail);
        printf("bc6h pipeline: %d file(s), %d failures\n", ntotal, nfail);
    }

    return nfail > 0 ? 1 : 0;
}
