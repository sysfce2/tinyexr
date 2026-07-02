/*
 * Self-contained CI cross-check for the ASTC HDR (UASTC HDR 4x4) encoder.
 *
 * Builds only with -DTEXCOMP_HAVE_ASTCENC. Encodes deterministic HDR images
 * with the pure-C tc ASTC-HDR encoder and decodes the resulting blocks with
 * astcenc's conformant HDR decoder, checking that:
 *   (a) a constant-colour HDR image round-trips near-exact (the void-extent
 *       FP16 block format is correct), and
 *   (b) a smooth HDR gradient clears a PSNR floor (the block pipeline works).
 * No external binaries or image assets required.
 */
#include "astcenc.h"
#include "texcomp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decode ASTC HDR blocks to float RGBA via astcenc (HDR profile). */
static int astc_hdr_decode(const uint8_t *blocks, size_t len, uint32_t w,
                           uint32_t h, float *out_rgba) {
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = out_rgba;
    int rc;
    if (astcenc_config_init(ASTCENC_PRF_HDR, 4, 4, 1, ASTCENC_PRE_MEDIUM,
                            ASTCENC_FLG_DECOMPRESS_ONLY, &cfg) != ASTCENC_SUCCESS)
        return -1;
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS)
        return -1;
    img.dim_x = w;
    img.dim_y = h;
    img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_F32;
    img.data = &slice;
    rc = astcenc_decompress_image(ctx, blocks, len, &img, &swz, 0u) ==
                 ASTCENC_SUCCESS
             ? 0
             : -1;
    astcenc_context_free(ctx);
    return rc;
}

/* PSNR over RGB channels. `src` is packed RGB (stride 3), `dec` is RGBA
 * (stride 4). peak = the image's own max value. */
static double psnr_rgb(const float *src, const float *dec, size_t texels,
                       double peak) {
    double sse = 0.0;
    size_t i, c;
    for (i = 0; i < texels; ++i) {
        for (c = 0; c < 3; ++c) {
            double d = (double)src[i * 3 + c] - (double)dec[i * 4 + c];
            sse += d * d;
        }
    }
    if (sse <= 0.0) return 99.0;
    return 10.0 * log10(peak * peak / (sse / (double)(texels * 3)));
}

int main(void) {
    enum { W = 64, H = 64 };
    static float src[W * H * 3];
    static float dec[W * H * 4];
    static uint8_t blocks[(W / 4) * (H / 4) * 16];
    tc_astc_hdr_options opt;
    size_t need, i;
    uint32_t x, y;
    double p;

    tc_astc_hdr_options_init(&opt);
    need = tc_astc_hdr_compressed_size(W, H);
    if (need > sizeof(blocks)) {
        fprintf(stderr, "FAIL: block buffer too small\n");
        return 1;
    }

    /* (a) constant-colour HDR image: must round-trip near-exact. */
    for (i = 0; i < (size_t)W * H; ++i) {
        src[i * 3 + 0] = 4.5f;
        src[i * 3 + 1] = 0.25f;
        src[i * 3 + 2] = 12.0f;
    }
    if (tc_astc_hdr_compress_rgbf(src, W, H, (size_t)W * 3u * sizeof(float),
                                  &opt, blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: hdr encode (const)\n");
        return 1;
    }
    if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
        fprintf(stderr, "FAIL: astcenc hdr decode (const)\n");
        return 1;
    }
    p = psnr_rgb(src, dec, (size_t)W * H, 12.0);
    printf("astc-hdr const 64x64: %.2f dB\n", p);
    if (p < 60.0) {
        fprintf(stderr, "FAIL: const-colour void-extent round-trip %.2f dB\n", p);
        return 1;
    }

    /* (b) smooth correlated HDR gradient: the per-texel CEM 11 base+offset
     * path should reconstruct this well above the void-extent baseline. */
    for (y = 0; y < H; ++y) {
        for (x = 0; x < W; ++x) {
            float t = (float)(x + y) / (float)(2 * (W - 1));
            float *px = src + ((size_t)y * W + x) * 3;
            px[0] = 0.2f + t * 20.0f;
            px[1] = 0.15f + t * 15.0f;
            px[2] = 0.10f + t * 10.0f;
        }
    }
    if (tc_astc_hdr_compress_rgbf(src, W, H, (size_t)W * 3u * sizeof(float),
                                  &opt, blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: hdr encode (gradient)\n");
        return 1;
    }
    if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
        fprintf(stderr, "FAIL: astcenc hdr decode (gradient)\n");
        return 1;
    }
    p = psnr_rgb(src, dec, (size_t)W * H, 24.0);
    printf("astc-hdr gradient 64x64: %.2f dB\n", p);
    if (p < 50.0) {
        fprintf(stderr, "FAIL: gradient psnr %.2f dB below floor\n", p);
        return 1;
    }

    /* (c) anti-correlated gradient (R up, G down, B quadratic): a single
     * weight line cannot fit this, so the dual-plane path must engage and
     * clear a floor above the ~44 dB single-subset ceiling. */
    for (y = 0; y < H; ++y) {
        for (x = 0; x < W; ++x) {
            float t = (float)(x + y) / (float)(2 * (W - 1));
            float *px = src + ((size_t)y * W + x) * 3;
            px[0] = 0.05f + t * 8.0f;
            px[1] = 0.02f + (1.0f - t) * 4.0f;
            px[2] = 0.10f + t * t * 16.0f;
        }
    }
    if (tc_astc_hdr_compress_rgbf(src, W, H, (size_t)W * 3u * sizeof(float),
                                  &opt, blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: hdr encode (anti-correlated)\n");
        return 1;
    }
    if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
        fprintf(stderr, "FAIL: astcenc hdr decode (anti-correlated)\n");
        return 1;
    }
    p = psnr_rgb(src, dec, (size_t)W * H, 24.0);
    printf("astc-hdr anti-correlated 64x64: %.2f dB\n", p);
    if (p < 46.0) {
        fprintf(stderr, "FAIL: anti-correlated psnr %.2f dB below floor\n", p);
        return 1;
    }

    /* (d) two differently-oriented gradients per block (left cols vary R,
     * right cols vary G): a single endpoint line cannot fit both, so the
     * 2-subset partition path must engage to clear the floor. */
    for (y = 0; y < H; ++y) {
        for (x = 0; x < W; ++x) {
            float t = (float)(y & 3) / 3.0f;
            float *px = src + ((size_t)y * W + x) * 3;
            if ((x & 3) < 2) {
                px[0] = 5.0f + t * 20.0f;
                px[1] = 2.0f;
            } else {
                px[0] = 2.0f;
                px[1] = 5.0f + t * 20.0f;
            }
            px[2] = 2.0f;
        }
    }
    if (tc_astc_hdr_compress_rgbf(src, W, H, (size_t)W * 3u * sizeof(float),
                                  &opt, blocks, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: hdr encode (two-gradient)\n");
        return 1;
    }
    if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
        fprintf(stderr, "FAIL: astcenc hdr decode (two-gradient)\n");
        return 1;
    }
    p = psnr_rgb(src, dec, (size_t)W * H, 24.0);
    printf("astc-hdr two-gradient 64x64: %.2f dB\n", p);
    if (p < 20.0) {
        fprintf(stderr, "FAIL: two-gradient psnr %.2f dB below floor\n", p);
        return 1;
    }

    printf("astc hdr xcheck: OK\n");
    return 0;
}
