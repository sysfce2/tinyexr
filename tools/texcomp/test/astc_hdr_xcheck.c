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
#include "astc_hdr_ref_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cross-check the pure-C HDR reference decoder against astcenc: decode every
 * block both ways and require the floats to match exactly. */
static int hdr_xcheck_refdec(const uint8_t *blocks, uint32_t w, uint32_t h,
                             const float *astc_dec) {
    uint32_t bxc = (w + 3u) / 4u, bx, by, xx, yy;
    for (by = 0; by < h; by += 4u)
        for (bx = 0; bx < w; bx += 4u) {
            float dec[16 * 4];
            if (!ahref_decode_block_hdr(
                    blocks + ((size_t)(by / 4u) * bxc + bx / 4u) * 16u, 4, 4,
                    dec))
                return 0;
            for (yy = 0; yy < 4u; ++yy)
                for (xx = 0; xx < 4u; ++xx) {
                    uint32_t x = bx + xx, y = by + yy, c;
                    if (x >= w || y >= h) continue;
                    for (c = 0; c < 3u; ++c)
                        if (dec[(yy * 4u + xx) * 4u + c] !=
                            astc_dec[((size_t)y * w + x) * 4u + c])
                            return 0;
                }
        }
    return 1;
}

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
    if (!hdr_xcheck_refdec(blocks, W, H, dec)) {
        fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc (const/void-extent)\n");
        return 1;
    }
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
    if (!hdr_xcheck_refdec(blocks, W, H, dec)) {
        fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc (gradient/single-subset)\n");
        return 1;
    }
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
    if (!hdr_xcheck_refdec(blocks, W, H, dec)) {
        fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc (anti-correlated/dual-plane)\n");
        return 1;
    }
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
    if (!hdr_xcheck_refdec(blocks, W, H, dec)) {
        fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc (two-gradient/2-subset)\n");
        return 1;
    }
    if (p < 20.0) {
        fprintf(stderr, "FAIL: two-gradient psnr %.2f dB below floor\n", p);
        return 1;
    }

    /* (e) fixed near-gray hue with a per-block multiplicative brightness ramp:
     * a near-uniform LNS offset between the two endpoints, so CEM 7 (base+scale,
     * 4 values -> a finer weight grid) should win over CEM 11 here. Assert at
     * least one CEM 7 block is emitted, that it round-trips through astcenc, and
     * that the pure-C decoder agrees. */
    {
        uint32_t seen_cem7 = 0, b;
        for (y = 0; y < H; ++y)
            for (x = 0; x < W; ++x) {
                float bmul = 1.0f + 3.0f * (float)((x & 3u) + (y & 3u)) / 6.0f;
                float *px = src + ((size_t)y * W + x) * 3;
                px[0] = 1.0f * bmul;
                px[1] = 0.92f * bmul;
                px[2] = 0.85f * bmul;
            }
        if (tc_astc_hdr_compress_rgbf(src, W, H, (size_t)W * 3u * sizeof(float),
                                      &opt, blocks, need) != TC_SUCCESS) {
            fprintf(stderr, "FAIL: hdr encode (brightness ramp)\n");
            return 1;
        }
        if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
            fprintf(stderr, "FAIL: astcenc hdr decode (brightness ramp)\n");
            return 1;
        }
        for (b = 0; b + 16u <= need; b += 16u) {
            const uint8_t *p = blocks + b;
            unsigned bm = (unsigned)p[0] | ((unsigned)(p[1] & 7u) << 8);
            if ((bm & 0x1ffu) == 0x1fcu) continue; /* void-extent */
            if (((p[1] >> 3) & 3u) != 0u) continue; /* single-subset only */
            if (((((unsigned)p[1] >> 5) & 7u) | (((unsigned)p[2] & 1u) << 3)) ==
                7u)
                seen_cem7 = 1;
        }
        p = psnr_rgb(src, dec, (size_t)W * H, 4.0);
        printf("astc-hdr brightness-ramp 64x64: %.2f dB (cem7 used=%u)\n", p,
               seen_cem7);
        if (!seen_cem7) {
            fprintf(stderr, "FAIL: no CEM 7 block emitted on base+scale content\n");
            return 1;
        }
        if (!hdr_xcheck_refdec(blocks, W, H, dec)) {
            fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc "
                            "(brightness-ramp/CEM 7)\n");
            return 1;
        }
        if (p < 32.0) {
            fprintf(stderr, "FAIL: brightness-ramp psnr %.2f dB below floor\n", p);
            return 1;
        }
    }

    printf("astc hdr xcheck: OK\n");
    return 0;
}
