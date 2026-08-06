/*
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 *
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
#include "../src/texcomp_internal.h" /* tc_encode_astc_hdr_cem15_block, lns16 */

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


/* ---------------------------------------------------------------------------
 * Foreign-block conformance: everything above feeds the decoder blocks our own
 * encoder produced, which only ever uses a corner of the format. These two
 * sweeps feed it blocks it did not make.
 *
 * (1) astcenc encodes; we decode. Covers what a real third-party ASTC HDR asset
 *     contains: mixed-CEM partitions, HDR luminance (CEM 2/3), CEM 7/11/15,
 *     HDR void-extent, several block sizes and presets.
 * (2) CEM mutation. astcenc's HDR profile never emits CEM 14, the LDR endpoint
 *     modes, or an LDR void-extent, so those would stay untested. CEMs within
 *     one class encode the same number of endpoint values, so rewriting a
 *     block's CEM field to another CEM of the same class yields a still-valid
 *     block -- which astcenc will decode, giving us an oracle for the modes it
 *     will not produce.
 *
 * Both require an exact match against astcenc, texel for texel.
 * ------------------------------------------------------------------------- */

static uint32_t xs_rs = 12345u;
static uint32_t xs_rnd(void) { xs_rs = xs_rs * 1664525u + 1013904223u; return xs_rs >> 8; }

static unsigned xs_rdb(const uint8_t *p, unsigned pos, unsigned n) {
    unsigned v = 0, i;
    for (i = 0; i < n; ++i)
        v |= (unsigned)((p[(pos + i) >> 3] >> ((pos + i) & 7)) & 1u) << i;
    return v;
}

static void xs_wrb(uint8_t *p, unsigned pos, unsigned n, unsigned v) {
    unsigned i;
    for (i = 0; i < n; ++i) {
        unsigned bit = pos + i, b = (v >> i) & 1u;
        p[bit >> 3] = (uint8_t)((p[bit >> 3] & ~(1u << (bit & 7))) | (b << (bit & 7)));
    }
}

static int xs_decode_blocks(const uint8_t *blocks, size_t len, uint32_t w,
                            uint32_t h, uint32_t bx, uint32_t by, float *dec);

/* astcenc encode / decode at an arbitrary footprint. */
static int xs_astcenc(const float *src, uint32_t w, uint32_t h, uint32_t bx,
                      uint32_t by, int preset, uint8_t *blocks, size_t len,
                      float *dec) {
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = (void *)(uintptr_t)src;
    if (astcenc_config_init(ASTCENC_PRF_HDR, bx, by, 1, (float)preset, 0, &cfg) !=
        ASTCENC_SUCCESS) return -1;
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS) return -1;
    img.dim_x = w; img.dim_y = h; img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_F32; img.data = &slice;
    if (astcenc_compress_image(ctx, &img, &swz, blocks, len, 0u) != ASTCENC_SUCCESS) {
        astcenc_context_free(ctx); return -1;
    }
    astcenc_context_free(ctx);
    return xs_decode_blocks(blocks, len, w, h, bx, by, dec);
}

static int xs_decode_blocks(const uint8_t *blocks, size_t len, uint32_t w,
                            uint32_t h, uint32_t bx, uint32_t by, float *dec) {
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = dec;
    int rc;
    if (astcenc_config_init(ASTCENC_PRF_HDR, bx, by, 1, ASTCENC_PRE_MEDIUM,
                            ASTCENC_FLG_DECOMPRESS_ONLY, &cfg) != ASTCENC_SUCCESS)
        return -1;
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS) return -1;
    img.dim_x = w; img.dim_y = h; img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_F32; img.data = &slice;
    rc = astcenc_decompress_image(ctx, blocks, len, &img, &swz, 0u) ==
                 ASTCENC_SUCCESS ? 0 : -1;
    astcenc_context_free(ctx);
    return rc;
}

/* Compare our block decode against astcenc's image decode. Returns bad count. */
static long xs_cmp(const uint8_t *blocks, size_t nb, uint32_t w, uint32_t h,
                   uint32_t bx, uint32_t by, const float *adec, long *tot) {
    uint32_t bxc = (w + bx - 1u) / bx;
    long bad = 0;
    size_t i;
    for (i = 0; i < nb; ++i) {
        float ours[12 * 12 * 4];
        const uint8_t *b = blocks + i * 16u;
        uint32_t ox = (uint32_t)(i % bxc) * bx, oy = (uint32_t)(i / bxc) * by;
        uint32_t xx, yy;
        int isbad = 0;
        (*tot)++;
        if (!tc_astc_decode_block_hdr_rgbaf(b, bx, by, ours)) { bad++; continue; }
        for (yy = 0; yy < by && !isbad; ++yy)
            for (xx = 0; xx < bx && !isbad; ++xx) {
                uint32_t px = ox + xx, py = oy + yy, c;
                if (px >= w || py >= h) continue;
                for (c = 0; c < 4u; ++c) {
                    float a = ours[(yy * bx + xx) * 4u + c];
                    float e = adec[((size_t)py * w + px) * 4u + c];
                    if (fabsf(a - e) > 1e-4f * (fabsf(e) + 1.0f)) {
                        bad++; isbad = 1; break;
                    }
                }
            }
    }
    return bad;
}

static int hdr_foreign_xcheck(void) {
    enum { W = 32, H = 32 };
    static float src[W * H * 4];
    static float adec[W * H * 4];
    static uint8_t blocks[64 * 64 * 16];
    static const uint32_t bs[6][2] = {{4,4},{5,4},{6,6},{8,5},{8,8},{10,6}};
    static const int pres[3] = {(int)ASTCENC_PRE_FAST, (int)ASTCENC_PRE_MEDIUM,
                                (int)ASTCENC_PRE_THOROUGH};
    /* CEM swaps within a class (same endpoint value count => still valid). */
    static const int swaps[13][2] = {{15,14},{15,12},{15,13},{11,8},{11,9},{11,10},
                                     {7,6},{7,4},{7,5},{2,0},{3,0},{3,1},{2,1}};
    long tot = 0, bad = 0, mut_tot = 0, mut_bad = 0, ve_tot = 0, ve_bad = 0;
    int trial;

    for (trial = 0; trial < 60; ++trial) {
        uint32_t bx = bs[trial % 6][0], by = bs[trial % 6][1];
        int kind = (trial / 6) % 10;
        size_t nb = (size_t)((W + bx - 1u) / bx) * ((H + by - 1u) / by);
        size_t len = nb * 16u;
        uint32_t x, y;
        int s;

        for (y = 0; y < H; ++y)
            for (x = 0; x < W; ++x) {
                float *p = src + ((size_t)y * W + x) * 4;
                float t = (float)(x + y) / (2.0f * (float)W);
                switch (kind) {
                case 0: p[0]=(float)(xs_rnd()%9000)/1000.f; p[1]=(float)(xs_rnd()%9000)/1000.f;
                        p[2]=(float)(xs_rnd()%9000)/1000.f; p[3]=(float)(xs_rnd()%9000)/1000.f; break;
                case 1: p[0]=t*8.f; p[1]=(1.f-t)*4.f; p[2]=2.f; p[3]=1.f; break;
                case 2: p[0]=p[1]=p[2]=(float)(xs_rnd()%9000)/1000.f; p[3]=1.f; break;
                case 3: p[0]=(float)(x&1u); p[1]=(float)(y&1u); p[2]=0.5f; p[3]=(float)((x+y)&1u); break;
                case 4: p[0]=0.1f+t*20.f; p[1]=0.2f; p[2]=t*t*30.f; p[3]=t*5.f; break;
                case 5: p[0]=(float)(xs_rnd()%256)/255.f; p[1]=(float)(xs_rnd()%256)/255.f;
                        p[2]=(float)(xs_rnd()%256)/255.f; p[3]=(float)(xs_rnd()%256)/255.f; break;
                case 6: p[0]=3.5f; p[1]=1.25f; p[2]=0.75f; p[3]=1.f; break; /* -> void-extent */
                case 7: p[0]=0.5f+t*30.f; p[1]=0.3f+t*20.f; p[2]=0.2f+t*10.f;
                        p[3]=(float)((x*7u+y*3u)%256u)/255.f; break;         /* HDR rgb, LDR-ish a */
                case 8: p[0]=p[1]=p[2]=0.05f+t*40.f; p[3]=1.f; break;        /* -> HDR luminance */
                default:
                    if (x < W/2u) { p[0]=(float)(xs_rnd()%256)/255.f; p[1]=p[0]; p[2]=p[0]; p[3]=1.f; }
                    else { p[0]=(float)(xs_rnd()%5000)/1000.f; p[1]=(float)(xs_rnd()%3000)/1000.f;
                           p[2]=(float)(xs_rnd()%7000)/1000.f; p[3]=0.5f; }
                    break;
                }
            }

        if (len > sizeof(blocks)) return 1;
        if (xs_astcenc(src, W, H, bx, by, pres[trial % 3], blocks, len, adec) != 0) {
            fprintf(stderr, "FAIL: astcenc encode/decode (foreign sweep)\n");
            return 1;
        }
        bad += xs_cmp(blocks, nb, W, H, bx, by, adec, &tot);

        /* (2) CEM mutation, on the 4x4 trials (single-subset blocks only). */
        if (bx != 4u || by != 4u) continue;
        for (s = 0; s < 13; ++s) {
            static uint8_t mut[64 * 64 * 16];
            size_t i;
            int hit = 0;
            memcpy(mut, blocks, len);
            for (i = 0; i < nb; ++i) {
                uint8_t *b = mut + i * 16u;
                if ((xs_rdb(b, 0, 11) & 0x1ffu) == 0x1fcu) continue; /* void-extent */
                if (xs_rdb(b, 11, 2) != 0u) continue;                /* single subset */
                if ((int)xs_rdb(b, 13, 4) != swaps[s][0]) continue;
                xs_wrb(b, 13, 4, (unsigned)swaps[s][1]);
                hit = 1;
            }
            if (!hit) continue;
            if (xs_decode_blocks(mut, len, W, H, 4u, 4u, adec) != 0) return 1;
            mut_bad += xs_cmp(mut, nb, W, H, 4u, 4u, adec, &mut_tot);
        }
        /* LDR void-extent: clear the HDR flag on any void-extent block. */
        {
            static uint8_t mut[64 * 64 * 16];
            size_t i;
            int hit = 0;
            memcpy(mut, blocks, len);
            for (i = 0; i < nb; ++i) {
                uint8_t *b = mut + i * 16u;
                if ((xs_rdb(b, 0, 11) & 0x1ffu) != 0x1fcu) continue;
                xs_wrb(b, 9, 1, 0u);
                hit = 1;
            }
            if (hit) {
                if (xs_decode_blocks(mut, len, W, H, 4u, 4u, adec) != 0) return 1;
                ve_bad += xs_cmp(mut, nb, W, H, 4u, 4u, adec, &ve_tot);
            }
        }
    }

    printf("astc-hdr foreign blocks (astcenc-encoded): %ld blocks, %ld bad\n", tot, bad);
    printf("astc-hdr mutated CEMs: %ld blocks, %ld bad\n", mut_tot, mut_bad);
    printf("astc-hdr LDR void-extent: %ld blocks, %ld bad\n", ve_tot, ve_bad);
    if (tot < 1000 || mut_tot < 100 || ve_tot < 10) {
        fprintf(stderr, "FAIL: foreign-block sweep did not reach its coverage floor\n");
        return 1;
    }
    if (bad || mut_bad || ve_bad) {
        fprintf(stderr, "FAIL: pure-C HDR decoder disagrees with astcenc on foreign blocks\n");
        return 1;
    }
    return 0;
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

    /* (f) HDR RGB + HDR alpha via the CEM 15 block encoder: an RGBA gradient
     * (alpha correlated with the colour so a single weight line fits all four
     * channels). astcenc must decode all four channels -- proves the CEM 15
     * field, 8-value endpoint stream (incl. the 2 HDR-alpha values) and QUANT_6
     * weights are all assembled correctly. */
    {
        static float srgba[W * H * 4];
        double sse = 0.0, pa;
        size_t j, c2;
        for (y = 0; y < H; ++y)
            for (x = 0; x < W; ++x) {
                float t = (float)x / (float)(W - 1);
                float *px = srgba + ((size_t)y * W + x) * 4;
                px[0] = 0.2f + t * 18.0f;
                px[1] = 0.15f + t * 12.0f;
                px[2] = 0.10f + t * 8.0f;
                px[3] = 0.05f + t * 10.0f; /* HDR alpha, correlated */
            }
        /* Full public RGBA-HDR path (CEM 15 blocks + void-extent fallback). */
        if (tc_astc_hdr_compress_rgbaf(srgba, W, H,
                                       (size_t)W * 4u * sizeof(float), &opt,
                                       blocks, need) != TC_SUCCESS) {
            fprintf(stderr, "FAIL: hdr rgba encode (cem15)\n");
            return 1;
        }
        if (astc_hdr_decode(blocks, need, W, H, dec) != 0) {
            fprintf(stderr, "FAIL: astcenc hdr decode (cem15)\n");
            return 1;
        }
        for (j = 0; j < (size_t)W * H; ++j)
            for (c2 = 0; c2 < 4; ++c2) {
                double dd = (double)srgba[j * 4 + c2] - (double)dec[j * 4 + c2];
                sse += dd * dd;
            }
        pa = 10.0 * log10(18.0 * 18.0 / (sse / ((double)W * H * 4)));
        printf("astc-hdr cem15 rgba 64x64: %.2f dB\n", pa);
        if (pa < 42.0) {
            fprintf(stderr, "FAIL: cem15 rgba psnr %.2f dB below floor\n", pa);
            return 1;
        }
        /* Independent pure-C decode must match astcenc on all four channels
         * (including alpha) -- validates the pure-C CEM 15 decode path. */
        {
            uint32_t bxc = (W + 3u) / 4u, bx, by, xx, yy;
            for (by = 0; by < H; by += 4u)
                for (bx = 0; bx < W; bx += 4u) {
                    float d4[16 * 4];
                    if (!ahref_decode_block_hdr(
                            blocks + ((size_t)(by / 4u) * bxc + bx / 4u) * 16u,
                            4, 4, d4)) {
                        fprintf(stderr, "FAIL: pure-C cem15 decode failed\n");
                        return 1;
                    }
                    for (yy = 0; yy < 4u; ++yy)
                        for (xx = 0; xx < 4u; ++xx) {
                            uint32_t px = bx + xx, py = by + yy, c;
                            if (px >= W || py >= H) continue;
                            for (c = 0; c < 4u; ++c)
                                if (d4[(yy * 4u + xx) * 4u + c] !=
                                    dec[((size_t)py * W + px) * 4u + c]) {
                                    fprintf(stderr, "FAIL: pure-C CEM 15 decoder "
                                                    "disagrees with astcenc\n");
                                    return 1;
                                }
                        }
                }
        }
    }

    if (hdr_foreign_xcheck()) return 1;

    printf("astc hdr xcheck: OK\n");
    return 0;
}
