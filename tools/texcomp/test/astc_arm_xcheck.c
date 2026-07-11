/*
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Self-contained CI cross-check for the vendored Arm astcenc backend.
 *
 * Builds only with -DTEXCOMP_HAVE_ASTCENC. Encodes one deterministic image
 * with both the pure-C `tc` encoder and the vendored astcenc encoder, decodes
 * both with astcenc's own conformant decoder, and asserts both clear a PSNR
 * floor and land within tolerance of each other. Needs no external
 * astcenc-native binary and no image assets, so it runs in CI wherever the
 * C++ backend builds. This is the gate that keeps `deps/astcenc` compiling
 * and wired up, and catches gross regressions in either encoder.
 *
 * Both outputs are decoded with the same astcenc decoder (unorm8 decode mode)
 * so the comparison is apples-to-apples; the in-tree reference decoder is not
 * used here because it deliberately supports only the block-mode subset the
 * `tc` encoder emits, not astcenc's full space.
 */
#include "astcenc.h"
#include "texcomp.h"
#include "../src/texcomp_internal.h" /* tc_astc_decode_block_rgba8 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double psnr8(const uint8_t *a, const uint8_t *b, size_t n) {
    uint64_t sse = 0;
    size_t i;
    double mse;
    for (i = 0; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        sse += (uint64_t)(d * d);
    }
    if (!sse) return 99.0;
    mse = (double)sse / (double)n;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

/* Single-thread astcenc encode, mirroring tc_cli_astcenc_compress(). */
static int arm_encode(const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t bx,
                      uint32_t by, float preset, uint8_t *out, size_t out_size) {
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = (void *)rgba;
    int rc;
    if (astcenc_config_init(ASTCENC_PRF_LDR, bx, by, 1u, preset, 0, &cfg) !=
        ASTCENC_SUCCESS)
        return -1;
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS)
        return -1;
    img.dim_x = w;
    img.dim_y = h;
    img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_U8;
    img.data = &slice;
    rc = astcenc_compress_image(ctx, &img, &swz, out, out_size, 0u) ==
                 ASTCENC_SUCCESS
             ? 0
             : -1;
    astcenc_context_free(ctx);
    return rc;
}

/* Decode any standard LDR ASTC stream to unorm8 RGBA via astcenc. */
static int astc_decode(const uint8_t *blocks, size_t len, uint32_t w,
                       uint32_t h, uint32_t bx, uint32_t by, uint8_t *out) {
    struct astcenc_config cfg;
    struct astcenc_context *ctx = NULL;
    struct astcenc_image img;
    const struct astcenc_swizzle swz = {ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                        ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    void *slice = out;
    int rc;
    if (astcenc_config_init(ASTCENC_PRF_LDR, bx, by, 1u, ASTCENC_PRE_MEDIUM,
                            ASTCENC_FLG_DECOMPRESS_ONLY |
                                ASTCENC_FLG_USE_DECODE_UNORM8,
                            &cfg) != ASTCENC_SUCCESS)
        return -1;
    if (astcenc_context_alloc(&cfg, 1u, &ctx, NULL) != ASTCENC_SUCCESS)
        return -1;
    img.dim_x = w;
    img.dim_y = h;
    img.dim_z = 1u;
    img.data_type = ASTCENC_TYPE_U8;
    img.data = &slice;
    rc = astcenc_decompress_image(ctx, blocks, len, &img, &swz, 0u) ==
                 ASTCENC_SUCCESS
             ? 0
             : -1;
    astcenc_context_free(ctx);
    return rc;
}

/* Smooth RGB gradient with a little deterministic noise + flat alpha; the
 * kind of content where both encoders should land close together. */
static void fill_image(uint8_t *img, uint32_t w, uint32_t h) {
    unsigned s = 2654435761u;
    uint32_t x, y;
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            uint8_t *p = img + ((size_t)y * w + x) * 4u;
            int r = (int)(x * 255u / (w - 1u));
            int g = (int)(y * 255u / (h - 1u));
            int b = (int)((x + y) * 255u / (w + h - 2u));
            int n;
            s = s * 1664525u + 1013904223u;
            n = (int)((s >> 26) & 15u) - 8; /* +/-8 */
            r += n;
            p[0] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            p[1] = (uint8_t)g;
            p[2] = (uint8_t)b;
            p[3] = 255u;
        }
    }
}


/* ---------------------------------------------------------------------------
 * Foreign-block decode conformance for the LDR decoder.
 *
 * Everything else in this gate feeds our decoder blocks our own encoder made,
 * which only uses a corner of the format. Here astcenc encodes and *we* decode,
 * so the decoder sees what a real third-party ASTC LDR asset contains: the
 * base+offset endpoint modes, mixed-CEM partitions, dual-plane blocks and a
 * spread of footprints. The two decodes must agree exactly -- this is the check
 * that caught our texel interpolation being off by 1 LSB from a GPU (we
 * interpolated in 8 bits; the spec bit-replicates the endpoints to 16 first).
 * ------------------------------------------------------------------------- */
static uint32_t ldrx_rs = 7u;
static uint32_t ldrx_rnd(void) {
    ldrx_rs = ldrx_rs * 1664525u + 1013904223u;
    return ldrx_rs >> 8;
}

static int ldr_foreign_xcheck(void) {
    enum { W = 32, H = 32 };
    static uint8_t src[W * H * 4];
    static uint8_t adec[W * H * 4];
    static uint8_t blocks[64 * 64 * 16];
    static const uint32_t bs[6][2] = {{4,4},{5,4},{6,6},{8,5},{8,8},{10,6}};
    static const float pres[3] = {ASTCENC_PRE_FAST, ASTCENC_PRE_MEDIUM,
                                  ASTCENC_PRE_THOROUGH};
    long tot = 0, bad = 0, unsup = 0;
    int trial;

    for (trial = 0; trial < 120; ++trial) {
        uint32_t bx = bs[trial % 6][0], by = bs[trial % 6][1];
        int kind = (trial / 6) % 5;
        size_t nb = (size_t)((W + bx - 1u) / bx) * ((H + by - 1u) / by);
        size_t len = nb * 16u;
        uint32_t x, y, i, bxc = (W + bx - 1u) / bx;

        for (y = 0; y < H; ++y)
            for (x = 0; x < W; ++x) {
                uint8_t *p = src + ((size_t)y * W + x) * 4u;
                switch (kind) {
                case 0: p[0]=(uint8_t)ldrx_rnd(); p[1]=(uint8_t)ldrx_rnd();
                        p[2]=(uint8_t)ldrx_rnd(); p[3]=(uint8_t)ldrx_rnd(); break;
                case 1: p[0]=(uint8_t)(x*8u); p[1]=(uint8_t)(y*8u); p[2]=128u; p[3]=255u; break;
                case 2: { uint8_t g=(uint8_t)ldrx_rnd(); p[0]=p[1]=p[2]=g; p[3]=255u; break; }
                case 3: p[0]=200u; p[1]=100u; p[2]=50u;
                        p[3]=(uint8_t)((x * 7u + y * 3u) & 0xffu); break;
                default: p[0]=(uint8_t)((x & 1u) ? 255u : 0u);
                         p[1]=(uint8_t)((y & 1u) ? 255u : 0u);
                         p[2]=(uint8_t)ldrx_rnd(); p[3]=255u; break;
                }
            }

        if (len > sizeof(blocks)) return 1;
        if (arm_encode(src, W, H, bx, by, pres[trial % 3], blocks, len) != 0) {
            fprintf(stderr, "FAIL: astcenc encode (LDR foreign sweep)\n");
            return 1;
        }
        if (astc_decode(blocks, len, W, H, bx, by, adec) != 0) {
            fprintf(stderr, "FAIL: astcenc decode (LDR foreign sweep)\n");
            return 1;
        }
        for (i = 0; i < nb; ++i) {
            uint8_t ours[12 * 12 * 4];
            const uint8_t *b = blocks + (size_t)i * 16u;
            uint32_t ox = (i % bxc) * bx, oy = (i / bxc) * by, xx, yy;
            int isbad = 0;
            tot++;
            if (!tc_astc_decode_block_rgba8(b, bx, by, ours)) { unsup++; continue; }
            for (yy = 0; yy < by && !isbad; ++yy)
                for (xx = 0; xx < bx && !isbad; ++xx) {
                    uint32_t px = ox + xx, py = oy + yy, c;
                    if (px >= W || py >= H) continue;
                    for (c = 0; c < 4u; ++c)
                        if (ours[(yy * bx + xx) * 4u + c] !=
                            adec[((size_t)py * W + px) * 4u + c]) {
                            bad++; isbad = 1; break;
                        }
                }
        }
    }
    printf("astc-ldr foreign blocks (astcenc-encoded): %ld blocks, %ld unsupported, "
           "%ld mismatched\n", tot, unsup, bad);
    if (tot < 2000) {
        fprintf(stderr, "FAIL: LDR foreign sweep coverage floor\n");
        return 1;
    }
    if (unsup || bad) {
        fprintf(stderr, "FAIL: pure-C LDR decoder disagrees with astcenc\n");
        return 1;
    }
    return 0;
}

int main(void) {
    enum { W = 64, H = 64 };
    static uint8_t img[W * H * 4];
    static uint8_t dec_tc[W * H * 4], dec_arm[W * H * 4];
    static uint8_t blk_tc[(W / 4) * (H / 4) * 16];
    static uint8_t blk_arm[(W / 4) * (H / 4) * 16];
    const uint32_t bx = 6u, by = 6u;
    const double FLOOR = 30.0; /* both encoders must clear this */
    const double TOL = 3.0;    /* arm may not trail tc by more than this */
    tc_astc_options opt;
    size_t need;
    double ptc, parm;

    fill_image(img, W, H);
    tc_astc_options_init(&opt);
    opt.block_x = bx;
    opt.block_y = by;
    opt.quality = 2; /* normal */
    need = tc_astc_compressed_size(W, H, &opt);
    if (need > sizeof(blk_tc)) {
        fprintf(stderr, "FAIL: block buffer too small\n");
        return 1;
    }
    if (tc_astc_compress_rgba8(img, W, H, W * 4u, &opt, blk_tc, need) !=
        TC_SUCCESS) {
        fprintf(stderr, "FAIL: tc encode\n");
        return 1;
    }
    if (arm_encode(img, W, H, bx, by, ASTCENC_PRE_THOROUGH, blk_arm, need) !=
        0) {
        fprintf(stderr, "FAIL: arm encode\n");
        return 1;
    }
    if (astc_decode(blk_tc, need, W, H, bx, by, dec_tc) != 0) {
        fprintf(stderr, "FAIL: tc output decode\n");
        return 1;
    }
    if (astc_decode(blk_arm, need, W, H, bx, by, dec_arm) != 0) {
        fprintf(stderr, "FAIL: arm output decode\n");
        return 1;
    }
    ptc = psnr8(img, dec_tc, sizeof(img));
    parm = psnr8(img, dec_arm, sizeof(img));
    printf("astc arm xcheck %ux%u: tc=%.2f dB  arm=%.2f dB  (arm-tc %.2f)\n",
           bx, by, ptc, parm, parm - ptc);
    if (ptc < FLOOR || parm < FLOOR) {
        fprintf(stderr, "FAIL: psnr below floor %.1f dB\n", FLOOR);
        return 1;
    }
    if (parm < ptc - TOL) {
        fprintf(stderr, "FAIL: arm %.2f dB trails tc %.2f dB by > %.1f dB\n",
                parm, ptc, TOL);
        return 1;
    }
    /* UASTC LDR 4x4: constrained encode must decode cleanly and use only the
     * single-subset CEM 8/12 modes (or the solid void-extent). */
    {
        static uint8_t ublk[(W / 4) * (H / 4) * 16];
        tc_astc_options uopt;
        size_t un, b;
        double pu;
        tc_astc_options_init(&uopt);
        uopt.block_x = 4;
        uopt.block_y = 4;
        uopt.uastc = 1;
        un = tc_astc_compressed_size(W, H, &uopt);
        if (tc_astc_compress_rgba8(img, W, H, W * 4u, &uopt, ublk, un) !=
            TC_SUCCESS) {
            fprintf(stderr, "FAIL: uastc encode\n");
            return 1;
        }
        if (astc_decode(ublk, un, W, H, 4, 4, dec_tc) != 0) {
            fprintf(stderr, "FAIL: uastc decode\n");
            return 1;
        }
        for (b = 0; b + 16 <= un; b += 16) {
            const uint8_t *p = ublk + b;
            unsigned bm = (unsigned)p[0] | ((unsigned)(p[1] & 7u) << 8);
            unsigned pc;
            if ((bm & 0x1ffu) == 0x1fcu) continue; /* void-extent (solid) */
            pc = ((unsigned)p[1] >> 3) & 3u; /* partition count - 1 */
            /* UASTC uses at most 3 subsets; single-subset CEM = bits 13..16
             * (3 in byte 1, 1 in byte 2) and must be LA (4), RGB (8) or
             * RGBA (12). */
            if (pc > 2u) {
                fprintf(stderr, "FAIL: uastc block %zu has %u subsets\n",
                        b / 16u, pc + 1u);
                return 1;
            }
            if (pc == 0u) {
                unsigned cem =
                    (((unsigned)p[1] >> 5) & 7u) | (((unsigned)p[2] & 1u) << 3);
                if (cem != 4u && cem != 8u && cem != 12u) {
                    fprintf(stderr,
                            "FAIL: uastc block %zu single-subset CEM %u\n",
                            b / 16u, cem);
                    return 1;
                }
            }
        }
        pu = psnr8(img, dec_tc, sizeof(img));
        printf("uastc-ldr 64x64: %.2f dB (constrained modes)\n", pu);
        if (pu < FLOOR) {
            fprintf(stderr, "FAIL: uastc psnr %.2f below floor\n", pu);
            return 1;
        }

        /* Grayscale + alpha correlated gradient: the CEM 4 (luminance+alpha)
         * modes carry this in 4 endpoint values (vs 8 for CEM 12), so they win
         * on grayscale and must round-trip through astcenc. Assert at least one
         * CEM 4 block is emitted and the image clears a high floor. */
        {
            uint32_t x2, y2, seen_cem4 = 0;
            double pg;
            for (y2 = 0; y2 < H; ++y2)
                for (x2 = 0; x2 < W; ++x2) {
                    double t = (double)((x2 & 3u) + (y2 & 3u)) / 6.0;
                    uint8_t l = (uint8_t)(30.0 + t * 200.0);
                    uint8_t *px = img + ((size_t)y2 * W + x2) * 4u;
                    px[0] = px[1] = px[2] = l;
                    px[3] = (uint8_t)(50.0 + t * 150.0);
                }
            if (tc_astc_compress_rgba8(img, W, H, W * 4u, &uopt, ublk, un) !=
                TC_SUCCESS) {
                fprintf(stderr, "FAIL: uastc gray encode\n");
                return 1;
            }
            if (astc_decode(ublk, un, W, H, 4, 4, dec_tc) != 0) {
                fprintf(stderr, "FAIL: uastc gray decode\n");
                return 1;
            }
            for (b = 0; b + 16 <= un; b += 16) {
                const uint8_t *p = ublk + b;
                unsigned bm = (unsigned)p[0] | ((unsigned)(p[1] & 7u) << 8);
                unsigned pc, cem;
                if ((bm & 0x1ffu) == 0x1fcu) continue;
                pc = ((unsigned)p[1] >> 3) & 3u;
                if (pc == 0u) {
                    cem = (((unsigned)p[1] >> 5) & 7u) |
                          (((unsigned)p[2] & 1u) << 3);
                } else {
                    /* all-same-CEM multi-partition: 6-bit field at bit 23 =
                     * (cem << 2), so cem = field >> 2. */
                    unsigned field = (((unsigned)p[2] >> 7) & 1u) |
                                     (((unsigned)p[3] & 0x1fu) << 1);
                    cem = field >> 2;
                }
                if (cem == 4u) seen_cem4 = 1;
            }
            pg = psnr8(img, dec_tc, sizeof(img));
            printf("uastc-ldr grayscale+alpha: %.2f dB (cem4 used=%u)\n", pg,
                   seen_cem4);
            if (!seen_cem4) {
                fprintf(stderr, "FAIL: uastc emitted no CEM 4 block on "
                                "grayscale+alpha\n");
                return 1;
            }
            if (pg < 35.0) {
                fprintf(stderr, "FAIL: uastc grayscale psnr %.2f below floor\n",
                        pg);
                return 1;
            }
        }
    }

    if (ldr_foreign_xcheck()) return 1;

    printf("astc arm xcheck: OK\n");
    return 0;
}
