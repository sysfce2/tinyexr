/* Self-contained CI cross-check for the vendored Arm astcenc backend.
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
    printf("astc arm xcheck: OK\n");
    return 0;
}
