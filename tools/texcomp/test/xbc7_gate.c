/*
 * Gate for the xbc7 path: BC7 windowed RDO (tc_bc7_options.rdo) + the zstd
 * container round-trip. No BC7 decoder is needed here -- the container is
 * lossless (transcode is a byte-exact zstd decode), and the RDO's distortion
 * is bounded by construction, so this checks the two properties that matter:
 *   (a) rdo == 0 is exactly the plain BC7 encoding (no accidental change), and
 *   (b) rdo  > 0 makes the stream materially more compressible under zstd,
 *       while staying the same size (still standard BC7).
 * The CLI encode->transcode bit-exact round-trip is exercised by the shell
 * step in the `texcomp-xbc7-gate` make target.
 */
#include "texcomp.h"
#include "tinyexr_zstd.h"
#include "bc7_ref_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PSNR (RGB) of a BC7 stream decoded with the reference decoder vs the source. */
static double bc7_psnr(const uint8_t *img, uint32_t w, uint32_t h,
                       const uint8_t *bc7) {
    uint32_t bxc = (w + 3u) / 4u, bx, by, xx, yy;
    double sse = 0.0;
    size_t npx = 0;
    for (by = 0; by < h; by += 4u)
        for (bx = 0; bx < w; bx += 4u) {
            uint8_t dec[16][4];
            size_t bi = ((size_t)(by / 4u) * bxc + bx / 4u) * 16u;
            tc_bc7_ref_decode_block(bc7 + bi, dec);
            for (yy = 0; yy < 4u; ++yy)
                for (xx = 0; xx < 4u; ++xx) {
                    uint32_t x = bx + xx, y = by + yy;
                    const uint8_t *s, *d;
                    int ch;
                    if (x >= w || y >= h) continue;
                    s = img + ((size_t)y * w + x) * 4u;
                    d = dec[yy * 4u + xx];
                    for (ch = 0; ch < 3; ++ch) {
                        int e = (int)s[ch] - d[ch];
                        sse += (double)e * e;
                    }
                    ++npx;
                }
        }
    if (sse <= 0.0) return 99.0;
    return 10.0 * log10(65025.0 / (sse / ((double)npx * 3.0)));
}

int main(void) {
    enum { W = 128, H = 128 };
    static uint8_t img[W * H * 4];
    uint32_t x, y;
    size_t need = tc_bc7_compressed_size(W, H);
    uint8_t *base = (uint8_t *)malloc(need);
    uint8_t *rdo = (uint8_t *)malloc(need);
    size_t zb = tinyexr_zstd_compress_bound(need);
    uint8_t *z0 = (uint8_t *)malloc(zb);
    uint8_t *z1 = (uint8_t *)malloc(zb);
    size_t zc0, zc1;
    tc_bc7_options o;

    /* Smooth gradients: each 4x4 block is distinct (so the plain BC7 stream is
     * not trivially compressible), but adjacent blocks differ only slightly, so
     * the windowed RDO can merge runs of near-identical blocks -- exactly the
     * case where RDO buys real compressibility without much distortion. */
    for (y = 0; y < H; ++y)
        for (x = 0; x < W; ++x) {
            uint8_t *p = img + ((size_t)y * W + x) * 4u;
            p[0] = (uint8_t)(x * 255u / (W - 1u));
            p[1] = (uint8_t)(y * 255u / (H - 1u));
            p[2] = (uint8_t)((x + y) * 255u / (W + H - 2u));
            p[3] = 255u;
        }

    if (!base || !rdo || !z0 || !z1) {
        fprintf(stderr, "FAIL: oom\n");
        return 1;
    }

    tc_bc7_options_init(&o);
    o.rdo = 0;
    if (tc_bc7_compress_rgba8(img, W, H, W * 4u, &o, base, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: bc7 encode (rdo=0)\n");
        return 1;
    }
    o.rdo = 12;
    if (tc_bc7_compress_rgba8(img, W, H, W * 4u, &o, rdo, need) != TC_SUCCESS) {
        fprintf(stderr, "FAIL: bc7 encode (rdo=12)\n");
        return 1;
    }

    /* (a) rdo=0 must be identical to a fresh plain encode. */
    {
        uint8_t *plain = (uint8_t *)malloc(need);
        tc_bc7_options p0;
        tc_bc7_options_init(&p0);
        if (tc_bc7_compress_rgba8(img, W, H, W * 4u, &p0, plain, need) !=
                TC_SUCCESS ||
            memcmp(plain, base, need) != 0) {
            fprintf(stderr, "FAIL: default options are not rdo-off\n");
            return 1;
        }
        free(plain);
    }

    zc0 = tinyexr_zstd_compress(z0, zb, base, need, 19);
    zc1 = tinyexr_zstd_compress(z1, zb, rdo, need, 19);
    if (tinyexr_zstd_is_error(zc0) || tinyexr_zstd_is_error(zc1)) {
        fprintf(stderr, "FAIL: zstd error\n");
        return 1;
    }
    printf("xbc7 gate %dx%d: zstd(bc7)=%zu  zstd(rdo=12)=%zu  (%.1f%%)\n", W, H,
           zc0, zc1, 100.0 * (double)zc1 / (double)zc0);

    /* (b) RDO must shrink the entropy-coded stream by a clear margin. */
    if (zc1 >= zc0 * 9u / 10u) {
        fprintf(stderr, "FAIL: rdo did not improve compressibility (%zu >= %zu)\n",
                zc1, zc0);
        return 1;
    }

    /* (c) Reference-decoder self-check: a solid-colour block must decode back
     * exactly, and the RDO must not *raise* PSNR (it only ever trades quality
     * for size). This validates the decoder end-to-end without any external
     * reference. */
    {
        static uint8_t solid[16 * 16 * 4];
        uint8_t sb[16], sdec[16][4];
        tc_bc7_options so;
        uint32_t t;
        double pbase, prdo;
        for (t = 0; t < 16u * 16u; ++t) {
            solid[t * 4 + 0] = 173u;
            solid[t * 4 + 1] = 66u;
            solid[t * 4 + 2] = 214u;
            solid[t * 4 + 3] = 255u;
        }
        tc_bc7_options_init(&so);
        if (tc_bc7_compress_rgba8(solid, 16, 16, 16 * 4u, &so, base,
                                  tc_bc7_compressed_size(16, 16)) != TC_SUCCESS) {
            fprintf(stderr, "FAIL: solid encode\n");
            return 1;
        }
        memcpy(sb, base, 16u);
        tc_bc7_ref_decode_block(sb, sdec);
        /* BC7 endpoint quantization can round a solid colour by ~1 LSB, so use
         * a small tolerance (the decoder itself is bit-exact vs bcdec). */
        for (t = 0; t < 16u; ++t)
            if (abs((int)sdec[t][0] - 173) > 2 || abs((int)sdec[t][1] - 66) > 2 ||
                abs((int)sdec[t][2] - 214) > 2 || sdec[t][3] != 255u) {
                fprintf(stderr, "FAIL: solid block decoded off (texel %u = "
                                "%u,%u,%u,%u)\n",
                        t, sdec[t][0], sdec[t][1], sdec[t][2], sdec[t][3]);
                return 1;
            }

        /* rebuild the gradient streams (base was overwritten above). */
        tc_bc7_options_init(&so);
        so.rdo = 0;
        tc_bc7_compress_rgba8(img, W, H, W * 4u, &so, base, need);
        so.rdo = 8;
        tc_bc7_compress_rgba8(img, W, H, W * 4u, &so, rdo, need);
        /* The lib's own decoder (tc_bc7_decompress_rgba8, used by the
         * decoder-driven RDO) must agree bit-exactly with this independent
         * reference decoder on every pixel. */
        {
            static uint8_t libdec[W * H * 4];
            uint32_t bx, by, xx, yy, bxc = (W + 3u) / 4u;
            tc_bc7_decompress_rgba8(base, W, H, W * 4u, libdec, sizeof(libdec));
            for (by = 0; by < H; by += 4u)
                for (bx = 0; bx < W; bx += 4u) {
                    uint8_t rd[16][4];
                    tc_bc7_ref_decode_block(
                        base + ((size_t)(by / 4u) * bxc + bx / 4u) * 16u, rd);
                    for (yy = 0; yy < 4u; ++yy)
                        for (xx = 0; xx < 4u; ++xx) {
                            uint32_t x = bx + xx, y = by + yy;
                            if (x >= W || y >= H) continue;
                            if (memcmp(libdec + ((size_t)y * W + x) * 4u,
                                       rd[yy * 4u + xx], 4u) != 0) {
                                fprintf(stderr, "FAIL: lib decoder != reference "
                                                "at (%u,%u)\n", x, y);
                                return 1;
                            }
                        }
                }
        }

        pbase = bc7_psnr(img, W, H, base);
        prdo = bc7_psnr(img, W, H, rdo);
        printf("xbc7 gate PSNR: bc7=%.2f dB  rdo=8=%.2f dB (lib==ref decoder)\n",
               pbase, prdo);
        if (pbase < 30.0) {
            fprintf(stderr, "FAIL: base BC7 PSNR too low (%.2f)\n", pbase);
            return 1;
        }
        if (prdo > pbase + 0.1) {
            fprintf(stderr, "FAIL: rdo raised PSNR (%.2f > %.2f)\n", prdo, pbase);
            return 1;
        }
    }

    free(base);
    free(rdo);
    free(z0);
    free(z1);
    printf("xbc7 gate: OK\n");
    return 0;
}
