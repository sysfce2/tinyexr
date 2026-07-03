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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    free(base);
    free(rdo);
    free(z0);
    free(z1);
    printf("xbc7 gate: OK\n");
    return 0;
}
