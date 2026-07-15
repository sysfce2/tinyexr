/*
 * TinyEXR texpipe - Three.js KTX2Loader interoperability fixture generator.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "texcomp.h"
#include "texpipe.h"
#include "tinyexr_zstd.h"

static size_t zstd_bound(void *user, size_t src_size) {
    (void)user;
    return tinyexr_zstd_compress_bound(src_size);
}

static size_t zstd_compress(void *user, uint8_t *dst, size_t dst_cap,
                            const uint8_t *src, size_t src_size) {
    size_t result;
    (void)user;
    result = tinyexr_zstd_compress(dst, dst_cap, src, src_size, 3);
    return tinyexr_zstd_is_error(result) ? 0u : result;
}

int main(int argc, char **argv) {
    enum { WIDTH = 8, HEIGHT = 8 };
    uint8_t rgba[WIDTH * HEIGHT * 4];
    size_t uni_size = tc_uni_compressed_size(WIDTH, HEIGHT);
    uint8_t *uni = NULL;
    const uint8_t *levels[1];
    size_t level_sizes[1];
    uint8_t *ktx2 = NULL;
    size_t ktx2_size = 0;
    FILE *file = NULL;
    int x, y;
    int status = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s output.ktx2\n", argv[0]);
        return 2;
    }

    uni = (uint8_t *)malloc(uni_size);
    if (!uni) goto cleanup;
    for (y = 0; y < HEIGHT; ++y) {
        for (x = 0; x < WIDTH; ++x) {
            size_t offset = ((size_t)y * WIDTH + (size_t)x) * 4u;
            rgba[offset + 0] = (uint8_t)(x * 255 / (WIDTH - 1));
            rgba[offset + 1] = (uint8_t)(y * 255 / (HEIGHT - 1));
            rgba[offset + 2] = 64u;
            rgba[offset + 3] =
                (uint8_t)((x + y) * 255 / (WIDTH + HEIGHT - 2));
        }
    }
    if (tc_uni_compress_rgba8(rgba, WIDTH, HEIGHT, WIDTH * 4u, uni,
                              uni_size) != TC_SUCCESS)
        goto cleanup;

    levels[0] = uni;
    level_sizes[0] = uni_size;
    if (tp_ktx2_write_uni_zstd(NULL, zstd_bound, zstd_compress, NULL, levels,
                               level_sizes, WIDTH, HEIGHT, 1,
                               TP_UNI_SRGB | TP_UNI_ALPHA |
                                   TP_UNI_ASTC_KTX2,
                               &ktx2,
                               &ktx2_size) != TP_SUCCESS)
        goto cleanup;

    file = fopen(argv[1], "wb");
    if (!file) goto cleanup;
    if (fwrite(ktx2, 1, ktx2_size, file) != ktx2_size) {
        fclose(file);
        file = NULL;
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto cleanup;
    }
    file = NULL;
    printf("three-ktx2: wrote %zu-byte 8x8 scheme-2 ASTC fixture\n",
           ktx2_size);
    status = 0;

cleanup:
    if (file) fclose(file);
    tp_free(NULL, ktx2);
    free(uni);
    return status;
}
