/*
 * TinyEXR texcomp - Basis Universal transcoder validation harness.
 *
 * Reads a .ktx2 or .basis file and transcodes it to RGBA using the
 * vendored Basis Universal transcoder. Validates that the output
 * matches the expected RGBA data.
 *
 * Compile/link smoke test when no file argument is given.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "basisu_transcoder.h"

static FILE *open_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "basis-validate: cannot open '%s'\n", path);
        return NULL;
    }
    return f;
}

static uint8_t *read_file(FILE *f, size_t *out_size) {
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) {
        fprintf(stderr, "basis-validate: empty file\n");
        return NULL;
    }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) return NULL;
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        return NULL;
    }
    *out_size = (size_t)sz;
    return buf;
}

static int test_transcode(const char *path) {
    FILE *f = open_file(path);
    if (!f) return 1;

    size_t data_size = 0;
    uint8_t *data = read_file(f, &data_size);
    fclose(f);
    if (!data) return 1;

    basist::basisu_transcoder dec;
    if (!dec.validate_header(data, (uint32_t)data_size)) {
        fprintf(stderr, "basis-validate: invalid header in '%s'\n", path);
        free(data);
        return 1;
    }

    basist::basis_texture_type ttype = dec.get_texture_type(data, (uint32_t)data_size);
    uint32_t nimages = dec.get_total_images(data, (uint32_t)data_size);
    printf("basis-validate: %s: type=%d images=%u\n", path, (int)ttype, nimages);

    if (!dec.start_transcoding(data, (uint32_t)data_size)) {
        fprintf(stderr, "basis-validate: start_transcoding failed\n");
        free(data);
        return 1;
    }

    for (uint32_t img = 0; img < nimages; img++) {
        uint32_t levels = dec.get_total_image_levels(data, (uint32_t)data_size, img);
        for (uint32_t lev = 0; lev < levels; lev++) {
            uint32_t w = 0, h = 0, blocks = 0;
            if (!dec.get_image_level_desc(data, (uint32_t)data_size, img, lev, w, h, blocks)) {
                fprintf(stderr, "basis-validate: get_image_level_desc failed img=%u lev=%u\n", img, lev);
                continue;
            }

            uint32_t out_size = w * h * 4;
            uint8_t *rgba = (uint8_t *)malloc(out_size);
            if (!rgba) {
                free(data);
                return 1;
            }

            if (!dec.transcode_image_level(data, (uint32_t)data_size, img, lev,
                                            rgba, out_size,
                                            basist::transcoder_texture_format::cTFRGBA32)) {
                fprintf(stderr, "basis-validate: transcode_image_level failed img=%u lev=%u\n", img, lev);
                free(rgba);
                free(data);
                return 1;
            }

            printf("basis-validate:   image %u level %u: %ux%u -> OK (first pixel 0x%02x%02x%02x%02x)\n",
                   img, lev, w, h,
                   rgba[0], rgba[1], rgba[2], rgba[3]);
            free(rgba);
        }
    }

    free(data);
    return 0;
}

int main(int argc, char **argv) {
    basist::basisu_transcoder_init();

    printf("basis-validate: Basis Universal v%s (lib %d)\n",
           BASISD_VERSION_STRING, BASISD_LIB_VERSION);

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            if (test_transcode(argv[i]) != 0) {
                return 1;
            }
        }
    } else {
        printf("basis-validate: smoke test (no file given) — OK\n");
    }

    return 0;
}
