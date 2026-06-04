/*
 * TinyEXR v3 (pure C11) reader unit test.
 *
 * Run from the repository root (paths are relative to it):
 *     make test-c
 *
 * Validates the public load API against the bundled corpus and confirms that
 * malformed / fuzzer files fail cleanly without crashing (run under ASan).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            g_pass++;                                                          \
        } else {                                                               \
            g_fail++;                                                          \
            printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
        }                                                                      \
    } while (0)

/* Load a file expected to succeed; check dims + channel count of part 0. */
static void expect_load(const char *path, int parts, int w, int h, int nch) {
    exr_image img;
    exr_result rc;
    memset(&img, 0, sizeof(img));
    rc = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(rc)) {
        g_fail++;
        printf("  FAIL: %s did not load: %s\n", path, exr_result_string(rc));
        return;
    }
    CHECK(img.num_parts == parts, path);
    if (img.num_parts > 0) {
        CHECK(img.parts[0].width == w, path);
        CHECK(img.parts[0].height == h, path);
        CHECK(img.parts[0].header.num_channels == nch, path);
        CHECK(img.parts[0].images != NULL, path);
    }
    exr_image_free(&img);
    printf("  ok: %s (%dx%d, %d ch)\n", path, w, h, nch);
}

/* Load a file expected to fail; the subsequent free must be safe. */
static void expect_fail(const char *path) {
    exr_image img;
    exr_result rc;
    memset(&img, 0, sizeof(img));
    rc = exr_load_from_file(path, NULL, &img);
    CHECK(!EXR_OK(rc) || rc == EXR_WOULD_BLOCK ? !EXR_OK(rc) : 0, path);
    exr_image_free(&img); /* must not crash / double-free */
    if (!EXR_OK(rc)) printf("  ok (rejected): %s -> %s\n", path,
                            exr_result_string(rc));
}

/* Run a file purely for crash-safety (any result accepted). */
static void run_safe(const char *path) {
    exr_image img;
    memset(&img, 0, sizeof(img));
    (void)exr_load_from_file(path, NULL, &img);
    exr_image_free(&img);
    g_pass++;
}

static int images_equal(const exr_image *a, const exr_image *b) {
    int p, c;
    if (a->num_parts != b->num_parts) return 0;
    for (p = 0; p < a->num_parts; ++p) {
        const exr_part *pa = &a->parts[p], *pb = &b->parts[p];
        if (pa->width != pb->width || pa->height != pb->height) return 0;
        if (pa->header.num_channels != pb->header.num_channels) return 0;
        for (c = 0; c < pa->header.num_channels; ++c) {
            size_t ps =
                pa->header.channels[c].pixel_type == EXR_PIXEL_HALF ? 2 : 4;
            if (memcmp(pa->images[c], pb->images[c],
                       (size_t)pa->width * pa->height * ps) != 0)
                return 0;
        }
    }
    return 1;
}

/* Load -> save (codec) -> reload -> must equal the original. */
static void roundtrip(const char *path, exr_compression comp, const char *name) {
    exr_image src, back;
    void *buf = NULL;
    size_t sz = 0;
    exr_result rc;
    memset(&src, 0, sizeof(src));
    memset(&back, 0, sizeof(back));
    if (!EXR_OK(exr_load_from_file(path, NULL, &src))) {
        g_fail++;
        printf("  FAIL: %s load (roundtrip %s)\n", path, name);
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &src, comp);
    if (!EXR_OK(rc)) {
        g_fail++;
        printf("  FAIL: save %s: %s\n", name, exr_result_string(rc));
        exr_image_free(&src);
        return;
    }
    rc = exr_load_from_memory(buf, sz, NULL, &back);
    if (!EXR_OK(rc)) {
        g_fail++;
        printf("  FAIL: reload %s: %s\n", name, exr_result_string(rc));
    } else {
        CHECK(images_equal(&src, &back), name);
        printf("  ok: roundtrip %s (%s, %zu bytes)\n", name, path, sz);
    }
    free(buf);
    exr_image_free(&src);
    exr_image_free(&back);
}

int main(void) {
    static const char *poc[] = {
        "test/unit/regression/poc-1383755b301e5f505b2198dc0508918b537fdf48bbfc6deeffe268822e6f6cd6",
        "test/unit/regression/poc-255456016cca60ddb5c5ed6898182e13739bf687b17d1411e97bb60ad95e7a84_min",
        "test/unit/regression/poc-360c3b0555cb979ca108f2d178cf8a80959cfeabaa4ec1d310d062fa653a8c6b_min",
        "test/unit/regression/poc-e7fa6404daa861369d2172fe68e08f9d38c0989f57da7bcfb510bab67e19ca9f",
    };
    size_t i;

    printf("== valid corpus ==\n");
    expect_load("asakusa.exr", 1, 660, 440, 4);
    expect_load("test/unit/regression/2by2.exr", 1, 2, 2, 4);
    expect_load("test/unit/regression/flaga.exr", 1, 128, 64, 8);
    expect_load("test/unit/regression/000-issue194.exr", 1, 1024, 1024, 3);
    expect_load("test/unit/regression/issue-160-piz-decode.exr", 1, 420, 32, 3);
    expect_load("test/unit/regression/piz-bug-issue-100.exr", 1, 35, 1, 1);
    expect_load("test/unit/regression/tiled_half_1x1_alpha.exr", 1, 1, 1, 1);

    printf("== malformed (must reject + free safely) ==\n");
    expect_fail("test/unit/regression/issue-238-double-free.exr");
    expect_fail("test/unit/regression/issue-238-double-free-multipart.exr");

    printf("== fuzzer PoCs (must not crash) ==\n");
    for (i = 0; i < sizeof(poc) / sizeof(poc[0]); ++i) run_safe(poc[i]);

    printf("== writer round-trip (load -> save -> reload) ==\n");
    roundtrip("asakusa.exr", EXR_COMPRESSION_NONE, "NONE");
    roundtrip("asakusa.exr", EXR_COMPRESSION_RLE, "RLE");
    roundtrip("asakusa.exr", EXR_COMPRESSION_ZIPS, "ZIPS");
    roundtrip("asakusa.exr", EXR_COMPRESSION_ZIP, "ZIP");
    roundtrip("test/unit/regression/flaga.exr", EXR_COMPRESSION_ZIP, "ZIP-8ch");

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
