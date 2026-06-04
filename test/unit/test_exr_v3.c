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
#include "exr_internal.h" /* exr_fpnge_deflate / exr_inflate_zlib (internal) */

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

/* Build a tiled image from a scanline source in-memory, save tiled, reload. */
static void tiled_roundtrip(const char *path) {
    exr_image src, back;
    void *buf = NULL;
    size_t sz = 0;
    exr_result rc;
    memset(&src, 0, sizeof(src));
    memset(&back, 0, sizeof(back));
    if (!EXR_OK(exr_load_from_file(path, NULL, &src))) {
        g_fail++;
        printf("  FAIL: %s load (tiled rt)\n", path);
        return;
    }
    src.parts[0].header.tiled = 1;
    src.parts[0].header.part_type = EXR_PART_TILED;
    src.parts[0].header.tile_x_size = 64;
    src.parts[0].header.tile_y_size = 64;
    src.parts[0].header.level_mode = EXR_TILE_ONE_LEVEL;
    src.parts[0].header.rounding_mode = EXR_TILE_ROUND_DOWN;
    rc = exr_save_to_memory(&buf, &sz, NULL, &src, EXR_COMPRESSION_ZIP);
    if (!EXR_OK(rc)) {
        g_fail++;
        printf("  FAIL: tiled save: %s\n", exr_result_string(rc));
        exr_image_free(&src);
        return;
    }
    rc = exr_load_from_memory(buf, sz, NULL, &back);
    if (!EXR_OK(rc)) {
        g_fail++;
        printf("  FAIL: tiled reload: %s\n", exr_result_string(rc));
    } else {
        CHECK(back.parts[0].header.tiled, "tiled flag preserved");
        CHECK(images_equal(&src, &back), "tiled roundtrip pixels");
        printf("  ok: tiled roundtrip (%s, ZIP, %zu bytes)\n", path, sz);
    }
    free(buf);
    exr_image_free(&src);
    exr_image_free(&back);
}

/* fpnge PSHUFB literal encoder: SIMD output must equal scalar, and both must
 * round-trip through the inflater. */
static void fpnge_check(void) {
    const exr_allocator *a = exr_default_allocator();
    static const size_t sizes[] = {0, 1, 16, 17, 1000, 50000};
    size_t s;
    for (s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        size_t n = sizes[s], i;
        uint8_t *src = (uint8_t *)malloc(n ? n : 1);
        uint8_t *c0 = NULL, *c1 = NULL, *dec = NULL;
        size_t l0 = 0, l1 = 0, got = 0;
        unsigned rng = 1234567u;
        for (i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            src[i] = (uint8_t)((i % 40 < 6) ? (rng >> 9) : (i / 11));
        }
        if (EXR_OK(exr_fpnge_deflate(a, src, n, &c0, &l0, 0)) &&
            EXR_OK(exr_fpnge_deflate(a, src, n, &c1, &l1, 1))) {
            CHECK(l0 == l1 && (l0 == 0 || memcmp(c0, c1, l0) == 0),
                  "fpnge simd==scalar");
            dec = (uint8_t *)malloc(n ? n : 1);
            CHECK(EXR_OK(exr_inflate_zlib(c1, l1, dec, n, &got)) && got == n &&
                      (n == 0 || memcmp(src, dec, n) == 0),
                  "fpnge round-trip");
        } else {
            g_fail++;
        }
        free(src);
        free(c0);
        free(c1);
        free(dec);
    }
    printf("  ok: fpnge PSHUFB encoder (scalar==simd, round-trip)\n");

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    /* SSE4.1 and AVX2 lookup kernels must match the scalar reference exactly. */
    {
        size_t n = 100000, i;
        uint8_t *buf = (uint8_t *)malloc(n);
        uint8_t *ns = (uint8_t *)malloc(n), *ls = (uint8_t *)malloc(n),
                *hs = (uint8_t *)malloc(n);
        uint8_t *nx = (uint8_t *)malloc(n), *lx = (uint8_t *)malloc(n),
                *hx = (uint8_t *)malloc(n);
        uint64_t freqs[286];
        exr_fpnge_table tbl;
        unsigned rng = 42;
        uint32_t caps = exr_simd_capabilities();
        memset(freqs, 0, sizeof(freqs));
        for (i = 0; i < n; ++i) {
            rng = rng * 1664525u + 1013904223u;
            buf[i] = (uint8_t)(rng >> 8);
            freqs[buf[i]]++;
        }
        if (exr_fpnge_build_table(exr_default_allocator(), freqs, &tbl)) {
            exr_fpnge_lookup_scalar(&tbl, buf, n, ns, ls, hs);
            if (caps & EXR_SIMD_SSE41) {
                exr_fpnge_lookup_sse41(&tbl, buf, n, nx, lx, hx);
                CHECK(!memcmp(ns, nx, n) && !memcmp(ls, lx, n) && !memcmp(hs, hx, n),
                      "fpnge sse4.1 lookup == scalar");
            }
            if (caps & EXR_SIMD_AVX2) {
                exr_fpnge_lookup_avx2(&tbl, buf, n, nx, lx, hx);
                CHECK(!memcmp(ns, nx, n) && !memcmp(ls, lx, n) && !memcmp(hs, hx, n),
                      "fpnge avx2 lookup == scalar");
            }
            printf("  ok: fpnge lookup kernels match scalar\n");
        }
        free(buf); free(ns); free(ls); free(hs); free(nx); free(lx); free(hx);
    }
#endif
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
    roundtrip("asakusa.exr", EXR_COMPRESSION_PIZ, "PIZ");
    roundtrip("test/unit/regression/000-issue194.exr", EXR_COMPRESSION_PIZ, "PIZ-3ch");
    tiled_roundtrip("asakusa.exr");

    printf("== fpnge PSHUFB Huffman-emit ==\n");
    fpnge_check();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
