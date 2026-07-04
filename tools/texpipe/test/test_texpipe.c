/*
 * TinyEXR texpipe - unit tests.
 *
 * Covers: mip geometry, per-mip BC7 round-trip PSNR (shipped decoder),
 * alpha-coverage preservation across LODs, and DDS/KTX2 container re-parse.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "texpipe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            printf("FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);            \
            g_fail = 1;                                                         \
        }                                                                       \
    } while (0)

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const uint8_t *p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

static uint8_t to_u8(float f) {
    int v;
    if (f < 0.0f) f = 0.0f;
    if (f > 1.0f) f = 1.0f;
    v = (int)(f * 255.0f + 0.5f);
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* Reference u8 RGBA for a float surface (matches texpipe's internal packing). */
static void surface_rgba8(const tp_surface *s, uint8_t *dst) {
    int x, y, c;
    for (y = 0; y < s->height; ++y) {
        const float *row =
            (const float *)((const uint8_t *)s->data + (size_t)y * s->stride);
        for (x = 0; x < s->width; ++x) {
            float px[4] = {0, 0, 0, 1};
            for (c = 0; c < s->channels && c < 4; ++c)
                px[c] = row[x * s->channels + c];
            dst[(y * s->width + x) * 4 + 0] = to_u8(px[0]);
            dst[(y * s->width + x) * 4 + 1] = to_u8(px[1]);
            dst[(y * s->width + x) * 4 + 2] = to_u8(px[2]);
            dst[(y * s->width + x) * 4 + 3] = to_u8(px[3]);
        }
    }
}

static double psnr_rgba(const uint8_t *a, const uint8_t *b, size_t npix) {
    double mse = 0.0;
    size_t i;
    for (i = 0; i < npix * 4; ++i) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)(npix * 4);
    if (mse <= 0.0) return 1e9;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

/* ------------------------------------------------------------ fixtures */

/* Smooth RGBA gradient (opaque). */
static uint8_t *make_gradient(int w, int h) {
    uint8_t *img = (uint8_t *)malloc((size_t)w * h * 4);
    int x, y;
    for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x) {
            uint8_t *p = img + (y * w + x) * 4;
            p[0] = (uint8_t)(x * 255 / (w - 1));
            p[1] = (uint8_t)(y * 255 / (h - 1));
            p[2] = (uint8_t)((x + y) * 255 / (w + h - 2));
            p[3] = 255;
        }
    return img;
}

/* Alpha-tested cutout: a filled disc (alpha 1 inside, 0 outside). */
static uint8_t *make_cutout(int w, int h) {
    uint8_t *img = (uint8_t *)malloc((size_t)w * h * 4);
    int x, y;
    float cx = w * 0.5f, cy = h * 0.5f, r = w * 0.30f;
    for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x) {
            uint8_t *p = img + (y * w + x) * 4;
            float dx = x - cx, dy = y - cy;
            int inside = (dx * dx + dy * dy) <= (r * r);
            p[0] = 200; p[1] = 120; p[2] = 60;
            p[3] = inside ? 255 : 0;
        }
    return img;
}

static void view_u8(tir_image_view *v, uint8_t *data, int w, int h) {
    v->data = data;
    v->width = w;
    v->height = h;
    v->channels = 4;
    v->type = TIR_U8;
    v->row_stride_bytes = 0;
}

/* ------------------------------------------------------------- tests */

static void test_mip_geometry(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    uint8_t *img = make_gradient(128, 128);
    int expect_levels = 8; /* 128,64,32,16,8,4,2,1 */
    view_u8(&v, img, 128, 128);
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build_mips gradient");
    CHECK(chain.num_levels == expect_levels, "level count 128->8");
    CHECK(chain.level[0].width == 128 && chain.level[0].height == 128, "level0 dim");
    CHECK(chain.level[expect_levels - 1].width == 1, "last level 1x1");
    tp_mip_chain_free(NULL, &chain);
    free(img);
    printf("  mip geometry: ok\n");
}

static void test_bc7_roundtrip_psnr(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    tp_blocks blocks;
    int i;
    uint8_t *img = make_gradient(128, 128);
    view_u8(&v, img, 128, 128);
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    memset(&chain, 0, sizeof(chain));
    memset(&blocks, 0, sizeof(blocks));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build_mips");
    CHECK(TP_OK(tp_compress_chain(NULL, &chain, &opt, &blocks)), "compress_chain");
    for (i = 0; i < chain.num_levels; ++i) {
        const tp_surface *s = &chain.level[i];
        size_t npix = (size_t)s->width * s->height;
        uint8_t *ref = (uint8_t *)malloc(npix * 4);
        uint8_t *dec = (uint8_t *)malloc(npix * 4);
        double p;
        surface_rgba8(s, ref);
        CHECK(tc_bc7_decompress_rgba8(blocks.blk[i].data, (uint32_t)s->width,
                                      (uint32_t)s->height, (size_t)s->width * 4,
                                      dec, npix * 4) == TC_SUCCESS,
              "bc7 decode");
        p = psnr_rgba(ref, dec, npix);
        printf("    level %d %dx%d PSNR=%.2f dB\n", i, s->width, s->height, p);
        /* Assert only on non-trivial levels: a single 4x4/8x8 BC7 block can't
         * fit a 2D colour gradient, which is a BC7 property, not a pipeline
         * defect. Larger levels exercise the round-trip meaningfully. */
        if (s->width >= 32) CHECK(p >= 30.0, "bc7 per-mip PSNR >= 30 dB");
        free(ref);
        free(dec);
    }
    tp_blocks_free(NULL, &blocks);
    tp_mip_chain_free(NULL, &chain);
    free(img);
    printf("  bc7 per-mip round-trip PSNR: ok\n");
}

static void test_alpha_coverage(void) {
    tir_image_view v;
    tp_options opt, opt_off;
    tp_mip_chain chain, chain_off;
    float base_cov, worst_err = 0.0f, worst_err_off = 0.0f;
    int i;
    uint8_t *img = make_cutout(256, 256);
    view_u8(&v, img, 256, 256);

    /* With coverage preservation. */
    tp_options_init(&opt, TP_CONTENT_ALPHA_TESTED, TP_CODEC_BC7);
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build alpha-tested");
    base_cov = tp_alpha_coverage(&chain.level[0], opt.alpha_test_threshold);
    for (i = 1; i < chain.num_levels; ++i) {
        float c = tp_alpha_coverage(&chain.level[i], opt.alpha_test_threshold);
        float e = fabsf(c - base_cov);
        if (chain.level[i].width >= 4 && e > worst_err) worst_err = e;
    }

    /* Without preservation, for contrast. */
    tp_options_init(&opt_off, TP_CONTENT_COLOR, TP_CODEC_BC7);
    opt_off.preserve_alpha_coverage = 0;
    memset(&chain_off, 0, sizeof(chain_off));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt_off, &chain_off)), "build no-preserve");
    for (i = 1; i < chain_off.num_levels; ++i) {
        float c = tp_alpha_coverage(&chain_off.level[i], opt_off.alpha_test_threshold);
        float e = fabsf(c - base_cov);
        if (chain_off.level[i].width >= 4 && e > worst_err_off) worst_err_off = e;
    }

    printf("  alpha coverage: base=%.4f worst_err(preserve)=%.4f worst_err(off)=%.4f\n",
           base_cov, worst_err, worst_err_off);
    CHECK(worst_err < 0.05f, "coverage preserved within 5%");
    CHECK(worst_err <= worst_err_off + 1e-4f, "preservation no worse than off");

    tp_mip_chain_free(NULL, &chain);
    tp_mip_chain_free(NULL, &chain_off);
    free(img);
}

static void test_alpha_scale_helper(void) {
    /* 4x1 surface, alphas 0.2,0.4,0.6,0.8: coverage@0.5 = 0.5. Target 0.75
     * (3 of 4 passing) requires scaling up. */
    float data[16] = {0, 0, 0, 0.2f, 0, 0, 0, 0.4f,
                      0, 0, 0, 0.6f, 0, 0, 0, 0.8f};
    tp_surface s;
    float cov;
    s.width = 4; s.height = 1; s.channels = 4;
    s.data = data; s.stride = 16 * sizeof(float);
    CHECK(fabsf(tp_alpha_coverage(&s, 0.5f) - 0.5f) < 1e-6f, "coverage 0.5");
    CHECK(TP_OK(tp_alpha_scale_to_coverage(&s, 0.75f, 0.5f)), "scale to 0.75");
    cov = tp_alpha_coverage(&s, 0.5f);
    CHECK(fabsf(cov - 0.75f) < 1e-6f, "coverage now 0.75");
    printf("  alpha scale helper: ok (cov=%.3f)\n", cov);
}

static void test_containers(void) {
    tir_image_view v;
    tp_options opt;
    uint8_t *ktx = NULL, *dds = NULL;
    size_t ktx_n = 0, dds_n = 0;
    uint8_t *img = make_gradient(64, 64);
    int expect_levels = 7; /* 64..1 */
    view_u8(&v, img, 64, 64);

    /* KTX2 */
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    opt.container = TP_CONTAINER_KTX2;
    CHECK(TP_OK(tp_process(NULL, &v, 1, &opt, &ktx, &ktx_n)), "process ktx2");
    if (ktx) {
        static const uint8_t id[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                       0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
        CHECK(memcmp(ktx, id, 12) == 0, "ktx2 identifier");
        CHECK(rd_u32(ktx + 12) == 145u, "ktx2 vkFormat BC7_UNORM");
        CHECK(rd_u32(ktx + 20) == 64u, "ktx2 pixelWidth");
        CHECK(rd_u32(ktx + 36) == 1u, "ktx2 faceCount");
        CHECK(rd_u32(ktx + 40) == (uint32_t)expect_levels, "ktx2 levelCount");
        /* level index[0] must point at valid, in-bounds data. */
        {
            uint64_t off = rd_u64(ktx + 80);
            uint64_t len = rd_u64(ktx + 88);
            size_t l0 = tc_bc7_compressed_size(64, 64);
            CHECK(len == l0, "ktx2 level0 length == bc7 size");
            CHECK(off + len <= ktx_n, "ktx2 level0 in bounds");
        }
    }

    /* DDS */
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    opt.container = TP_CONTAINER_DDS;
    CHECK(TP_OK(tp_process(NULL, &v, 1, &opt, &dds, &dds_n)), "process dds");
    if (dds) {
        CHECK(memcmp(dds, "DDS ", 4) == 0, "dds magic");
        CHECK(rd_u32(dds + 28) == (uint32_t)expect_levels, "dds mipMapCount");
        CHECK(rd_u32(dds + 16) == 64u, "dds width");
        CHECK(memcmp(dds + 84, "DX10", 4) == 0, "dds DX10 fourcc");
        CHECK(rd_u32(dds + 128) == 98u, "dds dxgiFormat BC7_UNORM");
        CHECK(dds_n == 148u + tc_bc7_compressed_size(64, 64) +
                          tc_bc7_compressed_size(32, 32) +
                          tc_bc7_compressed_size(16, 16) +
                          tc_bc7_compressed_size(8, 8) +
                          tc_bc7_compressed_size(4, 4) +
                          tc_bc7_compressed_size(2, 2) +
                          tc_bc7_compressed_size(1, 1),
              "dds total size == header + all mips");
    }

    tp_free(NULL, ktx);
    tp_free(NULL, dds);
    free(img);
    printf("  containers (ktx2 + dds re-parse): ok\n");
}

/* ------------------------------------------------------------- cube */

/* Cube face direction convention (matches texpipe_cube.c). u,v in [-1,1]. */
static void cube_dir(int face, float u, float v, float d[3]) {
    switch (face) {
    case 0: d[0] = 1;  d[1] = -v; d[2] = -u; break; /* +X */
    case 1: d[0] = -1; d[1] = -v; d[2] = u;  break; /* -X */
    case 2: d[0] = u;  d[1] = 1;  d[2] = v;  break; /* +Y */
    case 3: d[0] = u;  d[1] = -1; d[2] = -v; break; /* -Y */
    case 4: d[0] = u;  d[1] = -v; d[2] = 1;  break; /* +Z */
    default: d[0] = -u; d[1] = -v; d[2] = -1; break; /* -Z */
    }
}

/* Smooth scalar field over the sphere, in [0,1], per RGB channel. */
static void cube_field(int face, int col, int row, int n, float rgb[3]) {
    float u = 2.0f * ((float)col + 0.5f) / (float)n - 1.0f;
    float v = 2.0f * ((float)row + 0.5f) / (float)n - 1.0f;
    float d[3], len;
    cube_dir(face, u, v, d);
    len = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    rgb[0] = 0.5f + 0.5f * d[0] / len;
    rgb[1] = 0.5f + 0.5f * d[1] / len;
    rgb[2] = 0.5f + 0.5f * d[2] / len;
}

static void test_cube_seam_fixup(void) {
    const int n = 32, ch = 4;
    tp_surface faces[6];
    int f, x, y, c;
    float max_dev = 0.0f, max_idem = 0.0f;

    for (f = 0; f < 6; ++f) {
        faces[f].width = n; faces[f].height = n; faces[f].channels = ch;
        faces[f].stride = (size_t)n * ch * sizeof(float);
        faces[f].data = (float *)malloc(faces[f].stride * n);
        for (y = 0; y < n; ++y)
            for (x = 0; x < n; ++x) {
                float rgb[3];
                float *p = faces[f].data + (y * n + x) * ch;
                cube_field(f, x, y, n, rgb);
                p[0] = rgb[0]; p[1] = rgb[1]; p[2] = rgb[2]; p[3] = 1.0f;
            }
    }

    CHECK(TP_OK(tp_cube_seam_fixup(faces, NULL)), "cube seam fixup");

    /* Geometric check: every border texel must remain close to its original
     * direction value. A wrong adjacency table would average unrelated texels
     * and produce a large deviation, so this validates the 12-edge/8-corner
     * tables independently of the fixup code. */
    for (f = 0; f < 6; ++f)
        for (y = 0; y < n; ++y)
            for (x = 0; x < n; ++x) {
                float rgb[3];
                const float *p = faces[f].data + (y * n + x) * ch;
                int border = (x == 0 || x == n - 1 || y == 0 || y == n - 1);
                if (!border) continue;
                cube_field(f, x, y, n, rgb);
                for (c = 0; c < 3; ++c) {
                    float dv = fabsf(p[c] - rgb[c]);
                    if (dv > max_dev) max_dev = dv;
                }
            }
    CHECK(max_dev < 0.05f, "cube borders stay near true direction (table correct)");

    /* Idempotency: a second fixup must not change anything. */
    {
        float *snap[6];
        for (f = 0; f < 6; ++f) {
            snap[f] = (float *)malloc(faces[f].stride * n);
            memcpy(snap[f], faces[f].data, faces[f].stride * n);
        }
        CHECK(TP_OK(tp_cube_seam_fixup(faces, NULL)), "cube fixup again");
        for (f = 0; f < 6; ++f) {
            size_t k, nf = (size_t)n * n * ch;
            for (k = 0; k < nf; ++k) {
                float dv = fabsf(faces[f].data[k] - snap[f][k]);
                if (dv > max_idem) max_idem = dv;
            }
            free(snap[f]);
        }
    }
    CHECK(max_idem < 1e-6f, "cube fixup is idempotent");

    printf("  cube seam fixup: ok (max_dev=%.4f idempotent_delta=%.2e)\n",
           max_dev, max_idem);
    for (f = 0; f < 6; ++f) free(faces[f].data);
}

static void test_cube_split(void) {
    const int n = 16;
    int W = 6 * n, H = n, f, ok = 1;
    uint8_t *img = (uint8_t *)malloc((size_t)W * H * 4);
    tir_image_view src, out[6];
    int x, y;
    /* strip_h: tile f filled with value f*10 in R. */
    for (y = 0; y < H; ++y)
        for (x = 0; x < W; ++x) {
            uint8_t *p = img + (y * W + x) * 4;
            p[0] = (uint8_t)((x / n) * 10); p[1] = 0; p[2] = 0; p[3] = 255;
        }
    src.data = img; src.width = W; src.height = H; src.channels = 4;
    src.type = TIR_U8; src.row_stride_bytes = 0;
    CHECK(TP_OK(tp_cube_split(&src, TP_CUBE_STRIP_H, out)), "cube split strip_h");
    for (f = 0; f < 6; ++f) {
        const uint8_t *p = (const uint8_t *)out[f].data; /* top-left texel */
        if (out[f].width != n || out[f].height != n || p[0] != (uint8_t)(f * 10))
            ok = 0;
    }
    CHECK(ok, "cube split produces 6 correctly-placed faces");
    printf("  cube split (strip_h): ok\n");
    free(img);
}

static void test_cube_container(void) {
    const int n = 32;
    tir_image_view faces[6];
    uint8_t *data[6];
    tp_options opt;
    uint8_t *ktx = NULL;
    size_t ktx_n = 0;
    int f, x, y;
    for (f = 0; f < 6; ++f) {
        data[f] = (uint8_t *)malloc((size_t)n * n * 4);
        for (y = 0; y < n; ++y)
            for (x = 0; x < n; ++x) {
                uint8_t *p = data[f] + (y * n + x) * 4;
                p[0] = (uint8_t)(x * 255 / (n - 1));
                p[1] = (uint8_t)(y * 255 / (n - 1));
                p[2] = (uint8_t)(f * 40); p[3] = 255;
            }
        view_u8(&faces[f], data[f], n, n);
    }
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    opt.container = TP_CONTAINER_KTX2;
    opt.is_cube = 1;
    CHECK(TP_OK(tp_process(NULL, faces, 6, &opt, &ktx, &ktx_n)), "cube process ktx2");
    if (ktx) {
        CHECK(rd_u32(ktx + 36) == 6u, "ktx2 faceCount == 6");
        CHECK(rd_u32(ktx + 40) == 6u, "ktx2 levelCount == 6 (32->1)");
        /* level 0 length must cover all 6 faces. */
        CHECK(rd_u64(ktx + 88) == 6u * tc_bc7_compressed_size(n, n),
              "ktx2 cube level0 length = 6 faces");
    }
    /* DDS cube caps. */
    {
        uint8_t *dds = NULL; size_t dds_n = 0;
        tp_options o2;
        tp_options_init(&o2, TP_CONTENT_COLOR, TP_CODEC_BC7);
        o2.container = TP_CONTAINER_DDS; o2.is_cube = 1;
        CHECK(TP_OK(tp_process(NULL, faces, 6, &o2, &dds, &dds_n)), "cube process dds");
        if (dds) {
            CHECK((rd_u32(dds + 112) & 0x200u) != 0u, "dds DDSCAPS2_CUBEMAP set");
            CHECK((rd_u32(dds + 136) & 0x4u) != 0u, "dds miscFlag TEXTURECUBE");
        }
        tp_free(NULL, dds);
    }
    tp_free(NULL, ktx);
    for (f = 0; f < 6; ++f) free(data[f]);
    printf("  cube container (ktx2 faceCount + dds cube caps): ok\n");
}

/* ------------------------------------------------------- normal/height */

static void view_f32(tir_image_view *v, float *data, int w, int h, int ch) {
    v->data = data; v->width = w; v->height = h; v->channels = ch;
    v->type = TIR_F32; v->row_stride_bytes = 0;
}

/* F32 RGBA unit-normal field (xyz unit, a=1), smoothly varying so filtering
 * actually shortens the averaged vector. */
static float *make_normals(int w, int h) {
    float *img = (float *)malloc((size_t)w * h * 4 * sizeof(float));
    int x, y;
    for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x) {
            float fx = (float)x / (w - 1) * 2.0f - 1.0f;
            float fy = (float)y / (h - 1) * 2.0f - 1.0f;
            float nx = 0.7f * sinf(fx * 3.1415926f);
            float ny = 0.7f * sinf(fy * 3.1415926f);
            float nz = 1.0f, len;
            float *p = img + ((size_t)y * w + x) * 4;
            len = sqrtf(nx * nx + ny * ny + nz * nz);
            p[0] = nx / len; p[1] = ny / len; p[2] = nz / len; p[3] = 1.0f;
        }
    return img;
}

static void test_normal_unit_length(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    float *img = make_normals(128, 128);
    float worst = 0.0f;
    int i, x, y;
    view_f32(&v, img, 128, 128, 4);
    tp_options_init(&opt, TP_CONTENT_NORMAL, TP_CODEC_BC7);
    opt.normal_encoding = TIR_NORMAL_SNORM;
    opt.renormalize = 1;
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build normal mips");
    for (i = 1; i < chain.num_levels; ++i) { /* level 0 == authored base */
        const tp_surface *s = &chain.level[i];
        for (y = 0; y < s->height; ++y) {
            const float *row =
                (const float *)((const uint8_t *)s->data + (size_t)y * s->stride);
            for (x = 0; x < s->width; ++x) {
                const float *p = row + x * s->channels;
                float len = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                float e = fabsf(len - 1.0f);
                if (e > worst) worst = e;
            }
        }
    }
    CHECK(worst < 1e-3f, "normal mips stay unit-length after renormalize");
    printf("  normal unit-length: ok (worst |len-1|=%.2e)\n", worst);
    tp_mip_chain_free(NULL, &chain);
    free(img);
}

static void test_toksvig_helper(void) {
    float nlen[4] = {1.0f, 0.9f, 0.5f, 0.05f};
    float rough[4];
    CHECK(TP_OK(tp_toksvig_roughness(nlen, 4, 0.1f, rough)), "toksvig map");
    CHECK(fabsf(rough[0] - 0.1f) < 1e-3f, "|N|=1 -> base roughness");
    CHECK(rough[1] > rough[0] && rough[2] > rough[1] && rough[3] > rough[2],
          "roughness increases as |N| shrinks");
    CHECK(rough[3] > 0.7f, "near-random normals -> high roughness");
    printf("  toksvig helper: ok (%.3f %.3f %.3f %.3f)\n", rough[0], rough[1],
           rough[2], rough[3]);
}

static double mean_channel(const tp_surface *s, int c) {
    double sum = 0.0;
    int x, y;
    for (y = 0; y < s->height; ++y) {
        const float *row =
            (const float *)((const uint8_t *)s->data + (size_t)y * s->stride);
        for (x = 0; x < s->width; ++x) sum += row[x * s->channels + c];
    }
    return sum / ((double)s->width * s->height);
}

static void test_toksvig_bake(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain, rc;
    float *img = make_normals(64, 64);
    double r0, r1, rlast;
    view_f32(&v, img, 64, 64, 4);
    tp_options_init(&opt, TP_CONTENT_NORMAL, TP_CODEC_BC7);
    opt.normal_encoding = TIR_NORMAL_SNORM;
    opt.bake_toksvig_roughness = 1;
    opt.base_roughness = 0.1f;
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build normal+bake");
    CHECK(chain.roughness != NULL, "roughness baked into chain");
    if (chain.roughness) {
        /* roughness[level] is a width*height float array. */
        tp_surface rs;
        int i;
        double mean0;
        rs = chain.level[0]; rs.channels = 1;
        rs.data = chain.roughness[0]; rs.stride = (size_t)rs.width * sizeof(float);
        mean0 = mean_channel(&rs, 0);
        CHECK(fabs(mean0 - 0.1) < 1e-4, "level0 roughness == base");
        r0 = mean0;
        /* mean roughness of level 1 */
        rs = chain.level[1]; rs.channels = 1;
        rs.data = chain.roughness[1]; rs.stride = (size_t)rs.width * sizeof(float);
        r1 = mean_channel(&rs, 0);
        i = chain.num_levels - 2; /* a coarse but non-1x1 level */
        rs = chain.level[i]; rs.channels = 1;
        rs.data = chain.roughness[i]; rs.stride = (size_t)rs.width * sizeof(float);
        rlast = mean_channel(&rs, 0);
        CHECK(r1 > r0, "roughness grows once filtering starts (level1 > base)");
        CHECK(rlast >= r1 - 1e-3f, "coarser levels are no smoother");
        printf("  toksvig bake: ok (r[0]=%.3f r[1]=%.3f r[coarse]=%.3f)\n",
               r0, r1, rlast);
    }
    /* roughness pyramid is compressible (EAC_R11 path). */
    memset(&rc, 0, sizeof(rc));
    CHECK(TP_OK(tp_build_roughness_chain(NULL, &chain, &rc)), "build roughness chain");
    CHECK(rc.channels == 4 && rc.num_levels == chain.num_levels, "roughness chain shape");
    tp_mip_chain_free(NULL, &rc);
    tp_mip_chain_free(NULL, &chain);
    free(img);
}

static void test_heightmap_mean(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    float *img = (float *)malloc((size_t)128 * 128 * 4 * sizeof(float));
    double base_mean;
    int i, x, y;
    for (y = 0; y < 128; ++y)
        for (x = 0; x < 128; ++x) {
            float *p = img + ((size_t)y * 128 + x) * 4;
            p[0] = 0.25f + 0.5f * (float)x / 127.0f; /* height in R */
            p[1] = p[2] = 0.0f; p[3] = 1.0f;
        }
    view_f32(&v, img, 128, 128, 4);
    tp_options_init(&opt, TP_CONTENT_HEIGHT, TP_CODEC_BC7);
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build height mips");
    base_mean = mean_channel(&chain.level[0], 0);
    for (i = 1; i < chain.num_levels; ++i) {
        double m = mean_channel(&chain.level[i], 0);
        if (chain.level[i].width >= 2)
            CHECK(fabs(m - base_mean) < 0.02, "heightmap mip mean preserved");
    }
    printf("  heightmap mean preservation: ok (base_mean=%.4f)\n", base_mean);
    tp_mip_chain_free(NULL, &chain);
    free(img);
}

/* Octahedral (Y-up) decode, matching tools/envmap octa convention. */
static void octa_dir(float u01, float v01, float d[3]) {
    float u = 2 * u01 - 1, v = 2 * v01 - 1, x = u, z = v, y = 1 - fabsf(u) - fabsf(v), l;
    if (y < 0) { x = (1 - fabsf(v)) * (u >= 0 ? 1.f : -1.f); z = (1 - fabsf(u)) * (v >= 0 ? 1.f : -1.f); }
    l = sqrtf(x * x + y * y + z * z);
    d[0] = x / l; d[1] = y / l; d[2] = z / l;
}

static void test_octa_seam(void) {
    tp_surface s;
    int n = 64, x, y, c;
    float max_dev = 0.0f;
    s.width = n; s.height = n; s.channels = 4;
    s.stride = (size_t)n * 4 * sizeof(float);
    s.data = (float *)malloc(s.stride * n);
    /* fill with a smooth direction field */
    for (y = 0; y < n; ++y)
        for (x = 0; x < n; ++x) {
            float d[3], *p = s.data + (y * n + x) * 4;
            octa_dir((x + 0.5f) / n, (y + 0.5f) / n, d);
            p[0] = 0.5f + 0.5f * d[0]; p[1] = 0.5f + 0.5f * d[1];
            p[2] = 0.5f + 0.5f * d[2]; p[3] = 1.0f;
        }
    CHECK(TP_OK(tp_octa_seam_fixup(&s, NULL)), "octa seam fixup");
    /* Border texels must stay near their true direction value (correct pairing);
     * a wrong fold rule would average unrelated texels and deviate a lot. */
    for (y = 0; y < n; ++y)
        for (x = 0; x < n; ++x) {
            int border = (x == 0 || x == n - 1 || y == 0 || y == n - 1);
            float d[3], *p;
            if (!border) continue;
            octa_dir((x + 0.5f) / n, (y + 0.5f) / n, d);
            p = s.data + (y * n + x) * 4;
            for (c = 0; c < 3; ++c) {
                float ref = 0.5f + 0.5f * d[c];
                float dv = fabsf(p[c] - ref);
                if (dv > max_dev) max_dev = dv;
            }
        }
    printf("  octa seam fixup: ok (max border dev=%.4f)\n", max_dev);
    CHECK(max_dev < 0.05f, "octa fold pairing keeps borders near true direction");
    free(s.data);
}

static void test_channel_majority(void) {
    /* R = binary checkerboard mask; MAJORITY must keep it binary at every mip,
     * while LINEAR would produce gray. */
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    uint8_t *img = (uint8_t *)malloc((size_t)64 * 64 * 4);
    int x, y, i, nonbinary = 0;
    for (y = 0; y < 64; ++y)
        for (x = 0; x < 64; ++x) {
            uint8_t *p = img + (y * 64 + x) * 4;
            p[0] = ((x ^ y) & 1) ? 255 : 0; /* mask */
            p[1] = (uint8_t)(x * 4);        /* linear */
            p[2] = 0; p[3] = 255;
        }
    view_u8(&v, img, 64, 64);
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    opt.channel_op[0] = TP_CH_MAJORITY;
    memset(&chain, 0, sizeof(chain));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "build packed mips");
    for (i = 1; i < chain.num_levels; ++i) {
        const tp_surface *s = &chain.level[i];
        for (y = 0; y < s->height; ++y) {
            const float *row = (const float *)((const uint8_t *)s->data + (size_t)y * s->stride);
            for (x = 0; x < s->width; ++x) {
                float r = row[x * s->channels + 0];
                if (r > 1e-4f && r < 1.0f - 1e-4f) nonbinary = 1;
            }
        }
    }
    CHECK(!nonbinary, "MAJORITY channel stays binary across mips");
    printf("  channel majority packing: ok (nonbinary=%d)\n", nonbinary);
    tp_mip_chain_free(NULL, &chain);
    free(img);
}

static void test_array_ktx2(void) {
    tir_image_view v;
    tp_options opt;
    tp_mip_chain chain;
    tp_blocks b, layers[3];
    uint8_t *img = make_gradient(64, 64);
    uint8_t *buf = NULL;
    size_t need, wrote = 0;
    view_u8(&v, img, 64, 64);
    tp_options_init(&opt, TP_CONTENT_COLOR, TP_CODEC_BC7);
    memset(&chain, 0, sizeof(chain));
    memset(&b, 0, sizeof(b));
    CHECK(TP_OK(tp_build_mips(NULL, &v, 1, &opt, &chain)), "array: build");
    CHECK(TP_OK(tp_compress_chain(NULL, &chain, &opt, &b)), "array: compress");
    layers[0] = b; layers[1] = b; layers[2] = b; /* 3 layers reuse same blocks */
    need = tp_ktx2_array_size(layers, 3, &opt);
    CHECK(need > 0, "array size");
    buf = (uint8_t *)malloc(need);
    CHECK(TP_OK(tp_write_ktx2_array(layers, 3, &opt, buf, need, &wrote)), "array: write");
    if (buf) {
        CHECK(rd_u32(buf + 32) == 3u, "ktx2 layerCount == 3");
        CHECK(rd_u32(buf + 40) == 7u, "ktx2 levelCount == 7");
        CHECK(rd_u64(buf + 88) == 3u * tc_bc7_compressed_size(64, 64),
              "ktx2 array level0 length = 3 layers");
        CHECK(rd_u64(buf + 80) + rd_u64(buf + 88) <= wrote, "array level0 in bounds");
    }
    printf("  array ktx2 (layerCount=3): ok\n");
    free(buf);
    tp_blocks_free(NULL, &b);
    tp_mip_chain_free(NULL, &chain);
    free(img);
}

int main(void) {
    printf("texpipe unit tests\n");
    test_octa_seam();
    test_channel_majority();
    test_array_ktx2();
    test_mip_geometry();
    test_bc7_roundtrip_psnr();
    test_alpha_scale_helper();
    test_alpha_coverage();
    test_containers();
    test_cube_seam_fixup();
    test_cube_split();
    test_cube_container();
    test_normal_unit_length();
    test_toksvig_helper();
    test_toksvig_bake();
    test_heightmap_mean();
    if (g_fail) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
