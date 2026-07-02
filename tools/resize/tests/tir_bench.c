/*
 * tir - throughput benchmark. Run from the repository root:
 *
 *   make resize-bench            # tir at every forced SIMD level vs the
 *                                # tinyexr core resizer (exr_resize_float)
 *   make resize-bench STB=1     # adds a stb_image_resize2 column
 *
 * Reports output megapixels per second (higher is better).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define _POSIX_C_SOURCE 199309L /* clock_gettime under -std=c11 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "exr.h"
#include "tir.h"

#if defined(TIR_BENCH_STB)
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t xr(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

typedef struct shape {
    const char *name;
    int sw, sh, dw, dh;
} shape;

static const shape SHAPES[] = {
    {"2x up   ", 960, 640, 1920, 1280},
    {"2x down ", 1920, 1280, 960, 640},
    {"7.3x dn ", 4096, 2560, 561, 350},
};

typedef struct fcase {
    const char *name;
    tir_filter tf;
    exr_resize_filter ef;
    int have_exr;
    int stb_filter; /* stbir_filter enum value; -1 = no stb equivalent */
    int stb_wider;  /* 1 if tir's filter is wider than stb's (unfair to tir) */
} fcase;

/* stbir_filter: 1=BOX 2=TRIANGLE 3=CUBICBSPLINE 4=CATMULLROM 5=MITCHELL.
 * stb has no Lanczos; the lanczos3 row runs a 4-tap Catmull-Rom on the stb
 * side, so tir is doing 1.5x the taps there (flagged stb_wider). */
static const fcase FILTERS[] = {
    {"triangle", TIR_FILTER_TRIANGLE, EXR_RESIZE_TRIANGLE, 1, 2, 0},
    {"mitchell", TIR_FILTER_MITCHELL, EXR_RESIZE_MITCHELL, 1, 5, 0},
    {"lanczos3", TIR_FILTER_LANCZOS3, (exr_resize_filter)0, 0, 4, 1},
};

/* run fn repeatedly until ~0.3s elapsed, return output MP/s */
#define BENCH(mps_out, dw, dh, CALL)                              \
    do {                                                          \
        double t0, el;                                            \
        int reps = 0;                                             \
        CALL; /* warm up */                                       \
        t0 = now_sec();                                           \
        do {                                                      \
            CALL;                                                 \
            reps++;                                               \
            el = now_sec() - t0;                                  \
        } while (el < 0.3);                                       \
        (mps_out) = (double)(dw) * (dh) * reps / el / 1e6;        \
    } while (0)

int main(void) {
    static const tir_simd_level LEVELS[] = {TIR_SIMD_SCALAR, TIR_SIMD_SSE2,
                                            TIR_SIMD_SSE41, TIR_SIMD_AVX2,
                                            TIR_SIMD_NEON, TIR_SIMD_SVE};
    static const char *LNAME[] = {"scalar", "sse2", "sse4.1",
                                  "avx2",   "neon", "sve"};
    uint32_t avail = tir_simd_available();
    int chset[2] = {1, 4};
    int si, fi, ci, li;

    printf("tir bench: output MP/s, f32 pixels (best SIMD: %s)\n\n",
           tir_simd_info());

    for (ci = 0; ci < 2; ++ci) {
        int ch = chset[ci];
        printf("== %d channel%s ==\n", ch, ch > 1 ? "s" : "");
        printf("%-9s %-9s", "shape", "filter");
        for (li = 0; li < 6; ++li)
            if (avail & (1u << LEVELS[li])) printf(" %9s", LNAME[li]);
        printf(" %9s", "exr_v3");
#if defined(TIR_BENCH_STB)
        printf(" %9s", "stb2");
#endif
        printf("\n");
        for (si = 0; si < 3; ++si) {
            const shape *sp = &SHAPES[si];
            size_t ns = (size_t)sp->sw * sp->sh * ch;
            size_t nd = (size_t)sp->dw * sp->dh * ch;
            float *src = (float *)malloc(ns * sizeof(float));
            float *dst = (float *)malloc(nd * sizeof(float));
            size_t i;
            for (i = 0; i < ns; ++i)
                src[i] = (float)(xr() >> 8) / (float)(1 << 20);
            for (fi = 0; fi < 3; ++fi) {
                const fcase *fc = &FILTERS[fi];
                tir_image_view sv = {src, sp->sw, sp->sh, ch, TIR_F32, 0};
                tir_image_view dv = {dst, sp->dw, sp->dh, ch, TIR_F32, 0};
                tir_options o;
                double mps;
                tir_options_init(&o);
                o.filter_x = o.filter_y = fc->tf;
                o.alpha = TIR_ALPHA_STRAIGHT;
                printf("%-9s %-9s", sp->name, fc->name);
                for (li = 0; li < 6; ++li) {
                    if (!(avail & (1u << LEVELS[li]))) continue;
                    tir_simd_force(LEVELS[li]);
                    BENCH(mps, sp->dw, sp->dh,
                          tir_resize(NULL, &sv, &dv, &o));
                    printf(" %9.1f", mps);
                }
                /* restore best level */
                for (li = 5; li >= 0; --li)
                    if (avail & (1u << LEVELS[li])) {
                        tir_simd_force(LEVELS[li]);
                        break;
                    }
                if (fc->have_exr) {
                    BENCH(mps, sp->dw, sp->dh,
                          exr_resize_float(NULL, src, sp->sw, sp->sh, 0, dst,
                                           sp->dw, sp->dh, 0, ch, fc->ef,
                                           EXR_EDGE_CLAMP, -1));
                    printf(" %9.1f", mps);
                } else {
                    printf(" %9s", "-");
                }
#if defined(TIR_BENCH_STB)
                { /* matched filter (Catmull-Rom stands in for Lanczos3) */
                    stbir_pixel_layout pl =
                        ch == 1 ? STBIR_1CHANNEL : STBIR_4CHANNEL;
                    stbir_filter sf = (stbir_filter)fc->stb_filter;
                    BENCH(mps, sp->dw, sp->dh,
                          stbir_resize(src, sp->sw, sp->sh, 0, dst, sp->dw,
                                       sp->dh, 0, pl, STBIR_TYPE_FLOAT,
                                       STBIR_EDGE_CLAMP, sf));
                    printf(" %9.1f%s", mps, fc->stb_wider ? "*" : " ");
                }
#endif
                printf("\n");
            }
            free(src);
            free(dst);
        }
        printf("\n");
    }
    return 0;
}
