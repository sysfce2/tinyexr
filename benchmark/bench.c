/*
 * TinyEXR v3 - standalone decode/encode + SIMD-kernel benchmark.
 *
 * Usage:
 *   make bench            # builds + runs on asakusa.exr
 *   ./build/bench [file.exr ...]
 *
 * Reports, per codec, decode and encode throughput (megapixels/s), and per
 * SIMD primitive (byte de-interleave, half<->float) the throughput at each
 * available ISA tier (scalar / SSE2 / AVX2 / F16C), with speedups.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define _POSIX_C_SOURCE 199309L

#include "exr.h"
#include "exr_internal.h" /* kernel symbols + exr_simd_force (benchmark-only) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Run `fn(arg)` repeatedly for >= min_sec; return seconds/iteration. */
typedef void (*bench_fn)(void *);
static double timeit(bench_fn fn, void *arg, double min_sec, long *out_iters) {
    long iters = 0;
    double t0 = now_sec(), elapsed;
    do {
        fn(arg);
        ++iters;
        elapsed = now_sec() - t0;
    } while (elapsed < min_sec);
    if (out_iters) *out_iters = iters;
    return elapsed / (double)iters;
}

/* ---- codec decode / encode ------------------------------------------------ */

struct codec_ctx {
    const exr_image *src;  /* for encode */
    const void *buf;       /* for decode */
    size_t buf_size;
    exr_compression comp;
};

static void do_encode(void *p) {
    struct codec_ctx *c = (struct codec_ctx *)p;
    void *out = NULL;
    size_t sz = 0;
    if (EXR_OK(exr_save_to_memory(&out, &sz, NULL, c->src, c->comp))) free(out);
}
static void do_decode(void *p) {
    struct codec_ctx *c = (struct codec_ctx *)p;
    exr_image img;
    memset(&img, 0, sizeof(img));
    if (EXR_OK(exr_load_from_memory(c->buf, c->buf_size, NULL, &img)))
        exr_image_free(&img);
}

static void bench_codecs(const char *path) {
    static const exr_compression codecs[] = {
        EXR_COMPRESSION_NONE, EXR_COMPRESSION_RLE, EXR_COMPRESSION_ZIPS,
        EXR_COMPRESSION_ZIP, EXR_COMPRESSION_PIZ};
    static const char *names[] = {"none", "rle", "zips", "zip", "piz"};
    exr_image src;
    double mpix;
    size_t i;

    memset(&src, 0, sizeof(src));
    if (!EXR_OK(exr_load_from_file(path, NULL, &src))) {
        printf("  (could not load %s)\n", path);
        return;
    }
    mpix = (double)src.parts[0].width * src.parts[0].height / 1e6;
    printf("\n== codec throughput: %s (%dx%d, %d ch) ==\n", path,
           src.parts[0].width, src.parts[0].height,
           src.parts[0].header.num_channels);
    printf("  %-6s %12s %12s %10s\n", "codec", "encode MP/s", "decode MP/s",
           "size");

    for (i = 0; i < sizeof(codecs) / sizeof(codecs[0]); ++i) {
        struct codec_ctx ec, dc;
        void *buf = NULL;
        size_t sz = 0;
        double te, td;
        if (!EXR_OK(exr_save_to_memory(&buf, &sz, NULL, &src, codecs[i]))) {
            printf("  %-6s   (encode unsupported)\n", names[i]);
            continue;
        }
        ec.src = &src;
        ec.comp = codecs[i];
        te = timeit(do_encode, &ec, 0.3, NULL);
        dc.buf = buf;
        dc.buf_size = sz;
        dc.comp = codecs[i];
        td = timeit(do_decode, &dc, 0.3, NULL);
        printf("  %-6s %12.1f %12.1f %9.0fK\n", names[i], mpix / te, mpix / td,
               sz / 1024.0);
        free(buf);
    }

    /* end-to-end SIMD tier comparison on the ZIP decode path */
    {
        void *buf = NULL;
        size_t sz = 0;
        if (EXR_OK(exr_save_to_memory(&buf, &sz, NULL, &src, EXR_COMPRESSION_ZIP))) {
            struct codec_ctx dc;
            int lvl;
            const char *lv[] = {"scalar", "sse2", "avx2"};
            dc.buf = buf;
            dc.buf_size = sz;
            printf("  ZIP decode by forced SIMD tier (de-interleave dispatch):\n");
            for (lvl = 0; lvl <= 2; ++lvl) {
                double td;
                exr_simd_force(lvl);
                td = timeit(do_decode, &dc, 0.3, NULL);
                printf("    %-7s %10.1f MP/s\n", lv[lvl], mpix / td);
            }
            exr_simd_init(); /* restore best */
            free(buf);
        }
    }
    exr_image_free(&src);
}

/* ---- SIMD micro-kernels --------------------------------------------------- */

static void bench_kernels(void) {
    const size_t N = 64u * 1024u * 1024u; /* 64 MB */
    uint8_t *src = (uint8_t *)malloc(N), *dst = (uint8_t *)malloc(N);
    size_t i;
    double t;
    long it;
    uint32_t caps = exr_simd_capabilities();

    if (!src || !dst) { free(src); free(dst); return; }
    for (i = 0; i < N; ++i) src[i] = (uint8_t)i;

    printf("\n== SIMD kernels (%.0f MB buffers) ==\n", N / 1048576.0);

    printf("  byte de-interleave:\n");
    {
        struct { const char *n; void (*f)(const uint8_t *, uint8_t *, size_t); int ok; } v[] = {
            {"scalar", exr_interleave_scalar, 1},
#if defined(EXR_X86)
            {"sse2", exr_interleave_sse2, (caps & EXR_SIMD_SSE2) != 0},
            {"avx2", exr_interleave_avx2, (caps & EXR_SIMD_AVX2) != 0},
#endif
        };
        double base = 0;
        size_t k;
        for (k = 0; k < sizeof(v) / sizeof(v[0]); ++k) {
            double gbs;
            if (!v[k].ok) { printf("    %-7s   (unavailable)\n", v[k].n); continue; }
            { /* inline timing (kernel sig differs from bench_fn) */
                double t0 = now_sec();
                long n = 0;
                do { v[k].f(src, dst, N); ++n; } while (now_sec() - t0 < 0.3);
                t = (now_sec() - t0) / (double)n;
                it = n;
            }
            gbs = (double)N / t / 1e9;
            if (k == 0) base = gbs;
            printf("    %-7s %8.2f GB/s  (%.2fx)  [%ld it]\n", v[k].n, gbs,
                   base > 0 ? gbs / base : 1.0, it);
        }
    }

    printf("  half->float:\n");
    {
        size_t M = N / 2; /* halves */
        uint16_t *h = (uint16_t *)src;
        float *f = (float *)malloc(M * sizeof(float));
        if (f) {
            double base = 0, t0, gbs;
            long n;
            t0 = now_sec(); n = 0;
            do { exr_half_to_float_scalar(h, f, M); ++n; } while (now_sec() - t0 < 0.3);
            t = (now_sec() - t0) / n;
            base = (double)M * 6 / t / 1e9; /* 2B in + 4B out */
            printf("    %-7s %8.2f GB/s  (1.00x)\n", "scalar", base);
#if defined(EXR_X86)
            if (caps & EXR_SIMD_AVX2) {
                t0 = now_sec(); n = 0;
                do { exr_half_to_float_f16c(h, f, M); ++n; } while (now_sec() - t0 < 0.3);
                t = (now_sec() - t0) / n;
                gbs = (double)M * 6 / t / 1e9;
                printf("    %-7s %8.2f GB/s  (%.2fx)\n", "f16c", gbs, gbs / base);
            }
#endif
            (void)gbs;
            free(f);
        }
    }
    free(src);
    free(dst);
}

int main(int argc, char **argv) {
    printf("TinyEXR v3 benchmark  |  SIMD: %s (caps=0x%x)\n", exr_simd_info(),
           exr_simd_capabilities());

    if (argc > 1) {
        int i;
        for (i = 1; i < argc; ++i) bench_codecs(argv[i]);
    } else {
        bench_codecs("asakusa.exr");
    }
    bench_kernels();

    printf("\nNote: the SSE4.1/AVX2 PSHUFB Huffman-emit kernel (fpnge port) for\n"
           "the DEFLATE/ZIP encoder is not yet wired in; the ZIP encoder above\n"
           "uses the scalar fpng Huffman path. Decode de-interleave is\n"
           "SIMD-dispatched (scalar/SSE2/AVX2 shown).\n");
    return 0;
}
