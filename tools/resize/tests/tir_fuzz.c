/*
 * tir - resize fuzz harness.
 *
 * Drives the whole surface from fuzzer bytes on every input:
 *   1. One-shot tir_resize: fuzzed dims / channels / src+dst pixel types /
 *      filters / edge modes / mode (general|normal|heightmap) / options
 *      (antiring, clamp, nonfinite, alpha, hicomp, registration, filter_scale,
 *      gaussian_sigma, deterministic) and, when built with threads, a fuzzed
 *      num_threads to exercise the banded whole-image path.
 *   2. Streaming tir_sampler push/pull with the same style of fuzzed config.
 * The source pixels are taken straight from the fuzz stream, so bytes that
 * happen to decode to NaN/Inf floats exercise the nonfinite policies too.
 *
 * Build (coverage-guided, libFuzzer):
 *   make resize-fuzz
 *   ./build/resize_fuzz -max_total_time=300 [corpus_dir]
 *
 * Build (deterministic replay / CI gate, ASan+UBSan, no libFuzzer):
 *   make resize-fuzz-corpus
 *   ./build/resize_fuzz_replay [files...]   (no args -> generated inputs)
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tir.h"

/* ---- byte-stream reader ---------------------------------------------------- */

typedef struct {
    const uint8_t *p;
    size_t n, i;
} rd;

static uint8_t u8(rd *r) { return r->i < r->n ? r->p[r->i++] : 0; }

static uint32_t u32(rd *r) {
    uint32_t v = 0;
    int k;
    for (k = 0; k < 4; ++k) v = (v << 8) | u8(r);
    return v;
}

/* inclusive [lo, hi] */
static int pick(rd *r, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(u32(r) % (uint32_t)(hi - lo + 1));
}

static size_t pixel_bytes(tir_pixel_type t) {
    return t == TIR_F32 ? 4 : t == TIR_F16 ? 2 : t == TIR_U8 ? 1 : 2;
}

/* Fill a buffer from the fuzz stream, then top up with a cheap PRNG so large
 * images still get varied (and occasionally NaN/Inf-looking) pixel bytes. */
static void fill(rd *r, void *buf, size_t bytes) {
    uint8_t *d = (uint8_t *)buf;
    uint32_t s = 0x9E3779B9u ^ (uint32_t)bytes;
    size_t k;
    for (k = 0; k < bytes; ++k) {
        if (r->i < r->n) {
            d[k] = r->p[r->i++];
        } else {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            d[k] = (uint8_t)s;
        }
    }
}

/* ---- one-shot resize ------------------------------------------------------- */

static void one_run(rd *r) {
    int sw = pick(r, 1, 40), sh = pick(r, 1, 40);
    int dw = pick(r, 1, 40), dh = pick(r, 1, 40);
    int ch = pick(r, 1, 4);
    tir_pixel_type st = (tir_pixel_type)pick(r, 0, 3);
    tir_pixel_type dt = (tir_pixel_type)pick(r, 0, 3);
    tir_options o;
    void *src, *dst;
    float *nlen = NULL;
    size_t sbytes, dbytes;

    tir_options_init(&o);
    o.filter_x = (tir_filter)pick(r, 0, 8);
    o.filter_y = (tir_filter)pick(r, 0, 8);
    o.edge_x = (tir_edge_mode)pick(r, 0, 2);
    o.edge_y = (tir_edge_mode)pick(r, 0, 2);
    o.mode = (tir_mode)pick(r, 0, 2);
    o.nonfinite = (tir_nonfinite)pick(r, 0, 3);
    o.alpha = (tir_alpha_mode)pick(r, 0, 2);
    o.normal_encoding = (tir_normal_enc)pick(r, 0, 3);
    o.registration = (tir_registration)pick(r, 0, 1);
    o.antiring = (u8(r) & 1) ? -1.0f : (float)u8(r) / 255.0f; /* auto or [0,1] */
    o.hicomp = u8(r) & 1;
    o.deterministic = u8(r) & 1;
    o.filter_scale = 0.5f + (float)u8(r) / 64.0f;   /* 0.5 .. ~4.5 */
    o.gaussian_sigma = 0.25f + (float)u8(r) / 128.0f;
    if (u8(r) & 1) o.clamp_min = 0.0f;
#if defined(TIR_ENABLE_THREADS)
    o.num_threads = pick(r, 0, 4); /* exercise the banded whole-image path */
#endif
    /* normal-map mode needs a matching channel count (else clean UNSUPPORTED) */
    if (o.mode == TIR_MODE_NORMAL_MAP)
        ch = (o.normal_encoding == TIR_NORMAL_RG) ? 2 : 3;

    sbytes = (size_t)sw * sh * ch * pixel_bytes(st);
    dbytes = (size_t)dw * dh * ch * pixel_bytes(dt);
    src = malloc(sbytes ? sbytes : 1);
    dst = malloc(dbytes ? dbytes : 1);
    if (o.mode == TIR_MODE_NORMAL_MAP && (u8(r) & 1))
        nlen = (float *)malloc((size_t)dw * dh * sizeof(float));
    o.normal_length_out = nlen;
    if (src && dst) {
        tir_image_view sv = {src, sw, sh, ch, st, 0};
        tir_image_view dv = {dst, dw, dh, ch, dt, 0};
        fill(r, src, sbytes);
        tir_resize(NULL, &sv, &dv, &o);
    }
    free(src);
    free(dst);
    free(nlen);
}

/* ---- streaming push/pull --------------------------------------------------- */

static void stream_run(rd *r) {
    int sw = pick(r, 1, 32), sh = pick(r, 1, 32);
    int dw = pick(r, 1, 32), dh = pick(r, 1, 32);
    int ch = pick(r, 1, 4);
    tir_pixel_type st = (tir_pixel_type)pick(r, 0, 3);
    tir_pixel_type dt = (tir_pixel_type)pick(r, 0, 3);
    tir_options o;
    size_t srow, drow;
    uint8_t *src, *dst;
    tir_sampler *s = NULL;

    tir_options_init(&o);
    o.filter_x = o.filter_y = (tir_filter)pick(r, 0, 8);
    o.edge_x = o.edge_y = (tir_edge_mode)pick(r, 0, 2);
    o.antiring = (u8(r) & 1) ? 1.0f : 0.0f;
    o.nonfinite = (tir_nonfinite)pick(r, 0, 3);
    if (u8(r) & 1) o.clamp_min = 0.0f;

    srow = (size_t)sw * ch * pixel_bytes(st);
    drow = (size_t)dw * ch * pixel_bytes(dt);
    src = (uint8_t *)malloc((size_t)sh * srow ? (size_t)sh * srow : 1);
    dst = (uint8_t *)malloc(drow ? drow : 1);
    if (src && dst &&
        tir_sampler_create(NULL, sw, sh, dw, dh, ch, st, dt, &o, &s) ==
            TIR_SUCCESS) {
        int out = 0, y = 0, guard = 0, cap = sh + dh + 16;
        fill(r, src, (size_t)sh * srow);
        while (out < dh && guard++ < cap) {
            int dy;
            tir_result rc = tir_sampler_pull_row(s, &dy, dst);
            if (rc == TIR_WOULD_BLOCK) {
                if (y >= sh) break;
                if (tir_sampler_push_row(s, y, src + (size_t)y * srow) !=
                    TIR_SUCCESS)
                    break;
                y++;
                continue;
            }
            if (rc != TIR_SUCCESS) break;
            out++;
        }
        tir_sampler_destroy(s);
    }
    free(src);
    free(dst);
}

/* ---- fuzzer entry ---------------------------------------------------------- */

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    rd r;
    r.p = data;
    r.n = size;
    r.i = 0;
    one_run(&r);
    stream_run(&r);
    return 0;
}

/* ---- deterministic replay / generator (no libFuzzer) ----------------------- */

#ifdef TIR_FUZZ_STANDALONE
#include <stdio.h>

static uint32_t g_seed = 0xC0FFEEu;
static uint32_t nx(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int i;
        for (i = 1; i < argc; ++i) {
            FILE *fp = fopen(argv[i], "rb");
            long n;
            uint8_t *buf;
            if (!fp) continue;
            fseek(fp, 0, SEEK_END);
            n = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (n < 0) {
                fclose(fp);
                continue;
            }
            buf = (uint8_t *)malloc(n ? (size_t)n : 1);
            if (!buf) {
                fclose(fp);
                continue;
            }
            if (n > 0 && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
                /* short read: run with what we have */
            }
            fclose(fp);
            LLVMFuzzerTestOneInput(buf, (size_t)n);
            free(buf);
        }
        printf("resize fuzz replay: %d file(s), no crash / no sanitizer error\n",
               argc - 1);
    } else {
        int t, iters = 8000;
        uint8_t buf[256];
        for (t = 0; t < iters; ++t) {
            size_t k, n = 1 + nx() % sizeof(buf);
            for (k = 0; k < n; ++k) buf[k] = (uint8_t)nx();
            LLVMFuzzerTestOneInput(buf, n);
        }
        printf("resize fuzz: %d generated inputs, no crash / no sanitizer "
               "error\n",
               iters);
    }
    return 0;
}
#endif /* TIR_FUZZ_STANDALONE */
