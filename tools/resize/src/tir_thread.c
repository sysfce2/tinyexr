/*
 * tir - optional C11-threads band parallelism (-DTIR_ENABLE_THREADS).
 *
 * Whole-image runs split the destination rows into horizontal bands; each
 * worker clones the sampler (own ring + scratch, tables rebuilt - creation
 * is cheap next to the filtering) and runs its band independently, reading
 * the shared source. Per-row math is identical to the serial path, so the
 * output is byte-identical for any thread count. Without the compile flag
 * this file is an empty translation unit and num_threads degrades to serial.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tir_internal.h"

#if defined(TIR_ENABLE_THREADS)

#include <threads.h>

#define TIR_MIN_BAND_ROWS 4

typedef struct tir__job {
    const tir_sampler *proto;
    const void *src;
    size_t ss;
    void *dst;
    size_t ds;
    int y0, y1;
    tir_result rc;
} tir__job;

static int tir__worker(void *arg) {
    tir__job *j = (tir__job *)arg;
    const tir_sampler *p = j->proto;
    tir_sampler *s = NULL;
    tir_result rc =
        tir_sampler_create(&p->alloc, p->sw, p->sh, p->dw, p->dh, p->ch,
                           p->st, p->dt, &p->opt, &s);
    if (TIR_OK(rc)) {
        rc = tir__run_range(s, j->src, j->ss, j->dst, j->ds, j->y0, j->y1);
        tir_sampler_destroy(s);
    }
    j->rc = rc;
    return 0;
}

tir_result tir__run_threads(tir_sampler *s, const void *src,
                            size_t src_row_stride_bytes, void *dst,
                            size_t dst_row_stride_bytes, int nthreads) {
    tir__job jobs[64];
    thrd_t tids[64];
    int n = nthreads, i, spawned = 0;
    tir_result rc = TIR_SUCCESS;

    if (n > 64) n = 64;
    if (n > s->dh / TIR_MIN_BAND_ROWS) n = s->dh / TIR_MIN_BAND_ROWS;
    if (n <= 1)
        return tir__run_range(s, src, src_row_stride_bytes, dst,
                              dst_row_stride_bytes, 0, s->dh);

    for (i = 0; i < n; ++i) {
        jobs[i].proto = s;
        jobs[i].src = src;
        jobs[i].ss = src_row_stride_bytes;
        jobs[i].dst = dst;
        jobs[i].ds = dst_row_stride_bytes;
        jobs[i].y0 = (int)((long long)s->dh * i / n);
        jobs[i].y1 = (int)((long long)s->dh * (i + 1) / n);
        jobs[i].rc = TIR_SUCCESS;
    }
    /* workers get bands 1..n-1; this thread reuses `s` for band 0 */
    for (i = 1; i < n; ++i) {
        if (thrd_create(&tids[i], tir__worker, &jobs[i]) != thrd_success)
            break;
        spawned = i;
    }
    rc = tir__run_range(s, src, src_row_stride_bytes, dst,
                        dst_row_stride_bytes, jobs[0].y0, jobs[0].y1);
    for (i = 1; i <= spawned; ++i) {
        thrd_join(tids[i], NULL);
        if (TIR_OK(rc) && !TIR_OK(jobs[i].rc)) rc = jobs[i].rc;
    }
    /* bands whose worker failed to spawn run serially here */
    for (i = spawned + 1; i < n; ++i) {
        if (!TIR_OK(rc)) break;
        rc = tir__run_range(s, src, src_row_stride_bytes, dst,
                            dst_row_stride_bytes, jobs[i].y0, jobs[i].y1);
    }
    return rc;
}

#else

/* keep the translation unit non-empty under strict C11 */
typedef int tir__threads_disabled;

#endif /* TIR_ENABLE_THREADS */
