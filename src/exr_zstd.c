/*
 * TinyEXR - Zstandard codec glue.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"
#include "tinyexr_zstd.h"

#define EXR_ZSTD_LEVEL 3

exr_result exr_zstd_decompress(const exr_allocator *a, const uint8_t *src,
                               size_t src_size, uint8_t *dst, size_t dst_size) {
    size_t n;

#ifdef EXR_FREESTANDING
    /* Freestanding: zstd's internal malloc is stubbed out (see the Makefile's
     * EXR_FREESTANDING_ZSTD object), so drive the no-malloc static-DCtx path
     * over a workspace from the caller's allocator. The DCtx state is small and
     * constant (one-shot decode reads back-references straight from dst). */
    void *ws;
    size_t wsize;
    ZSTD_DCtx *dctx;

    wsize = tinyexr_zstd_ZSTD_estimateDCtxSize();
    ws = exr_malloc(a, wsize ? wsize : 1);
    if (!ws) return EXR_ERROR_OUT_OF_MEMORY;
    dctx = tinyexr_zstd_ZSTD_initStaticDCtx(ws, wsize);
    if (!dctx) { exr_free(a, ws); return EXR_ERROR_CORRUPT; }
    n = tinyexr_zstd_ZSTD_decompressDCtx(dctx, dst, dst_size, src, src_size);
    exr_free(a, ws);
#else
    (void)a;
    n = tinyexr_zstd_decompress(dst, dst_size, src, src_size);
#endif
    if (tinyexr_zstd_is_error(n) || n != dst_size) return EXR_ERROR_CORRUPT;
    return EXR_SUCCESS;
}

exr_result exr_zstd_compress(const exr_allocator *a, const uint8_t *src,
                             size_t n, uint8_t **out_data, size_t *out_size) {
    size_t bound, clen;
    uint8_t *comp;
#ifdef EXR_ZSTD_DECODE_ONLY
    /* Decode-only build (freestanding): zstd encode is not available - the
     * amalgamation's compressor is stubbed out. Report it cleanly so every
     * caller (codec dispatch and the deep path) gets UNSUPPORTED, not a silent
     * store-raw fallback. */
    (void)a; (void)src; (void)n; (void)bound; (void)clen; (void)comp;
    *out_data = NULL;
    *out_size = 0;
    return EXR_ERROR_UNSUPPORTED;
#else

    *out_data = NULL;
    *out_size = 0;
    if (n == 0) {
        *out_data = (uint8_t *)exr_malloc(a, 1);
        if (!*out_data) return EXR_ERROR_OUT_OF_MEMORY;
        return EXR_SUCCESS;
    }

    bound = tinyexr_zstd_compress_bound(n);
    if (tinyexr_zstd_is_error(bound) || bound == 0) return EXR_ERROR_CORRUPT;

    comp = (uint8_t *)exr_malloc(a, bound);
    if (!comp) return EXR_ERROR_OUT_OF_MEMORY;

    clen = tinyexr_zstd_compress(comp, bound, src, n, EXR_ZSTD_LEVEL);
    if (tinyexr_zstd_is_error(clen) || clen >= n) {
        exr_free(a, comp);
        *out_data = (uint8_t *)exr_malloc(a, n);
        if (!*out_data) return EXR_ERROR_OUT_OF_MEMORY;
        memcpy(*out_data, src, n);
        *out_size = n;
        return EXR_SUCCESS;
    }

    *out_data = comp;
    *out_size = clen;
    return EXR_SUCCESS;
#endif /* EXR_ZSTD_DECODE_ONLY */
}
