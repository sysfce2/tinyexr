/*
 * TinyEXR v3 (pure-C11) fuzz harness.
 *
 * Exercises every decode entry point on the same bytes:
 *   1. high-level exr_load_from_memory()
 *   2. mid-level reader: parse_header + read every part
 *   3. streaming source path (full synchronous feed)
 *
 * Build (coverage-guided):  make fuzz       -> build/fuzz_v3
 * Build (corpus replay):    make fuzz-corpus -> build/fuzz_replay <files...>
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct mem_source {
    const uint8_t *data;
    size_t size;
};

static exr_result feed_read(void *user, uint64_t offset, uint64_t len,
                            void *dst) {
    struct mem_source *m = (struct mem_source *)user;
    if (offset > m->size || len > m->size - offset) return EXR_ERROR_CORRUPT;
    if (len) memcpy(dst, m->data + (size_t)offset, (size_t)len);
    return EXR_SUCCESS;
}

static void drain_reader(exr_reader *r) {
    int np, p;
    if (!EXR_OK(exr_reader_parse_header(r))) return;
    np = exr_reader_num_parts(r);
    for (p = 0; p < np; ++p) {
        exr_part out;
        memset(&out, 0, sizeof(out));
        if (EXR_OK(exr_reader_read_part(r, p, &out))) exr_part_free(NULL, &out);
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* 1. high-level load (all parts, all channels). */
    {
        exr_image img;
        memset(&img, 0, sizeof(img));
        if (EXR_OK(exr_load_from_memory(data, size, NULL, &img)))
            exr_image_free(&img);
    }

    /* 2. mid-level reader over the same bytes (zero-copy memory path). */
    {
        exr_reader *r = NULL;
        if (EXR_OK(exr_reader_open_memory(data, size, NULL, &r)) && r) {
            drain_reader(r);
            exr_reader_close(r);
        }
    }

    /* 3. streaming source path (synchronous full feed, exercises buffering). */
    if (size > 0) {
        struct mem_source m;
        exr_data_source ds;
        exr_reader *r = NULL;
        m.data = data;
        m.size = size;
        ds.user = &m;
        ds.read = feed_read;
        ds.total_size = (uint64_t)size;
        if (EXR_OK(exr_reader_open_source(&ds, NULL, &r)) && r) {
            drain_reader(r);
            exr_reader_close(r);
        }
    }
    return 0;
}

#ifdef EXR_FUZZ_STANDALONE
#include <stdio.h>
#include <stdlib.h>

/* Deterministic replay over a list of files (CI corpus check). */
int main(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; ++i) {
        FILE *fp = fopen(argv[i], "rb");
        long n;
        uint8_t *buf;
        if (!fp) { fprintf(stderr, "  skip (open failed): %s\n", argv[i]); continue; }
        fseek(fp, 0, SEEK_END);
        n = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (n < 0) { fclose(fp); continue; }
        buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
        if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) { /* ignore short read */ }
        fclose(fp);
        LLVMFuzzerTestOneInput(buf, (size_t)n);
        free(buf);
        printf("  ok (no crash): %s\n", argv[i]);
    }
    printf("fuzz corpus replay: %d file(s), no crash / no sanitizer error\n",
           argc - 1);
    return 0;
}
#endif
