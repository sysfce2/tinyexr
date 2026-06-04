/*
 * TinyEXR - mid-level reader: open, parse header(s), read pixels.
 *
 * Phase 1 covers the memory path, scanline parts, and NONE compression.
 * Compressed codecs are wired through exr_decompress_block(); tiled/deep parts
 * and streaming suspend/resume arrive in later phases.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

#include <stdlib.h>

/* ============================================================================
 * Open / close
 * ========================================================================== */

exr_result exr_reader_open_memory(const void *data, size_t size,
                                  const exr_allocator *alloc, exr_reader **out) {
    exr_reader *r;
    if (!data || !out) return EXR_ERROR_INVALID_ARGUMENT;
    if (!alloc) alloc = exr_default_allocator();
    r = (exr_reader *)exr_calloc(alloc, 1, sizeof(*r));
    if (!r) return EXR_ERROR_OUT_OF_MEMORY;
    r->alloc = *alloc;
    r->kind = EXR_SRC_MEMORY;
    r->mem = (const uint8_t *)data;
    r->mem_size = size;
    r->free_mem = 0;
    *out = r;
    return EXR_SUCCESS;
}

exr_result exr_reader_open_source(const exr_data_source *src,
                                  const exr_allocator *alloc, exr_reader **out) {
    exr_reader *r;
    uint8_t *buf;
    exr_result rc;
    if (!src || !src->read || !out) return EXR_ERROR_INVALID_ARGUMENT;
    if (!alloc) alloc = exr_default_allocator();

    /* Phase 1: support synchronous, size-known sources by buffering the whole
     * file. True incremental streaming/suspend lands in Phase 9. */
    if (src->total_size == 0) return EXR_ERROR_UNSUPPORTED;
    buf = (uint8_t *)exr_malloc(alloc, (size_t)src->total_size);
    if (!buf) return EXR_ERROR_OUT_OF_MEMORY;
    rc = src->read(src->user, 0, src->total_size, buf);
    if (rc == EXR_WOULD_BLOCK) rc = EXR_ERROR_UNSUPPORTED;
    if (!EXR_OK(rc)) {
        exr_free(alloc, buf);
        return rc;
    }
    r = (exr_reader *)exr_calloc(alloc, 1, sizeof(*r));
    if (!r) {
        exr_free(alloc, buf);
        return EXR_ERROR_OUT_OF_MEMORY;
    }
    r->alloc = *alloc;
    r->kind = EXR_SRC_MEMORY;
    r->mem = buf;
    r->mem_size = (size_t)src->total_size;
    r->free_mem = 1;
    *out = r;
    return EXR_SUCCESS;
}

void exr_reader_close(exr_reader *r) {
    const exr_allocator *a;
    int32_t i;
    if (!r) return;
    a = &r->alloc;
    if (r->parts) {
        for (i = 0; i < r->num_parts; ++i) {
            exr_int_part *p = &r->parts[i];
            exr_free(a, p->offsets);
            exr_free(a, p->level_width);
            exr_free(a, p->level_height);
            exr_free(a, p->level_x_tiles);
            exr_free(a, p->level_y_tiles);
            exr_header_free(a, &p->header);
        }
        exr_free(a, r->parts);
    }
    if (r->free_mem) exr_free(a, (void *)r->mem);
    exr_free(a, r);
}

/* ============================================================================
 * Random access
 * ========================================================================== */

exr_result exr_reader_fetch(exr_reader *r, uint64_t offset, size_t size,
                            void *scratch, const uint8_t **out_ptr) {
    if (r->kind == EXR_SRC_MEMORY) {
        if (offset > (uint64_t)r->mem_size) return EXR_ERROR_CORRUPT;
        if (size > r->mem_size - (size_t)offset) return EXR_ERROR_CORRUPT;
        *out_ptr = r->mem + (size_t)offset;
        return EXR_SUCCESS;
    }
    (void)scratch; /* callback path: Phase 9 */
    return EXR_ERROR_UNSUPPORTED;
}

/* ============================================================================
 * Header parsing helpers
 * ========================================================================== */

typedef struct {
    const uint8_t *p;
    size_t size;
    size_t pos;
} cursor;

static int cur_need(const cursor *c, size_t n) {
    return (c->pos <= c->size) && (n <= c->size - c->pos);
}

/* Read a NUL-terminated string; returns pointer + length (excludes NUL). */
static exr_result cur_string(cursor *c, const char **out, size_t *out_len) {
    size_t start = c->pos;
    while (c->pos < c->size && c->p[c->pos] != 0) c->pos++;
    if (c->pos >= c->size) return EXR_ERROR_CORRUPT; /* unterminated */
    *out = (const char *)(c->p + start);
    *out_len = c->pos - start;
    c->pos++; /* consume NUL */
    return EXR_SUCCESS;
}

static exr_result attr_append(const exr_allocator *a, exr_attr_list *list,
                              const char *name, size_t name_len,
                              const char *type, size_t type_len,
                              const uint8_t *data, uint32_t size) {
    exr_attr *slot;
    if (list->count >= EXR_MAX_ATTRIBUTES) return EXR_ERROR_CORRUPT;
    if (list->count == list->capacity) {
        uint32_t ncap = list->capacity ? list->capacity * 2 : 8;
        exr_attr *ni =
            (exr_attr *)exr_calloc(a, ncap, sizeof(exr_attr));
        if (!ni) return EXR_ERROR_OUT_OF_MEMORY;
        if (list->items) {
            memcpy(ni, list->items, list->count * sizeof(exr_attr));
            exr_free(a, list->items);
        }
        list->items = ni;
        list->capacity = ncap;
    }
    slot = &list->items[list->count];
    slot->name = (char *)exr_malloc(a, name_len + 1);
    slot->type_name = (char *)exr_malloc(a, type_len + 1);
    slot->data = (uint8_t *)exr_malloc(a, size ? size : 1);
    if (!slot->name || !slot->type_name || !slot->data) {
        exr_free(a, slot->name);
        exr_free(a, slot->type_name);
        exr_free(a, slot->data);
        memset(slot, 0, sizeof(*slot));
        return EXR_ERROR_OUT_OF_MEMORY;
    }
    memcpy(slot->name, name, name_len);
    slot->name[name_len] = '\0';
    memcpy(slot->type_name, type, type_len);
    slot->type_name[type_len] = '\0';
    if (size) memcpy(slot->data, data, size);
    slot->size = size;
    list->count++;
    return EXR_SUCCESS;
}

/* Parse one header's attribute list (until empty-name terminator). */
static exr_result parse_one_header(exr_reader *r, cursor *c,
                                   exr_int_part *part) {
    const exr_allocator *a = &r->alloc;
    exr_attr_list *list;

    list = (exr_attr_list *)exr_calloc(a, 1, sizeof(*list));
    if (!list) return EXR_ERROR_OUT_OF_MEMORY;
    part->header.attrs = list;

    for (;;) {
        const char *name, *type;
        size_t name_len, type_len;
        int32_t sz;
        exr_result rc;

        if (!cur_need(c, 1)) return EXR_ERROR_CORRUPT;
        if (c->p[c->pos] == 0) {
            c->pos++; /* empty name -> end of header */
            break;
        }
        rc = cur_string(c, &name, &name_len);
        if (!EXR_OK(rc)) return rc;
        if (name_len >= EXR_MAX_NAME) return EXR_ERROR_CORRUPT;
        rc = cur_string(c, &type, &type_len);
        if (!EXR_OK(rc)) return rc;
        if (type_len >= EXR_MAX_NAME) return EXR_ERROR_CORRUPT;
        if (!cur_need(c, 4)) return EXR_ERROR_CORRUPT;
        sz = exr_rd_i32(c->p + c->pos);
        c->pos += 4;
        if (sz < 0 || (uint32_t)sz > EXR_MAX_ATTR_SIZE) return EXR_ERROR_CORRUPT;
        if (!cur_need(c, (size_t)sz)) return EXR_ERROR_CORRUPT;
        rc = attr_append(a, list, name, name_len, type, type_len,
                         c->p + c->pos, (uint32_t)sz);
        if (!EXR_OK(rc)) return rc;
        c->pos += (size_t)sz;
    }
    return EXR_SUCCESS;
}

/* Interpret the standard attributes into header fields. */
static exr_result interpret_header(exr_reader *r, exr_int_part *part) {
    const exr_allocator *a = &r->alloc;
    exr_header *h = &part->header;
    const exr_attr *at;
    int has_data_window = 0;

    /* defaults */
    h->compression = EXR_COMPRESSION_NONE;
    h->line_order = EXR_LINEORDER_INCREASING_Y;
    h->pixel_aspect_ratio = 1.0f;
    h->screen_window_width = 1.0f;
    h->screen_window_center_x = 0.0f;
    h->screen_window_center_y = 0.0f;
    h->name[0] = '\0';

    /* channels */
    at = exr_attr_find(h->attrs, "channels");
    if (!at) return EXR_ERROR_CORRUPT;
    {
        exr_result rc = exr_parse_chlist(a, at->data, at->size, &h->channels,
                                         &h->num_channels);
        if (!EXR_OK(rc)) return rc;
    }

    at = exr_attr_find(h->attrs, "compression");
    if (at && at->size >= 1) {
        if (at->data[0] > 9) return EXR_ERROR_CORRUPT;
        h->compression = (exr_compression)at->data[0];
    }

    at = exr_attr_find(h->attrs, "dataWindow");
    if (at && at->size >= 16) {
        h->data_window.min_x = exr_rd_i32(at->data);
        h->data_window.min_y = exr_rd_i32(at->data + 4);
        h->data_window.max_x = exr_rd_i32(at->data + 8);
        h->data_window.max_y = exr_rd_i32(at->data + 12);
        has_data_window = 1;
    }
    if (!has_data_window) return EXR_ERROR_CORRUPT;

    at = exr_attr_find(h->attrs, "displayWindow");
    if (at && at->size >= 16) {
        h->display_window.min_x = exr_rd_i32(at->data);
        h->display_window.min_y = exr_rd_i32(at->data + 4);
        h->display_window.max_x = exr_rd_i32(at->data + 8);
        h->display_window.max_y = exr_rd_i32(at->data + 12);
    } else {
        h->display_window = h->data_window;
    }

    at = exr_attr_find(h->attrs, "lineOrder");
    if (at && at->size >= 1 && at->data[0] <= 2)
        h->line_order = (exr_line_order)at->data[0];

    at = exr_attr_find(h->attrs, "pixelAspectRatio");
    if (at && at->size >= 4) h->pixel_aspect_ratio = exr_rd_f32(at->data);

    at = exr_attr_find(h->attrs, "screenWindowCenter");
    if (at && at->size >= 8) {
        h->screen_window_center_x = exr_rd_f32(at->data);
        h->screen_window_center_y = exr_rd_f32(at->data + 4);
    }

    at = exr_attr_find(h->attrs, "screenWindowWidth");
    if (at && at->size >= 4) h->screen_window_width = exr_rd_f32(at->data);

    /* tiles */
    at = exr_attr_find(h->attrs, "tiles");
    if (at && at->size >= 9) {
        h->tiled = 1;
        h->tile_x_size = exr_rd_u32(at->data);
        h->tile_y_size = exr_rd_u32(at->data + 4);
        h->level_mode = (exr_tile_level_mode)(at->data[8] & 0x0f);
        h->rounding_mode = (exr_tile_rounding_mode)((at->data[8] >> 4) & 0x0f);
        if (h->level_mode > EXR_TILE_RIPMAP_LEVELS) return EXR_ERROR_CORRUPT;
    }

    /* name */
    at = exr_attr_find(h->attrs, "name");
    if (at && at->size > 0) {
        uint32_t n = at->size;
        if (n >= EXR_MAX_NAME) n = EXR_MAX_NAME - 1;
        memcpy(h->name, at->data, n);
        h->name[n] = '\0';
    }

    /* part type: from "type" attribute, else from version flags */
    at = exr_attr_find(h->attrs, "type");
    if (at && at->size > 0) {
        const char *t = (const char *)at->data;
        size_t n = at->size;
        if (n == 13 && memcmp(t, "scanlineimage", 13) == 0)
            h->part_type = EXR_PART_SCANLINE;
        else if (n == 10 && memcmp(t, "tiledimage", 10) == 0) {
            h->part_type = EXR_PART_TILED;
            h->tiled = 1;
        } else if (n == 12 && memcmp(t, "deepscanline", 12) == 0)
            h->part_type = EXR_PART_DEEP_SCANLINE;
        else if (n == 8 && memcmp(t, "deeptile", 8) == 0) {
            h->part_type = EXR_PART_DEEP_TILED;
            h->tiled = 1;
        } else
            h->part_type = h->tiled ? EXR_PART_TILED : EXR_PART_SCANLINE;
    } else {
        if (r->is_deep)
            h->part_type =
                h->tiled ? EXR_PART_DEEP_TILED : EXR_PART_DEEP_SCANLINE;
        else
            h->part_type = h->tiled ? EXR_PART_TILED : EXR_PART_SCANLINE;
    }

    /* geometry */
    {
        int64_t w = (int64_t)h->data_window.max_x - h->data_window.min_x + 1;
        int64_t hh = (int64_t)h->data_window.max_y - h->data_window.min_y + 1;
        if (w <= 0 || hh <= 0 || w > EXR_MAX_DIMENSION || hh > EXR_MAX_DIMENSION)
            return EXR_ERROR_CORRUPT;
        part->width = (int32_t)w;
        part->height = (int32_t)hh;
    }
    return EXR_SUCCESS;
}

/* Dimensions and tile counts of one tile level. */
static void tiled_level_size(const exr_int_part *p, int lx, int ly, int *w,
                             int *h, int *nxt, int *nyt) {
    int lw = p->width, lh = p->height, i;
    int up = (p->header.rounding_mode == EXR_TILE_ROUND_UP);
    int tx = (int)p->header.tile_x_size, ty = (int)p->header.tile_y_size;
    if (p->header.level_mode == EXR_TILE_MIPMAP_LEVELS) {
        for (i = 0; i < lx; ++i) {
            lw = up ? (lw + 1) / 2 : lw / 2;
            lh = up ? (lh + 1) / 2 : lh / 2;
            if (lw < 1) lw = 1;
            if (lh < 1) lh = 1;
        }
    } else if (p->header.level_mode == EXR_TILE_RIPMAP_LEVELS) {
        for (i = 0; i < lx; ++i) { lw = up ? (lw + 1) / 2 : lw / 2; if (lw < 1) lw = 1; }
        for (i = 0; i < ly; ++i) { lh = up ? (lh + 1) / 2 : lh / 2; if (lh < 1) lh = 1; }
    }
    if (w) *w = lw;
    if (h) *h = lh;
    if (tx > 0 && nxt) *nxt = (lw + tx - 1) / tx;
    if (ty > 0 && nyt) *nyt = (lh + ty - 1) / ty;
}

/* Total chunk count of a tiled part; also fills num_x/y_levels. */
static exr_result tiled_total_chunks(exr_int_part *p, uint32_t *out) {
    int up = (p->header.rounding_mode == EXR_TILE_ROUND_UP);
    int tx = (int)p->header.tile_x_size, ty = (int)p->header.tile_y_size;
    uint64_t total = 0;
    if (tx <= 0 || ty <= 0) return EXR_ERROR_CORRUPT;

    p->num_x_levels = 1;
    p->num_y_levels = 1;

    if (p->header.level_mode == EXR_TILE_ONE_LEVEL) {
        int nx = (p->width + tx - 1) / tx, ny = (p->height + ty - 1) / ty;
        total = (uint64_t)nx * (uint64_t)ny;
    } else if (p->header.level_mode == EXR_TILE_MIPMAP_LEVELS) {
        int w = p->width, h = p->height, levels = 0;
        for (;;) {
            int nx = (w + tx - 1) / tx, ny = (h + ty - 1) / ty;
            total += (uint64_t)nx * (uint64_t)ny;
            levels++;
            if (w <= 1 && h <= 1) break;
            w = up ? (w + 1) / 2 : w / 2;
            h = up ? (h + 1) / 2 : h / 2;
            if (w < 1) w = 1;
            if (h < 1) h = 1;
        }
        p->num_x_levels = (int32_t)levels;
        p->num_y_levels = (int32_t)levels;
    } else { /* RIPMAP */
        int w = p->width, h = p->height, nxl = 1, nyl = 1, lx, ly;
        while (w > 1) { nxl++; w = up ? (w + 1) / 2 : w / 2; if (w < 1) w = 1; }
        while (h > 1) { nyl++; h = up ? (h + 1) / 2 : h / 2; if (h < 1) h = 1; }
        p->num_x_levels = (int32_t)nxl;
        p->num_y_levels = (int32_t)nyl;
        for (ly = 0; ly < nyl; ++ly)
            for (lx = 0; lx < nxl; ++lx) {
                int nxt, nyt;
                tiled_level_size(p, lx, ly, NULL, NULL, &nxt, &nyt);
                total += (uint64_t)nxt * (uint64_t)nyt;
            }
    }
    if (total > 0xffffffffu) return EXR_ERROR_CORRUPT;
    *out = (uint32_t)total;
    return EXR_SUCCESS;
}

/* Number of chunks (offset-table entries) for a part. */
static exr_result part_chunk_count(exr_reader *r, exr_int_part *part,
                                   uint32_t *out_count) {
    const exr_header *h = &part->header;
    const exr_attr *cc = exr_attr_find(h->attrs, "chunkCount");
    (void)r;
    if (h->part_type == EXR_PART_TILED || h->part_type == EXR_PART_DEEP_TILED) {
        /* Tiled offset tables always have an entry per tile across all levels;
         * compute it (and the level counts) directly. */
        return tiled_total_chunks(part, out_count);
    }
    if (cc && cc->size >= 4) {
        int32_t v = exr_rd_i32(cc->data);
        if (v < 0) return EXR_ERROR_CORRUPT;
        *out_count = (uint32_t)v;
        return EXR_SUCCESS;
    }
    if (h->part_type == EXR_PART_SCANLINE ||
        h->part_type == EXR_PART_DEEP_SCANLINE) {
        int lpb = exr_lines_per_block(h->compression);
        *out_count = (uint32_t)(((int64_t)part->height + lpb - 1) / lpb);
        return EXR_SUCCESS;
    }
    return EXR_ERROR_UNSUPPORTED;
}

/* ============================================================================
 * Parse header (public)
 * ========================================================================== */

exr_result exr_reader_parse_header(exr_reader *r) {
    cursor c;
    uint32_t ver;
    int vnum;
    int32_t i;
    exr_result rc;
    int32_t cap = 0;

    if (!r) return EXR_ERROR_INVALID_ARGUMENT;
    if (r->parsed) return EXR_SUCCESS;
    if (r->kind != EXR_SRC_MEMORY) return EXR_ERROR_UNSUPPORTED;

    if (r->mem_size < 8) return EXR_ERROR_INVALID_FILE;
    if (exr_rd_u32(r->mem) != EXR_MAGIC) return EXR_ERROR_INVALID_FILE;
    ver = exr_rd_u32(r->mem + 4);
    vnum = (int)(ver & 0xff);
    if (vnum > EXR_VERSION_NUMBER) return EXR_ERROR_UNSUPPORTED;

    r->version_flags = ver;
    r->is_tiled = (ver & EXR_VERSION_FLAG_TILED) != 0;
    r->long_names = (ver & EXR_VERSION_FLAG_LONG_NAMES) != 0;
    r->is_deep = (ver & EXR_VERSION_FLAG_NON_IMAGE) != 0;
    r->is_multipart = (ver & EXR_VERSION_FLAG_MULTIPART) != 0;

    c.p = r->mem;
    c.size = r->mem_size;
    c.pos = 8;

    /* Parse part headers. */
    for (;;) {
        exr_int_part *np;
        if (r->is_multipart) {
            if (!cur_need(&c, 1)) {
                rc = EXR_ERROR_CORRUPT;
                goto fail;
            }
            if (c.p[c.pos] == 0) {
                c.pos++; /* terminating empty header */
                break;
            }
        }
        if (r->num_parts >= EXR_MAX_PARTS) {
            rc = EXR_ERROR_CORRUPT;
            goto fail;
        }
        if (r->num_parts == cap) {
            int32_t ncap = cap ? cap * 2 : 1;
            exr_int_part *npb = (exr_int_part *)exr_calloc(
                &r->alloc, (size_t)ncap, sizeof(exr_int_part));
            if (!npb) {
                rc = EXR_ERROR_OUT_OF_MEMORY;
                goto fail;
            }
            if (r->parts) {
                memcpy(npb, r->parts,
                       (size_t)r->num_parts * sizeof(exr_int_part));
                exr_free(&r->alloc, r->parts);
            }
            r->parts = npb;
            cap = ncap;
        }
        np = &r->parts[r->num_parts];
        memset(np, 0, sizeof(*np));
        rc = parse_one_header(r, &c, np);
        if (!EXR_OK(rc)) {
            r->num_parts++; /* so cleanup frees its attrs */
            goto fail;
        }
        rc = interpret_header(r, np);
        if (!EXR_OK(rc)) {
            r->num_parts++;
            goto fail;
        }
        r->num_parts++;
        if (!r->is_multipart) break;
    }

    if (r->num_parts == 0) {
        rc = EXR_ERROR_CORRUPT;
        goto fail;
    }

    /* Offset tables, in part order. */
    for (i = 0; i < r->num_parts; ++i) {
        exr_int_part *p = &r->parts[i];
        uint32_t n, k;
        size_t need;
        rc = part_chunk_count(r, p, &n);
        if (!EXR_OK(rc)) goto fail;
        if (n > r->mem_size / 8) {
            rc = EXR_ERROR_CORRUPT;
            goto fail;
        }
        if (exr_mul_ovf((size_t)n, 8u, &need)) {
            rc = EXR_ERROR_CORRUPT;
            goto fail;
        }
        if (!cur_need(&c, need)) {
            rc = EXR_ERROR_CORRUPT;
            goto fail;
        }
        p->offsets = (uint64_t *)exr_calloc(&r->alloc, n ? n : 1, sizeof(uint64_t));
        if (!p->offsets) {
            rc = EXR_ERROR_OUT_OF_MEMORY;
            goto fail;
        }
        for (k = 0; k < n; ++k)
            p->offsets[k] = exr_rd_u64(c.p + c.pos + (size_t)k * 8);
        p->num_chunks = n;
        c.pos += need;
    }

    r->parsed = 1;
    return EXR_SUCCESS;

fail:
    return rc;
}

int32_t exr_reader_num_parts(const exr_reader *r) {
    return r ? r->num_parts : 0;
}

const exr_header *exr_reader_part_header(const exr_reader *r, int32_t part) {
    if (!r || part < 0 || part >= r->num_parts) return NULL;
    return &r->parts[part].header;
}

/* ============================================================================
 * Scanline read
 * ========================================================================== */

static int64_t floordiv64(int64_t a, int64_t b) {
    int64_t q = a / b, rem = a % b;
    if ((rem != 0) && ((rem < 0) != (b < 0))) q--;
    return q;
}
static int nsamp(int lo, int hi, int s) {
    if (s <= 1) return hi - lo + 1;
    return (int)(floordiv64(hi, s) - floordiv64((int64_t)lo - 1, s));
}

/* Per-channel sampled width/height of a part's data window. */
static void channel_dims(const exr_header *h, int c, int *cw, int *ch_) {
    int xs = h->channels[c].x_sampling, ys = h->channels[c].y_sampling;
    *cw = nsamp(h->data_window.min_x, h->data_window.max_x, xs);
    *ch_ = nsamp(h->data_window.min_y, h->data_window.max_y, ys);
}

/* Scatter one decompressed scanline block into the planar channel buffers. */
static exr_result scatter_scanline_block(const exr_header *h, void **images,
                                         int32_t y0, int32_t nlines,
                                         const uint8_t *block, size_t block_size) {
    int xmin = h->data_window.min_x, xmax = h->data_window.max_x;
    int ymin = h->data_window.min_y;
    size_t off = 0;
    int line, c;
    for (line = 0; line < nlines; ++line) {
        int yy = y0 + line;
        for (c = 0; c < h->num_channels; ++c) {
            int xs = h->channels[c].x_sampling, ys = h->channels[c].y_sampling;
            size_t ps = exr_pixel_size(h->channels[c].pixel_type);
            int nx, row, cw, ch_;
            size_t bytes;
            if ((yy % ys) != 0) continue;
            nx = nsamp(xmin, xmax, xs);
            if (nx <= 0) continue;
            channel_dims(h, c, &cw, &ch_);
            row = nsamp(ymin, yy, ys) - 1;
            if (row < 0 || row >= ch_) return EXR_ERROR_CORRUPT;
            bytes = (size_t)nx * ps;
            if (off + bytes > block_size) return EXR_ERROR_CORRUPT;
            memcpy((uint8_t *)images[c] + (size_t)row * (size_t)cw * ps,
                   block + off, bytes);
            off += bytes;
        }
    }
    return EXR_SUCCESS;
}

static exr_result read_scanline_part(exr_reader *r, exr_int_part *p,
                                     int32_t part_idx, exr_part *out) {
    const exr_allocator *a = &r->alloc;
    const exr_header *h = &p->header;
    int lpb = exr_lines_per_block(h->compression);
    int ymin = h->data_window.min_y, ymax = h->data_window.max_y;
    uint8_t *block = NULL;
    size_t block_cap = 0;
    exr_result rc = EXR_SUCCESS;
    uint32_t ci;
    int c;

    /* allocate planar channel buffers */
    out->images = (void **)exr_calloc(a, (size_t)h->num_channels, sizeof(void *));
    if (!out->images) return EXR_ERROR_OUT_OF_MEMORY;
    for (c = 0; c < h->num_channels; ++c) {
        int cw, ch_;
        size_t n, bytes;
        channel_dims(h, c, &cw, &ch_);
        if (cw < 0) cw = 0;
        if (ch_ < 0) ch_ = 0;
        if (exr_mul_ovf((size_t)cw, (size_t)ch_, &n) ||
            exr_mul_ovf(n, exr_pixel_size(h->channels[c].pixel_type), &bytes))
            return EXR_ERROR_CORRUPT;
        out->images[c] = exr_calloc(a, bytes ? bytes : 1, 1);
        if (!out->images[c]) return EXR_ERROR_OUT_OF_MEMORY;
    }

    for (ci = 0; ci < p->num_chunks; ++ci) {
        uint64_t off = p->offsets[ci];
        const uint8_t *hdr;
        int32_t y0, data_size, nlines;
        size_t hdr_size = r->is_multipart ? 12 : 8; /* [part]+y+size */
        size_t want;
        const uint8_t *cdata;
        size_t dst_size;
        exr_codec_ctx ctx;

        rc = exr_reader_fetch(r, off, hdr_size, NULL, &hdr);
        if (!EXR_OK(rc)) goto done;
        if (r->is_multipart) {
            if (exr_rd_i32(hdr) != part_idx) {
                rc = EXR_ERROR_CORRUPT;
                goto done;
            }
            hdr += 4;
        }
        y0 = exr_rd_i32(hdr);
        data_size = exr_rd_i32(hdr + 4);
        if (data_size < 0 || y0 < ymin || y0 > ymax) {
            rc = EXR_ERROR_CORRUPT;
            goto done;
        }
        nlines = lpb;
        if (y0 + nlines - 1 > ymax) nlines = ymax - y0 + 1;

        rc = exr_block_uncompressed_size(h->channels, h->num_channels,
                                         h->data_window.min_x, y0, p->width,
                                         nlines, &dst_size);
        if (!EXR_OK(rc)) goto done;

        if (exr_add_ovf((size_t)off, hdr_size, &want)) {
            rc = EXR_ERROR_CORRUPT;
            goto done;
        }
        rc = exr_reader_fetch(r, want, (size_t)data_size, NULL, &cdata);
        if (!EXR_OK(rc)) goto done;

        if (dst_size > block_cap) {
            exr_free(a, block);
            block = (uint8_t *)exr_malloc(a, dst_size ? dst_size : 1);
            if (!block) {
                rc = EXR_ERROR_OUT_OF_MEMORY;
                goto done;
            }
            block_cap = dst_size;
        }

        ctx.alloc = a;
        ctx.compression = h->compression;
        ctx.channels = h->channels;
        ctx.num_channels = h->num_channels;
        ctx.x = h->data_window.min_x;
        ctx.y = y0;
        ctx.width = p->width;
        ctx.num_lines = nlines;
        rc = exr_decompress_block(&ctx, cdata, (size_t)data_size, block,
                                  dst_size);
        if (!EXR_OK(rc)) goto done;

        rc = scatter_scanline_block(h, out->images, y0, nlines, block, dst_size);
        if (!EXR_OK(rc)) goto done;
    }

done:
    exr_free(a, block);
    return rc;
}

/* Scatter one decompressed tile block into the planar channel buffers. */
static exr_result scatter_tile_block(const exr_header *h, void **images,
                                     int abs_x0, int abs_y0, int tile_w,
                                     int tile_h, const uint8_t *block,
                                     size_t block_size) {
    int xmin = h->data_window.min_x, ymin = h->data_window.min_y;
    size_t off = 0;
    int row, c;
    for (row = 0; row < tile_h; ++row) {
        int yy = abs_y0 + row;
        for (c = 0; c < h->num_channels; ++c) {
            int xs = h->channels[c].x_sampling, ys = h->channels[c].y_sampling;
            size_t ps = exr_pixel_size(h->channels[c].pixel_type);
            int tnx, cw, ch_, ch_row, ch_col;
            size_t bytes;
            if ((yy % ys) != 0) continue;
            tnx = nsamp(abs_x0, abs_x0 + tile_w - 1, xs);
            if (tnx <= 0) continue;
            channel_dims(h, c, &cw, &ch_);
            ch_row = nsamp(ymin, yy, ys) - 1;
            ch_col = nsamp(xmin, abs_x0 - 1, xs);
            if (ch_row < 0 || ch_row >= ch_) return EXR_ERROR_CORRUPT;
            if (ch_col < 0 || ch_col + tnx > cw) return EXR_ERROR_CORRUPT;
            bytes = (size_t)tnx * ps;
            if (off + bytes > block_size) return EXR_ERROR_CORRUPT;
            memcpy((uint8_t *)images[c] +
                       ((size_t)ch_row * cw + ch_col) * ps,
                   block + off, bytes);
            off += bytes;
        }
    }
    return EXR_SUCCESS;
}

static exr_result read_tiled_part(exr_reader *r, exr_int_part *p,
                                  int32_t part_idx, exr_part *out) {
    const exr_allocator *a = &r->alloc;
    const exr_header *h = &p->header;
    int tx = (int)h->tile_x_size, ty = (int)h->tile_y_size;
    int nxt, nyt, txi, tyi, c;
    uint8_t *block = NULL;
    size_t block_cap = 0;
    exr_result rc = EXR_SUCCESS;

    if (tx <= 0 || ty <= 0) return EXR_ERROR_CORRUPT;
    nxt = (p->width + tx - 1) / tx;
    nyt = (p->height + ty - 1) / ty;

    out->images = (void **)exr_calloc(a, (size_t)h->num_channels, sizeof(void *));
    if (!out->images) return EXR_ERROR_OUT_OF_MEMORY;
    for (c = 0; c < h->num_channels; ++c) {
        int cw, ch_;
        size_t n, bytes;
        channel_dims(h, c, &cw, &ch_);
        if (cw < 0) cw = 0;
        if (ch_ < 0) ch_ = 0;
        if (exr_mul_ovf((size_t)cw, (size_t)ch_, &n) ||
            exr_mul_ovf(n, exr_pixel_size(h->channels[c].pixel_type), &bytes))
            return EXR_ERROR_CORRUPT;
        out->images[c] = exr_calloc(a, bytes ? bytes : 1, 1);
        if (!out->images[c]) return EXR_ERROR_OUT_OF_MEMORY;
    }

    for (tyi = 0; tyi < nyt; ++tyi) {
        for (txi = 0; txi < nxt; ++txi) {
            uint32_t idx = (uint32_t)tyi * (uint32_t)nxt + (uint32_t)txi;
            uint64_t off;
            const uint8_t *hdr, *cdata;
            size_t hdr_size = r->is_multipart ? 24 : 20;
            int32_t htx, hty, hlx, hly, dsize;
            int x0, y0, tile_w, tile_h, abs_x0, abs_y0;
            size_t dst_size, want;
            exr_codec_ctx ctx;

            if (idx >= p->num_chunks) { rc = EXR_ERROR_CORRUPT; goto done; }
            off = p->offsets[idx];
            rc = exr_reader_fetch(r, off, hdr_size, NULL, &hdr);
            if (!EXR_OK(rc)) goto done;
            if (r->is_multipart) {
                if (exr_rd_i32(hdr) != part_idx) { rc = EXR_ERROR_CORRUPT; goto done; }
                hdr += 4;
            }
            htx = exr_rd_i32(hdr);
            hty = exr_rd_i32(hdr + 4);
            hlx = exr_rd_i32(hdr + 8);
            hly = exr_rd_i32(hdr + 12);
            dsize = exr_rd_i32(hdr + 16);
            if (hlx != 0 || hly != 0 || htx != txi || hty != tyi || dsize < 0) {
                rc = EXR_ERROR_CORRUPT;
                goto done;
            }

            x0 = txi * tx;
            y0 = tyi * ty;
            tile_w = (tx < p->width - x0) ? tx : (p->width - x0);
            tile_h = (ty < p->height - y0) ? ty : (p->height - y0);
            abs_x0 = h->data_window.min_x + x0;
            abs_y0 = h->data_window.min_y + y0;

            rc = exr_block_uncompressed_size(h->channels, h->num_channels,
                                             abs_x0, abs_y0, tile_w, tile_h,
                                             &dst_size);
            if (!EXR_OK(rc)) goto done;

            if (exr_add_ovf((size_t)off, hdr_size, &want)) { rc = EXR_ERROR_CORRUPT; goto done; }
            rc = exr_reader_fetch(r, want, (size_t)dsize, NULL, &cdata);
            if (!EXR_OK(rc)) goto done;

            if (dst_size > block_cap) {
                exr_free(a, block);
                block = (uint8_t *)exr_malloc(a, dst_size ? dst_size : 1);
                if (!block) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
                block_cap = dst_size;
            }

            ctx.alloc = a;
            ctx.compression = h->compression;
            ctx.channels = h->channels;
            ctx.num_channels = h->num_channels;
            ctx.x = abs_x0;
            ctx.y = abs_y0;
            ctx.width = tile_w;
            ctx.num_lines = tile_h;
            rc = exr_decompress_block(&ctx, cdata, (size_t)dsize, block, dst_size);
            if (!EXR_OK(rc)) goto done;

            rc = scatter_tile_block(h, out->images, abs_x0, abs_y0, tile_w,
                                    tile_h, block, dst_size);
            if (!EXR_OK(rc)) goto done;
        }
    }

done:
    exr_free(a, block);
    return rc;
}

exr_result exr_reader_read_part(exr_reader *r, int32_t part, exr_part *out) {
    exr_int_part *p;
    exr_result rc;
    if (!r || !out) return EXR_ERROR_INVALID_ARGUMENT;
    rc = exr_reader_parse_header(r);
    if (!EXR_OK(rc)) return rc;
    if (part < 0 || part >= r->num_parts) return EXR_ERROR_INVALID_ARGUMENT;
    p = &r->parts[part];

    memset(out, 0, sizeof(*out));
    rc = exr_header_copy(&r->alloc, &out->header, &p->header);
    if (!EXR_OK(rc)) return rc;
    out->width = p->width;
    out->height = p->height;

    switch (p->header.part_type) {
    case EXR_PART_SCANLINE:
        rc = read_scanline_part(r, p, part, out);
        break;
    case EXR_PART_TILED:
        rc = read_tiled_part(r, p, part, out);
        break;
    case EXR_PART_DEEP_SCANLINE:
    case EXR_PART_DEEP_TILED:
    default:
        rc = EXR_ERROR_UNSUPPORTED; /* Phase 6 */
        break;
    }
    if (!EXR_OK(rc)) {
        /* free partial channel buffers; header freed by caller via image_free */
        if (out->images) {
            int c;
            for (c = 0; c < out->header.num_channels; ++c)
                exr_free(&r->alloc, out->images[c]);
            exr_free(&r->alloc, out->images);
            out->images = NULL;
        }
    }
    return rc;
}

exr_result exr_reader_read_scanlines(exr_reader *r, int32_t part,
                                     int32_t y_start, int32_t y_count,
                                     exr_part *out) {
    (void)y_start;
    (void)y_count;
    /* Phase 5 exposes partial reads; for now read the whole part. */
    return exr_reader_read_part(r, part, out);
}

exr_result exr_reader_read_tile(exr_reader *r, int32_t part, int32_t tile_x,
                                int32_t tile_y, int32_t level_x, int32_t level_y,
                                exr_part *out) {
    (void)r;
    (void)part;
    (void)tile_x;
    (void)tile_y;
    (void)level_x;
    (void)level_y;
    (void)out;
    return EXR_ERROR_UNSUPPORTED; /* Phase 5 */
}

/* ============================================================================
 * Streaming suspend/resume (Phase 9 stubs)
 * ========================================================================== */

exr_result exr_reader_pending(const exr_reader *r, exr_pending_read *out) {
    if (!r || !out) return EXR_ERROR_INVALID_ARGUMENT;
    if (!r->have_pending) return EXR_ERROR_INVALID_ARGUMENT;
    *out = r->pending;
    return EXR_SUCCESS;
}

exr_result exr_reader_supply(exr_reader *r, const void *data, size_t size) {
    (void)r;
    (void)data;
    (void)size;
    return EXR_ERROR_UNSUPPORTED;
}
