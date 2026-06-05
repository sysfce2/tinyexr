/*
 * TinyEXR - mid-level writer + high-level save.
 *
 * Phase 8: scanline (single + multipart) output for NONE / RLE / ZIP / ZIPS.
 * Tiled and lossy-codec writing build on the same serializer.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

#include <stdio.h>
#include <stdlib.h>

/* ---- growable output buffer ---------------------------------------------- */

typedef struct {
    const exr_allocator *a;
    uint8_t *data;
    size_t len, cap;
    int err;
} obuf;

static void ob_reserve(obuf *b, size_t extra) {
    size_t need;
    if (b->err) return;
    if (exr_add_ovf(b->len, extra, &need)) { b->err = 1; return; }
    if (need <= b->cap) return;
    {
        size_t ncap = b->cap ? b->cap : 256;
        uint8_t *nd;
        while (ncap < need) {
            if (ncap > (SIZE_MAX / 2)) { b->err = 1; return; }
            ncap *= 2;
        }
        nd = (uint8_t *)exr_malloc(b->a, ncap);
        if (!nd) { b->err = 1; return; }
        if (b->data) {
            memcpy(nd, b->data, b->len);
            exr_free(b->a, b->data);
        }
        b->data = nd;
        b->cap = ncap;
    }
}
static void ob_bytes(obuf *b, const void *p, size_t n) {
    ob_reserve(b, n);
    if (b->err) return;
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void ob_u8(obuf *b, uint8_t v) { ob_bytes(b, &v, 1); }
static void ob_u32(obuf *b, uint32_t v) {
    uint8_t t[4];
    exr_wr_u32(t, v);
    ob_bytes(b, t, 4);
}
static void ob_i32(obuf *b, int32_t v) { ob_u32(b, (uint32_t)v); }
static void ob_u64(obuf *b, uint64_t v) {
    uint8_t t[8];
    exr_wr_u64(t, v);
    ob_bytes(b, t, 8);
}
static void ob_cstr(obuf *b, const char *s) { ob_bytes(b, s, strlen(s) + 1); }

/* attribute: name\0 type\0 size(i32) data */
static void ob_attr(obuf *b, const char *name, const char *type,
                    const void *data, uint32_t size) {
    ob_cstr(b, name);
    ob_cstr(b, type);
    ob_i32(b, (int32_t)size);
    ob_bytes(b, data, size);
}

/* ---- channel sort -------------------------------------------------------- */

static void sort_channels(const exr_header *h, int *order) {
    int i, j, n = h->num_channels;
    for (i = 0; i < n; ++i) order[i] = i;
    for (i = 1; i < n; ++i) { /* insertion sort by name */
        int v = order[i];
        for (j = i - 1; j >= 0 &&
                        strcmp(h->channels[order[j]].name, h->channels[v].name) > 0;
             --j)
            order[j + 1] = order[j];
        order[j + 1] = v;
    }
}

/* ---- header serialization ------------------------------------------------ */

static const char *part_type_string(exr_part_type t) {
    switch (t) {
    case EXR_PART_SCANLINE: return "scanlineimage";
    case EXR_PART_TILED: return "tiledimage";
    case EXR_PART_DEEP_SCANLINE: return "deepscanline";
    case EXR_PART_DEEP_TILED: return "deeptile";
    }
    return "scanlineimage";
}

static void write_box2i(obuf *b, const char *name, const exr_box2i *w) {
    uint8_t t[16];
    exr_wr_i32(t, w->min_x);
    exr_wr_i32(t + 4, w->min_y);
    exr_wr_i32(t + 8, w->max_x);
    exr_wr_i32(t + 12, w->max_y);
    ob_attr(b, name, "box2i", t, 16);
}

static void write_header(obuf *b, const exr_header *h, const int *order,
                         exr_compression comp, int multipart,
                         uint32_t chunk_count, int32_t max_samples) {
    /* channels */
    {
        obuf cl;
        int i;
        memset(&cl, 0, sizeof(cl));
        cl.a = b->a;
        for (i = 0; i < h->num_channels; ++i) {
            const exr_channel *c = &h->channels[order[i]];
            uint8_t t[16];
            ob_cstr(&cl, c->name);
            exr_wr_i32(t, (int32_t)c->pixel_type);
            t[4] = c->p_linear;
            t[5] = t[6] = t[7] = 0;
            exr_wr_i32(t + 8, c->x_sampling);
            exr_wr_i32(t + 12, c->y_sampling);
            ob_bytes(&cl, t, 16);
        }
        ob_u8(&cl, 0); /* terminator */
        if (cl.err) { b->err = 1; }
        else ob_attr(b, "channels", "chlist", cl.data, (uint32_t)cl.len);
        exr_free(b->a, cl.data);
    }

    {
        uint8_t c8 = (uint8_t)comp;
        ob_attr(b, "compression", "compression", &c8, 1);
    }
    write_box2i(b, "dataWindow", &h->data_window);
    {
        exr_box2i dw = h->display_window;
        if (dw.max_x < dw.min_x || dw.max_y < dw.min_y) dw = h->data_window;
        write_box2i(b, "displayWindow", &dw);
    }
    {
        uint8_t lo = (uint8_t)h->line_order;
        ob_attr(b, "lineOrder", "lineOrder", &lo, 1);
    }
    {
        uint8_t t[4];
        exr_wr_f32(t, h->pixel_aspect_ratio != 0 ? h->pixel_aspect_ratio : 1.0f);
        ob_attr(b, "pixelAspectRatio", "float", t, 4);
    }
    {
        uint8_t t[8];
        exr_wr_f32(t, h->screen_window_center_x);
        exr_wr_f32(t + 4, h->screen_window_center_y);
        ob_attr(b, "screenWindowCenter", "v2f", t, 8);
    }
    {
        uint8_t t[4];
        exr_wr_f32(t, h->screen_window_width != 0 ? h->screen_window_width : 1.0f);
        ob_attr(b, "screenWindowWidth", "float", t, 4);
    }
    if (h->tiled) {
        uint8_t t[9];
        exr_wr_u32(t, h->tile_x_size);
        exr_wr_u32(t + 4, h->tile_y_size);
        t[8] = (uint8_t)((h->level_mode & 0x0f) | ((h->rounding_mode & 0x0f) << 4));
        ob_attr(b, "tiles", "tiledesc", t, 9);
    }
    if (multipart) {
        const char *nm = h->name[0] ? h->name : "part";
        const char *ty = part_type_string(h->part_type);
        ob_attr(b, "name", "string", nm, (uint32_t)strlen(nm));
        ob_attr(b, "type", "string", ty, (uint32_t)strlen(ty));
        {
            uint8_t t[4];
            exr_wr_i32(t, (int32_t)chunk_count);
            ob_attr(b, "chunkCount", "int", t, 4);
        }
    }
    if (h->part_type == EXR_PART_DEEP_SCANLINE ||
        h->part_type == EXR_PART_DEEP_TILED) {
        uint8_t t[4];
        if (!multipart) {
            const char *ty = part_type_string(h->part_type);
            ob_attr(b, "type", "string", ty, (uint32_t)strlen(ty));
        }
        exr_wr_i32(t, 1);
        ob_attr(b, "version", "int", t, 4); /* deep data version */
        exr_wr_i32(t, max_samples);
        ob_attr(b, "maxSamplesPerPixel", "int", t, 4);
    }
    ob_u8(b, 0); /* end of header */
}

/* ---- scanline gather ----------------------------------------------------- */

static void gather_scanline_block(const exr_header *h, const int *order,
                                  void *const *images, int y0, int nlines,
                                  uint8_t *block) {
    int xmin = h->data_window.min_x, xmax = h->data_window.max_x;
    int ymin = h->data_window.min_y;
    size_t off = 0;
    int line, oi;
    for (line = 0; line < nlines; ++line) {
        int yy = y0 + line;
        for (oi = 0; oi < h->num_channels; ++oi) {
            int c = order[oi];
            int xs = h->channels[c].x_sampling, ys = h->channels[c].y_sampling;
            size_t ps = exr_pixel_size(h->channels[c].pixel_type);
            int nx, cw, row;
            if ((yy % ys) != 0) continue;
            nx = exr_num_samples(xmin, xmax, xs);
            cw = nx;
            row = exr_num_samples(ymin, yy, ys) - 1;
            memcpy(block + off, (const uint8_t *)images[c] + (size_t)row * cw * ps,
                   (size_t)nx * ps);
            off += (size_t)nx * ps;
        }
    }
}

static void gather_tile_block(const exr_header *h, const int *order,
                              void *const *images, int abs_x0, int abs_y0,
                              int tile_w, int tile_h, uint8_t *block) {
    int xmin = h->data_window.min_x, xmax = h->data_window.max_x;
    int ymin = h->data_window.min_y;
    size_t off = 0;
    int row, oi;
    for (row = 0; row < tile_h; ++row) {
        int yy = abs_y0 + row;
        for (oi = 0; oi < h->num_channels; ++oi) {
            int c = order[oi];
            int xs = h->channels[c].x_sampling, ys = h->channels[c].y_sampling;
            size_t ps = exr_pixel_size(h->channels[c].pixel_type);
            int tnx, cw, ch_row, ch_col;
            if ((yy % ys) != 0) continue;
            tnx = exr_num_samples(abs_x0, abs_x0 + tile_w - 1, xs);
            if (tnx <= 0) continue;
            cw = exr_num_samples(xmin, xmax, xs);
            ch_row = exr_num_samples(ymin, yy, ys) - 1;
            ch_col = exr_num_samples(xmin, abs_x0 - 1, xs);
            memcpy(block + off,
                   (const uint8_t *)images[c] + ((size_t)ch_row * cw + ch_col) * ps,
                   (size_t)tnx * ps);
            off += (size_t)tnx * ps;
        }
    }
}

/* Gather a tile from a level image into the canonical block. level_img[c] is
 * the channel's sampled grid (exr_num_samples(0,lw-1,xs) wide). x0/y0 are the
 * tile's level-local pixel origin. */
static void gather_level_tile(const exr_header *h, const int *order,
                              void *const *level_img, int lw, int x0, int y0,
                              int tw, int th, uint8_t *block) {
    size_t off = 0;
    int row, oi;
    for (row = 0; row < th; ++row) {
        int yy = y0 + row;
        for (oi = 0; oi < h->num_channels; ++oi) {
            int c = order[oi];
            int xs = h->channels[c].x_sampling < 1 ? 1 : h->channels[c].x_sampling;
            int ys = h->channels[c].y_sampling < 1 ? 1 : h->channels[c].y_sampling;
            size_t ps = exr_pixel_size(h->channels[c].pixel_type);
            int gw, g0, cnt;
            if ((yy % ys) != 0) continue;
            gw = exr_num_samples(0, lw - 1, xs);    /* grid width */
            g0 = ((x0 + xs - 1) / xs);              /* first sampled col idx */
            cnt = exr_num_samples(x0, x0 + tw - 1, xs);
            if (cnt > 0)
                memcpy(block + off,
                       (const uint8_t *)level_img[c] +
                           ((size_t)(yy / ys) * gw + g0) * ps,
                       (size_t)cnt * ps);
            off += (size_t)cnt * ps;
        }
    }
}

/* Emit every tile of one deep level (lx,ly) to the output buffer. */
static exr_result emit_deep_tile_level(obuf *b, const exr_allocator *a,
                                       exr_compression comp, const exr_part *lvl,
                                       const uint64_t *lpfx, int p, int multipart,
                                       int lx, int ly, uint64_t *offtab,
                                       uint32_t *ci) {
    int tx = (int)lvl->header.tile_x_size, ty = (int)lvl->header.tile_y_size;
    int nxt = (lvl->width + tx - 1) / tx, nyt = (lvl->height + ty - 1) / ty;
    int txi, tyi;
    exr_result rc = EXR_SUCCESS;
    for (tyi = 0; tyi < nyt && EXR_OK(rc); ++tyi) {
        for (txi = 0; txi < nxt; ++txi) {
            int x0 = txi * tx, y0 = tyi * ty;
            int tw = (tx < lvl->width - x0) ? tx : (lvl->width - x0);
            int th = (ty < lvl->height - y0) ? ty : (lvl->height - y0);
            uint8_t *poff = NULL, *psamp = NULL;
            size_t poff_sz = 0, psamp_sz = 0;
            uint64_t uoff = 0, usamp = 0;
            rc = exr_deep_encode_tile(a, comp, lvl, lpfx, x0, y0, tw, th, &poff,
                                      &poff_sz, &uoff, &psamp, &psamp_sz, &usamp);
            if (!EXR_OK(rc)) break;
            offtab[(*ci)++] = (uint64_t)b->len;
            if (multipart) ob_i32(b, p);
            ob_i32(b, txi);
            ob_i32(b, tyi);
            ob_i32(b, lx);
            ob_i32(b, ly);
            ob_u64(b, (uint64_t)poff_sz);
            ob_u64(b, (uint64_t)psamp_sz);
            ob_u64(b, usamp);
            ob_bytes(b, poff, poff_sz);
            ob_bytes(b, psamp, psamp_sz);
            exr_free(a, poff);
            exr_free(a, psamp);
            (void)uoff;
            if (b->err) { rc = EXR_ERROR_OUT_OF_MEMORY; break; }
        }
    }
    return rc;
}

/* Prefix-sum a deep part's sample counts into a freshly allocated array. */
static uint64_t *deep_prefix(const exr_allocator *a, const exr_part *pt) {
    size_t npix = (size_t)pt->width * pt->height, i;
    uint64_t acc = 0, *pfx = (uint64_t *)exr_malloc(a, (npix ? npix : 1) * sizeof(uint64_t));
    if (!pfx) return NULL;
    for (i = 0; i < npix; ++i) { pfx[i] = acc; acc += (uint64_t)pt->deep_sample_counts[i]; }
    return pfx;
}

/* ---- serialize a set of parts -------------------------------------------- */

static exr_result serialize(const exr_allocator *a, const exr_part *parts,
                            int num_parts, int comp_override, uint8_t **out_data,
                            size_t *out_size) {
    obuf b;
    int p;
    int multipart = (num_parts > 1);
    int any_tiled = 0, any_deep = 0;
    uint64_t **offset_tables = NULL;
    size_t *offset_positions = NULL;
    uint32_t *chunk_counts = NULL;
    int **orders = NULL;
    exr_channel **sorted_chans = NULL;
    exr_result rc = EXR_SUCCESS;

    if (num_parts <= 0) return EXR_ERROR_INVALID_ARGUMENT;

    memset(&b, 0, sizeof(b));
    b.a = a;

    for (p = 0; p < num_parts; ++p) {
        if (parts[p].header.tiled) {
            any_tiled = 1;
            /* ONE_LEVEL + MIPMAP + RIPMAP supported (flat and deep). */
            if (parts[p].header.tile_x_size == 0 || parts[p].header.tile_y_size == 0)
                return EXR_ERROR_INVALID_ARGUMENT;
        }
        if (parts[p].is_deep) any_deep = 1; /* deep tiled allowed (ONE_LEVEL) */
    }

    orders = (int **)exr_calloc(a, (size_t)num_parts, sizeof(int *));
    sorted_chans = (exr_channel **)exr_calloc(a, (size_t)num_parts, sizeof(exr_channel *));
    chunk_counts = (uint32_t *)exr_calloc(a, (size_t)num_parts, sizeof(uint32_t));
    offset_tables = (uint64_t **)exr_calloc(a, (size_t)num_parts, sizeof(uint64_t *));
    offset_positions = (size_t *)exr_calloc(a, (size_t)num_parts, sizeof(size_t));
    if (!orders || !sorted_chans || !chunk_counts || !offset_tables ||
        !offset_positions) {
        rc = EXR_ERROR_OUT_OF_MEMORY;
        goto done;
    }
    for (p = 0; p < num_parts; ++p) {
        const exr_part *pt = &parts[p];
        int nch = pt->header.num_channels;
        int lpb, ci2;
        exr_compression comp =
            comp_override >= 0 ? (exr_compression)comp_override : pt->header.compression;
        orders[p] = (int *)exr_calloc(a, nch ? (size_t)nch : 1, sizeof(int));
        sorted_chans[p] =
            (exr_channel *)exr_calloc(a, nch ? (size_t)nch : 1, sizeof(exr_channel));
        if (!orders[p] || !sorted_chans[p]) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
        sort_channels(&pt->header, orders[p]);
        for (ci2 = 0; ci2 < nch; ++ci2)
            sorted_chans[p][ci2] = pt->header.channels[orders[p][ci2]];
        if (pt->header.tiled) {
            int tx = (int)pt->header.tile_x_size, ty = (int)pt->header.tile_y_size;
            if (pt->header.level_mode == EXR_TILE_MIPMAP_LEVELS) {
                int up = (pt->header.rounding_mode == EXR_TILE_ROUND_UP);
                int ww = pt->width, hh = pt->height;
                uint32_t total = 0;
                for (;;) {
                    uint32_t nx = (uint32_t)((ww + tx - 1) / tx);
                    uint32_t ny = (uint32_t)((hh + ty - 1) / ty);
                    total += nx * ny;
                    if (ww <= 1 && hh <= 1) break;
                    ww = up ? (ww + 1) / 2 : ww / 2; if (ww < 1) ww = 1;
                    hh = up ? (hh + 1) / 2 : hh / 2; if (hh < 1) hh = 1;
                }
                chunk_counts[p] = total;
            } else if (pt->header.level_mode == EXR_TILE_RIPMAP_LEVELS) {
                /* sum_{lx,ly} ceil(lw/tx)*ceil(lh/ty) = (sum_lx ..)*(sum_ly ..) */
                int up = (pt->header.rounding_mode == EXR_TILE_ROUND_UP);
                uint32_t sx = 0, sy = 0;
                int s;
                for (s = pt->width;;) {
                    sx += (uint32_t)((s + tx - 1) / tx);
                    if (s <= 1) break;
                    s = up ? (s + 1) / 2 : s / 2; if (s < 1) s = 1;
                }
                for (s = pt->height;;) {
                    sy += (uint32_t)((s + ty - 1) / ty);
                    if (s <= 1) break;
                    s = up ? (s + 1) / 2 : s / 2; if (s < 1) s = 1;
                }
                chunk_counts[p] = sx * sy;
            } else {
                uint32_t nx = (uint32_t)((pt->width + tx - 1) / tx);
                uint32_t ny = (uint32_t)((pt->height + ty - 1) / ty);
                chunk_counts[p] = nx * ny;
            }
        } else {
            lpb = exr_lines_per_block(comp);
            chunk_counts[p] = (uint32_t)(((int64_t)pt->height + lpb - 1) / lpb);
        }
        (void)lpb;
    }

    /* magic + version */
    ob_u32(&b, EXR_MAGIC);
    {
        uint32_t ver = EXR_VERSION_NUMBER;
        if (multipart) ver |= EXR_VERSION_FLAG_MULTIPART;
        else if (any_tiled && !any_deep) ver |= EXR_VERSION_FLAG_TILED;
        if (any_deep) ver |= EXR_VERSION_FLAG_NON_IMAGE;
        ob_u32(&b, ver);
    }

    /* headers */
    for (p = 0; p < num_parts; ++p) {
        const exr_part *pt = &parts[p];
        exr_compression comp =
            comp_override >= 0 ? (exr_compression)comp_override : pt->header.compression;
        int32_t max_samples = 0;
        if (pt->is_deep && pt->deep_sample_counts) {
            size_t npix = (size_t)pt->width * pt->height, i;
            for (i = 0; i < npix; ++i)
                if (pt->deep_sample_counts[i] > max_samples)
                    max_samples = pt->deep_sample_counts[i];
        }
        write_header(&b, &pt->header, orders[p], comp, multipart,
                     chunk_counts[p], max_samples);
    }
    if (multipart) ob_u8(&b, 0); /* end of header list */

    /* offset tables (reserve, backpatch later) */
    for (p = 0; p < num_parts; ++p) {
        uint32_t k;
        offset_positions[p] = b.len;
        offset_tables[p] =
            (uint64_t *)exr_calloc(a, chunk_counts[p] ? chunk_counts[p] : 1,
                                   sizeof(uint64_t));
        if (!offset_tables[p]) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
        for (k = 0; k < chunk_counts[p]; ++k) ob_u64(&b, 0);
    }
    if (b.err) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }

    /* chunks */
    for (p = 0; p < num_parts; ++p) {
        const exr_part *pt = &parts[p];
        const exr_header *h = &pt->header;
        exr_compression comp =
            comp_override >= 0 ? (exr_compression)comp_override : h->compression;
        int xmin = h->data_window.min_x, ymin = h->data_window.min_y;
        int ymax = h->data_window.max_y;

        if (pt->is_deep) {
            int lpb = exr_lines_per_block(comp);
            uint64_t *prefix = NULL;
            size_t npix = (size_t)pt->width * pt->height, i;
            uint64_t acc = 0;
            uint32_t ci;
            prefix = (uint64_t *)exr_malloc(a, (npix ? npix : 1) * sizeof(uint64_t));
            if (!prefix) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
            for (i = 0; i < npix; ++i) {
                prefix[i] = acc;
                acc += (uint64_t)pt->deep_sample_counts[i];
            }
            if (h->tiled) {
                int up = (h->rounding_mode == EXR_TILE_ROUND_UP);
                uint32_t ci = 0;
                if (h->level_mode == EXR_TILE_MIPMAP_LEVELS) {
                    int ww = pt->width, hh = pt->height, L = 0;
                    for (;;) {
                        if (L == 0) {
                            rc = emit_deep_tile_level(&b, a, comp, pt, prefix, p,
                                                      multipart, 0, 0,
                                                      offset_tables[p], &ci);
                        } else {
                            exr_part lvl;
                            uint64_t *lpfx;
                            rc = exr_deep_build_level(a, pt, prefix, ww, hh, &lvl);
                            if (EXR_OK(rc)) {
                                lpfx = deep_prefix(a, &lvl);
                                if (!lpfx) rc = EXR_ERROR_OUT_OF_MEMORY;
                                else {
                                    rc = emit_deep_tile_level(&b, a, comp, &lvl,
                                                              lpfx, p, multipart, L,
                                                              L, offset_tables[p],
                                                              &ci);
                                    exr_free(a, lpfx);
                                }
                                exr_deep_level_free(a, &lvl);
                            }
                        }
                        if (!EXR_OK(rc) || (ww <= 1 && hh <= 1)) break;
                        ww = up ? (ww + 1) / 2 : ww / 2; if (ww < 1) ww = 1;
                        hh = up ? (hh + 1) / 2 : hh / 2; if (hh < 1) hh = 1;
                        L++;
                    }
                } else if (h->level_mode == EXR_TILE_RIPMAP_LEVELS) {
                    int nxl = 0, nyl = 0, s, lx, ly;
                    for (s = pt->width;;) { nxl++; if (s <= 1) break; s = up ? (s + 1) / 2 : s / 2; if (s < 1) s = 1; }
                    for (s = pt->height;;) { nyl++; if (s <= 1) break; s = up ? (s + 1) / 2 : s / 2; if (s < 1) s = 1; }
                    for (ly = 0; ly < nyl && EXR_OK(rc); ++ly) {
                        for (lx = 0; lx < nxl && EXR_OK(rc); ++lx) {
                            int lw = pt->width, lh = pt->height, t;
                            for (t = 0; t < lx; ++t) { lw = up ? (lw + 1) / 2 : lw / 2; if (lw < 1) lw = 1; }
                            for (t = 0; t < ly; ++t) { lh = up ? (lh + 1) / 2 : lh / 2; if (lh < 1) lh = 1; }
                            if (lx == 0 && ly == 0) {
                                rc = emit_deep_tile_level(&b, a, comp, pt, prefix, p,
                                                          multipart, 0, 0,
                                                          offset_tables[p], &ci);
                            } else {
                                exr_part lvl;
                                uint64_t *lpfx;
                                rc = exr_deep_build_level(a, pt, prefix, lw, lh, &lvl);
                                if (EXR_OK(rc)) {
                                    lpfx = deep_prefix(a, &lvl);
                                    if (!lpfx) rc = EXR_ERROR_OUT_OF_MEMORY;
                                    else {
                                        rc = emit_deep_tile_level(&b, a, comp, &lvl,
                                                                  lpfx, p, multipart,
                                                                  lx, ly,
                                                                  offset_tables[p],
                                                                  &ci);
                                        exr_free(a, lpfx);
                                    }
                                    exr_deep_level_free(a, &lvl);
                                }
                            }
                        }
                    }
                } else {
                    rc = emit_deep_tile_level(&b, a, comp, pt, prefix, p, multipart,
                                              0, 0, offset_tables[p], &ci);
                }
                exr_free(a, prefix);
                if (!EXR_OK(rc)) goto done;
                continue;
            }
            for (ci = 0; ci < chunk_counts[p]; ++ci) {
                int y0 = ymin + (int)ci * lpb;
                int nlines = (y0 + lpb - 1 > ymax) ? (ymax - y0 + 1) : lpb;
                uint8_t *poff = NULL, *psamp = NULL;
                size_t poff_sz = 0, psamp_sz = 0;
                uint64_t uoff = 0, usamp = 0;
                rc = exr_deep_encode_block(a, comp, pt, prefix, y0, nlines, &poff,
                                           &poff_sz, &uoff, &psamp, &psamp_sz,
                                           &usamp);
                if (!EXR_OK(rc)) { exr_free(a, prefix); goto done; }
                offset_tables[p][ci] = (uint64_t)b.len;
                if (multipart) ob_i32(&b, p);
                ob_i32(&b, y0);
                ob_u64(&b, (uint64_t)poff_sz);
                ob_u64(&b, (uint64_t)psamp_sz);
                ob_u64(&b, usamp);
                ob_bytes(&b, poff, poff_sz);
                ob_bytes(&b, psamp, psamp_sz);
                exr_free(a, poff);
                exr_free(a, psamp);
                (void)uoff;
                if (b.err) { exr_free(a, prefix); rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
            }
            exr_free(a, prefix);
        } else if (h->tiled && h->level_mode == EXR_TILE_MIPMAP_LEVELS) {
            int tx = (int)h->tile_x_size, ty = (int)h->tile_y_size;
            int L, txi, tyi;
            uint32_t ci = 0;
            exr_mip_pyramid pyr;
            rc = exr_mip_generate(a, pt, h->rounding_mode == EXR_TILE_ROUND_UP,
                                  &pyr);
            if (!EXR_OK(rc)) goto done;
            for (L = 0; L < pyr.num_levels; ++L) {
                int lw = pyr.lw[L], lh = pyr.lh[L];
                int nxt = (lw + tx - 1) / tx, nyt = (lh + ty - 1) / ty;
                for (tyi = 0; tyi < nyt && EXR_OK(rc); ++tyi) {
                    for (txi = 0; txi < nxt; ++txi) {
                        int x0 = txi * tx, y0 = tyi * ty;
                        int tw = (tx < lw - x0) ? tx : (lw - x0);
                        int th = (ty < lh - y0) ? ty : (lh - y0);
                        size_t blk_size, payload_size = 0;
                        uint8_t *block, *payload = NULL;
                        exr_codec_ctx cx;
                        rc = exr_block_uncompressed_size(sorted_chans[p],
                                                         h->num_channels, x0, y0,
                                                         tw, th, &blk_size);
                        if (!EXR_OK(rc)) break;
                        block = (uint8_t *)exr_malloc(a, blk_size ? blk_size : 1);
                        if (!block) { rc = EXR_ERROR_OUT_OF_MEMORY; break; }
                        gather_level_tile(h, orders[p], pyr.img[L], lw, x0, y0, tw,
                                          th, block);
                        cx.alloc = a;
                        cx.compression = comp;
                        cx.channels = sorted_chans[p];
                        cx.num_channels = h->num_channels;
                        cx.x = x0;
                        cx.y = y0;
                        cx.width = tw;
                        cx.num_lines = th;
                        rc = exr_compress_block(&cx, block, blk_size, &payload,
                                                &payload_size);
                        exr_free(a, block);
                        if (!EXR_OK(rc)) break;
                        offset_tables[p][ci++] = (uint64_t)b.len;
                        if (multipart) ob_i32(&b, p);
                        ob_i32(&b, txi);
                        ob_i32(&b, tyi);
                        ob_i32(&b, L);
                        ob_i32(&b, L);
                        ob_i32(&b, (int32_t)payload_size);
                        ob_bytes(&b, payload, payload_size);
                        exr_free(a, payload);
                        if (b.err) { rc = EXR_ERROR_OUT_OF_MEMORY; break; }
                    }
                }
                if (!EXR_OK(rc)) break;
            }
            exr_mip_free(&pyr);
            if (!EXR_OK(rc)) goto done;
            continue;
        } else if (h->tiled && h->level_mode == EXR_TILE_RIPMAP_LEVELS) {
            int tx = (int)h->tile_x_size, ty = (int)h->tile_y_size;
            int lx, ly, txi, tyi;
            uint32_t ci = 0;
            exr_ripmap_pyramid pyr;
            rc = exr_ripmap_generate(a, pt, h->rounding_mode == EXR_TILE_ROUND_UP,
                                     &pyr);
            if (!EXR_OK(rc)) goto done;
            /* offset-table order: ly outer, lx inner (matches the reader). */
            for (ly = 0; ly < pyr.num_y_levels && EXR_OK(rc); ++ly) {
                for (lx = 0; lx < pyr.num_x_levels && EXR_OK(rc); ++lx) {
                    int lw = pyr.lw[lx], lh = pyr.lh[ly];
                    void **limg = pyr.img[lx * pyr.num_y_levels + ly];
                    int nxt = (lw + tx - 1) / tx, nyt = (lh + ty - 1) / ty;
                    for (tyi = 0; tyi < nyt && EXR_OK(rc); ++tyi) {
                        for (txi = 0; txi < nxt; ++txi) {
                            int x0 = txi * tx, y0 = tyi * ty;
                            int tw = (tx < lw - x0) ? tx : (lw - x0);
                            int th = (ty < lh - y0) ? ty : (lh - y0);
                            size_t blk_size, payload_size = 0;
                            uint8_t *block, *payload = NULL;
                            exr_codec_ctx cx;
                            rc = exr_block_uncompressed_size(sorted_chans[p],
                                                             h->num_channels, x0,
                                                             y0, tw, th, &blk_size);
                            if (!EXR_OK(rc)) break;
                            block = (uint8_t *)exr_malloc(a, blk_size ? blk_size : 1);
                            if (!block) { rc = EXR_ERROR_OUT_OF_MEMORY; break; }
                            gather_level_tile(h, orders[p], limg, lw, x0, y0, tw,
                                              th, block);
                            cx.alloc = a;
                            cx.compression = comp;
                            cx.channels = sorted_chans[p];
                            cx.num_channels = h->num_channels;
                            cx.x = x0;
                            cx.y = y0;
                            cx.width = tw;
                            cx.num_lines = th;
                            rc = exr_compress_block(&cx, block, blk_size, &payload,
                                                    &payload_size);
                            exr_free(a, block);
                            if (!EXR_OK(rc)) break;
                            offset_tables[p][ci++] = (uint64_t)b.len;
                            if (multipart) ob_i32(&b, p);
                            ob_i32(&b, txi);
                            ob_i32(&b, tyi);
                            ob_i32(&b, lx);
                            ob_i32(&b, ly);
                            ob_i32(&b, (int32_t)payload_size);
                            ob_bytes(&b, payload, payload_size);
                            exr_free(a, payload);
                            if (b.err) { rc = EXR_ERROR_OUT_OF_MEMORY; break; }
                        }
                    }
                }
            }
            exr_ripmap_free(&pyr);
            if (!EXR_OK(rc)) goto done;
            continue;
        } else if (h->tiled) {
            int tx = (int)h->tile_x_size, ty = (int)h->tile_y_size;
            int nxt = (pt->width + tx - 1) / tx;
            int nyt = (pt->height + ty - 1) / ty;
            int txi, tyi;
            for (tyi = 0; tyi < nyt; ++tyi) {
                for (txi = 0; txi < nxt; ++txi) {
                    uint32_t ci = (uint32_t)tyi * (uint32_t)nxt + (uint32_t)txi;
                    int x0 = txi * tx, y0 = tyi * ty;
                    int tile_w = (tx < pt->width - x0) ? tx : (pt->width - x0);
                    int tile_h = (ty < pt->height - y0) ? ty : (pt->height - y0);
                    int abs_x0 = xmin + x0, abs_y0 = ymin + y0;
                    size_t blk_size, payload_size = 0;
                    uint8_t *block, *payload = NULL;

                    rc = exr_block_uncompressed_size(h->channels, h->num_channels,
                                                     abs_x0, abs_y0, tile_w,
                                                     tile_h, &blk_size);
                    if (!EXR_OK(rc)) goto done;
                    block = (uint8_t *)exr_malloc(a, blk_size ? blk_size : 1);
                    if (!block) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
                    gather_tile_block(h, orders[p], (void *const *)pt->images,
                                      abs_x0, abs_y0, tile_w, tile_h, block);
                    {
                        exr_codec_ctx cx;
                        cx.alloc = a;
                        cx.compression = comp;
                        cx.channels = sorted_chans[p];
                        cx.num_channels = h->num_channels;
                        cx.x = abs_x0;
                        cx.y = abs_y0;
                        cx.width = tile_w;
                        cx.num_lines = tile_h;
                        rc = exr_compress_block(&cx, block, blk_size, &payload,
                                                &payload_size);
                    }
                    exr_free(a, block);
                    if (!EXR_OK(rc)) goto done;

                    offset_tables[p][ci] = (uint64_t)b.len;
                    if (multipart) ob_i32(&b, p);
                    ob_i32(&b, txi);
                    ob_i32(&b, tyi);
                    ob_i32(&b, 0); /* level x */
                    ob_i32(&b, 0); /* level y */
                    ob_i32(&b, (int32_t)payload_size);
                    ob_bytes(&b, payload, payload_size);
                    exr_free(a, payload);
                    if (b.err) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
                }
            }
        } else {
            int lpb = exr_lines_per_block(comp);
            uint32_t ci;
            for (ci = 0; ci < chunk_counts[p]; ++ci) {
                int y0 = ymin + (int)ci * lpb;
                int nlines = (y0 + lpb - 1 > ymax) ? (ymax - y0 + 1) : lpb;
                size_t blk_size, payload_size = 0;
                uint8_t *block, *payload = NULL;

                rc = exr_block_uncompressed_size(h->channels, h->num_channels,
                                                 xmin, y0, pt->width, nlines,
                                                 &blk_size);
                if (!EXR_OK(rc)) goto done;
                block = (uint8_t *)exr_malloc(a, blk_size ? blk_size : 1);
                if (!block) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
                gather_scanline_block(h, orders[p], (void *const *)pt->images, y0,
                                      nlines, block);
                {
                    exr_codec_ctx cx;
                    cx.alloc = a;
                    cx.compression = comp;
                    cx.channels = sorted_chans[p];
                    cx.num_channels = h->num_channels;
                    cx.x = xmin;
                    cx.y = y0;
                    cx.width = pt->width;
                    cx.num_lines = nlines;
                    rc = exr_compress_block(&cx, block, blk_size, &payload,
                                            &payload_size);
                }
                exr_free(a, block);
                if (!EXR_OK(rc)) goto done;

                offset_tables[p][ci] = (uint64_t)b.len;
                if (multipart) ob_i32(&b, p);
                ob_i32(&b, y0);
                ob_i32(&b, (int32_t)payload_size);
                ob_bytes(&b, payload, payload_size);
                exr_free(a, payload);
                if (b.err) { rc = EXR_ERROR_OUT_OF_MEMORY; goto done; }
            }
        }
    }

    /* backpatch offset tables */
    for (p = 0; p < num_parts; ++p) {
        uint32_t k;
        for (k = 0; k < chunk_counts[p]; ++k)
            exr_wr_u64(b.data + offset_positions[p] + (size_t)k * 8,
                       offset_tables[p][k]);
    }

    *out_data = b.data;
    *out_size = b.len;
    b.data = NULL; /* ownership transferred */

done:
    if (orders) {
        for (p = 0; p < num_parts; ++p) exr_free(a, orders[p]);
        exr_free(a, orders);
    }
    if (sorted_chans) {
        for (p = 0; p < num_parts; ++p) exr_free(a, sorted_chans[p]);
        exr_free(a, sorted_chans);
    }
    if (offset_tables) {
        for (p = 0; p < num_parts; ++p) exr_free(a, offset_tables[p]);
        exr_free(a, offset_tables);
    }
    exr_free(a, offset_positions);
    exr_free(a, chunk_counts);
    exr_free(a, b.data);
    return rc;
}

/* ---- high-level save ----------------------------------------------------- */

exr_result exr_save_to_memory(void **out_data, size_t *out_size,
                              const exr_allocator *alloc, const exr_image *img,
                              exr_compression compression) {
    if (!out_data || !out_size || !img || img->num_parts <= 0)
        return EXR_ERROR_INVALID_ARGUMENT;
    if (!alloc) alloc = exr_default_allocator();
    *out_data = NULL;
    *out_size = 0;
    return serialize(alloc, img->parts, img->num_parts, (int)compression,
                     (uint8_t **)out_data, out_size);
}

exr_result exr_save_to_file(const char *path, const exr_image *img,
                            exr_compression compression) {
    void *data = NULL;
    size_t size = 0;
    exr_result rc;
    FILE *fp;
    const exr_allocator *a = exr_default_allocator();

    if (!path || !img) return EXR_ERROR_INVALID_ARGUMENT;
    rc = exr_save_to_memory(&data, &size, a, img, compression);
    if (!EXR_OK(rc)) return rc;

    fp = fopen(path, "wb");
    if (!fp) {
        exr_free(a, data);
        return EXR_ERROR_IO;
    }
    if (fwrite(data, 1, size, fp) != size) {
        fclose(fp);
        exr_free(a, data);
        return EXR_ERROR_IO;
    }
    fclose(fp);
    exr_free(a, data);
    return EXR_SUCCESS;
}

/* ---- mid-level writer ---------------------------------------------------- */

struct exr_writer {
    exr_allocator alloc;
    exr_part *parts;
    int num_parts, cap;
};

exr_result exr_writer_create(const exr_allocator *alloc, exr_writer **out) {
    exr_writer *w;
    if (!out) return EXR_ERROR_INVALID_ARGUMENT;
    if (!alloc) alloc = exr_default_allocator();
    w = (exr_writer *)exr_calloc(alloc, 1, sizeof(*w));
    if (!w) return EXR_ERROR_OUT_OF_MEMORY;
    w->alloc = *alloc;
    *out = w;
    return EXR_SUCCESS;
}

void exr_writer_destroy(exr_writer *w) {
    int p;
    if (!w) return;
    for (p = 0; p < w->num_parts; ++p) {
        exr_free(&w->alloc, w->parts[p].images); /* pixels are caller-owned */
        exr_header_free(&w->alloc, &w->parts[p].header);
    }
    exr_free(&w->alloc, w->parts);
    exr_free(&w->alloc, w);
}

exr_result exr_writer_add_part(exr_writer *w, const exr_header *hdr,
                               int32_t *out_part) {
    exr_part *pt;
    exr_result rc;
    if (!w || !hdr) return EXR_ERROR_INVALID_ARGUMENT;
    if (w->num_parts == w->cap) {
        int ncap = w->cap ? w->cap * 2 : 2;
        exr_part *np = (exr_part *)exr_calloc(&w->alloc, (size_t)ncap, sizeof(exr_part));
        if (!np) return EXR_ERROR_OUT_OF_MEMORY;
        if (w->parts) {
            memcpy(np, w->parts, (size_t)w->num_parts * sizeof(exr_part));
            exr_free(&w->alloc, w->parts);
        }
        w->parts = np;
        w->cap = ncap;
    }
    pt = &w->parts[w->num_parts];
    memset(pt, 0, sizeof(*pt));
    rc = exr_header_copy(&w->alloc, &pt->header, hdr);
    if (!EXR_OK(rc)) return rc;
    pt->width = hdr->data_window.max_x - hdr->data_window.min_x + 1;
    pt->height = hdr->data_window.max_y - hdr->data_window.min_y + 1;
    pt->images = (void **)exr_calloc(&w->alloc,
                                     hdr->num_channels ? (size_t)hdr->num_channels : 1,
                                     sizeof(void *));
    if (!pt->images) {
        exr_header_free(&w->alloc, &pt->header);
        return EXR_ERROR_OUT_OF_MEMORY;
    }
    if (out_part) *out_part = w->num_parts;
    w->num_parts++;
    return EXR_SUCCESS;
}

exr_result exr_writer_set_channel(exr_writer *w, int32_t part, const char *name,
                                  const void *pixels) {
    int c;
    exr_part *pt;
    if (!w || !name || part < 0 || part >= w->num_parts)
        return EXR_ERROR_INVALID_ARGUMENT;
    pt = &w->parts[part];
    for (c = 0; c < pt->header.num_channels; ++c)
        if (strcmp(pt->header.channels[c].name, name) == 0) {
            pt->images[c] = (void *)pixels;
            return EXR_SUCCESS;
        }
    return EXR_ERROR_INVALID_ARGUMENT;
}

exr_result exr_writer_finalize_to_memory(exr_writer *w, void **out_data,
                                         size_t *out_size) {
    if (!w || !out_data || !out_size) return EXR_ERROR_INVALID_ARGUMENT;
    *out_data = NULL;
    *out_size = 0;
    return serialize(&w->alloc, w->parts, w->num_parts, -1,
                     (uint8_t **)out_data, out_size);
}

exr_result exr_writer_finalize_to_file(exr_writer *w, const char *path) {
    void *data = NULL;
    size_t size = 0;
    exr_result rc;
    FILE *fp;
    if (!w || !path) return EXR_ERROR_INVALID_ARGUMENT;
    rc = exr_writer_finalize_to_memory(w, &data, &size);
    if (!EXR_OK(rc)) return rc;
    fp = fopen(path, "wb");
    if (!fp) { exr_free(&w->alloc, data); return EXR_ERROR_IO; }
    if (fwrite(data, 1, size, fp) != size) {
        fclose(fp);
        exr_free(&w->alloc, data);
        return EXR_ERROR_IO;
    }
    fclose(fp);
    exr_free(&w->alloc, data);
    return EXR_SUCCESS;
}
