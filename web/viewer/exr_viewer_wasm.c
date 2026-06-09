/*
 * TinyEXR v3 - WebAssembly binding for the browser viewer (web/viewer/).
 *
 * A small stateful session layer over the v3 streaming block API
 * (include/exr.h). It lets JavaScript:
 *   - open an EXR held in WASM memory and parse all part headers,
 *   - read the full structured header as JSON (for the info panel + selectors),
 *   - select one (part, mip level) and decode it block-by-block so the UI can
 *     show load progress, accumulating the result into an RGBA float buffer,
 *   - hand that buffer to JS as a HEAPF32 view (for WebGL upload + the pixel
 *     picker).
 *
 * No filesystem, no stdio. Pixel data crosses the JS boundary through
 * emscripten's _malloc/_free and the HEAPF32/HEAPU8 views. The in-memory reader
 * never returns EXR_WOULD_BLOCK, so no suspend/resume loop is needed; the
 * progress reporting comes from driving the per-block decode loop from JS.
 *
 * Custom/standard attributes (exr_header.attrs) are opaque in the public v3 API
 * with no iteration entry point, so the JSON exposes the structured header
 * fields and the channel list only.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXRV_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXRV_EXPORT
#endif

/* ------------------------------------------------------------------ session */

typedef struct exrv_session {
    uint8_t *data; /* owned copy of the file bytes (reader is zero-copy) */
    size_t size;
    exr_reader *r;
    int num_parts;

    /* current selection (set by exrv_select) */
    int sel_part;
    int lw, lh;             /* selected level pixel dimensions */
    int dw_min_x, dw_min_y; /* selected part data-window origin */
    uint32_t *blocks;       /* global block indices belonging to part+level */
    int nblocks;
    int rmap[4]; /* header channel index for R,G,B,A or -1 */
    float *rgba; /* lw*lh*4 accumulation buffer (RGBA float) */
} exrv_session;

#define EXRV_MAX_SESSIONS 16
static exrv_session *g_sessions[EXRV_MAX_SESSIONS];

static exrv_session *session_from_handle(int h) {
    if (h < 1 || h > EXRV_MAX_SESSIONS) return NULL;
    return g_sessions[h - 1];
}

static void session_reset_selection(exrv_session *s) {
    free(s->blocks);
    s->blocks = NULL;
    s->nblocks = 0;
    free(s->rgba);
    s->rgba = NULL;
    s->lw = s->lh = 0;
    s->sel_part = -1;
}

/* ------------------------------------------------------------------ helpers */

static int find_channel(const exr_header *h, const char *name) {
    int c;
    for (c = 0; c < h->num_channels; ++c)
        if (strcmp(h->channels[c].name, name) == 0) return c;
    return -1;
}

static int pixel_elem_size(exr_pixel_type t) {
    switch (t) {
    case EXR_PIXEL_HALF:
        return 2;
    case EXR_PIXEL_FLOAT:
    case EXR_PIXEL_UINT:
        return 4;
    }
    return 0;
}

/* ------------------------------------------------------------- string builder
 * Minimal growable buffer for assembling the header JSON. */

typedef struct {
    char *buf;
    size_t len, cap;
    int oom;
} strbuf;

static void sb_ensure(strbuf *b, size_t extra) {
    if (b->oom) return;
    if (b->len + extra + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 1024;
        char *nb;
        while (ncap < b->len + extra + 1) ncap *= 2;
        nb = (char *)realloc(b->buf, ncap);
        if (!nb) {
            b->oom = 1;
            return;
        }
        b->buf = nb;
        b->cap = ncap;
    }
}

static void sb_puts(strbuf *b, const char *s) {
    size_t n = strlen(s);
    sb_ensure(b, n);
    if (b->oom) return;
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

/* Append a JSON string value with the few escapes that matter here. */
static void sb_json_str(strbuf *b, const char *s) {
    sb_puts(b, "\"");
    for (; *s; ++s) {
        char c = *s;
        if (c == '"' || c == '\\') {
            char esc[3];
            esc[0] = '\\';
            esc[1] = c;
            esc[2] = '\0';
            sb_puts(b, esc);
        } else if ((unsigned char)c < 0x20) {
            char u[8];
            snprintf(u, sizeof(u), "\\u%04x", (unsigned)(unsigned char)c);
            sb_puts(b, u);
        } else {
            char one[2];
            one[0] = c;
            one[1] = '\0';
            sb_puts(b, one);
        }
    }
    sb_puts(b, "\"");
}

static void sb_kv_int(strbuf *b, const char *key, long v) {
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "\"%s\":%ld", key, v);
    sb_puts(b, tmp);
}

static const char *comp_name(exr_compression c) {
    switch (c) {
    case EXR_COMPRESSION_NONE:
        return "none";
    case EXR_COMPRESSION_RLE:
        return "rle";
    case EXR_COMPRESSION_ZIPS:
        return "zips";
    case EXR_COMPRESSION_ZIP:
        return "zip";
    case EXR_COMPRESSION_PIZ:
        return "piz";
    case EXR_COMPRESSION_PXR24:
        return "pxr24";
    case EXR_COMPRESSION_B44:
        return "b44";
    case EXR_COMPRESSION_B44A:
        return "b44a";
    case EXR_COMPRESSION_DWAA:
        return "dwaa";
    case EXR_COMPRESSION_DWAB:
        return "dwab";
    case EXR_COMPRESSION_HTJ2K256:
        return "htj2k256";
    case EXR_COMPRESSION_HTJ2K32:
        return "htj2k32";
    case EXR_COMPRESSION_ZSTD:
        return "zstd";
    }
    return "unknown";
}

static const char *parttype_name(exr_part_type t) {
    switch (t) {
    case EXR_PART_SCANLINE:
        return "scanline";
    case EXR_PART_TILED:
        return "tiled";
    case EXR_PART_DEEP_SCANLINE:
        return "deepscanline";
    case EXR_PART_DEEP_TILED:
        return "deeptiled";
    }
    return "unknown";
}

static const char *lineorder_name(exr_line_order o) {
    switch (o) {
    case EXR_LINEORDER_INCREASING_Y:
        return "increasingY";
    case EXR_LINEORDER_DECREASING_Y:
        return "decreasingY";
    case EXR_LINEORDER_RANDOM_Y:
        return "randomY";
    }
    return "unknown";
}

static const char *pixeltype_name(exr_pixel_type t) {
    switch (t) {
    case EXR_PIXEL_UINT:
        return "uint";
    case EXR_PIXEL_HALF:
        return "half";
    case EXR_PIXEL_FLOAT:
        return "float";
    }
    return "unknown";
}

static const char *levelmode_name(exr_tile_level_mode m) {
    switch (m) {
    case EXR_TILE_ONE_LEVEL:
        return "one";
    case EXR_TILE_MIPMAP_LEVELS:
        return "mipmap";
    case EXR_TILE_RIPMAP_LEVELS:
        return "ripmap";
    }
    return "unknown";
}

/* ------------------------------------------------------------------- exports */

EXRV_EXPORT int exrv_open(const uint8_t *data, int size) {
    int slot;
    exrv_session *s;
    if (!data || size <= 0) return 0;

    for (slot = 0; slot < EXRV_MAX_SESSIONS; ++slot)
        if (!g_sessions[slot]) break;
    if (slot == EXRV_MAX_SESSIONS) return 0;

    s = (exrv_session *)calloc(1, sizeof(*s));
    if (!s) return 0;
    s->sel_part = -1;

    /* The reader is zero-copy and requires the bytes to outlive it, so keep an
     * owned copy independent of whatever JS does with its input buffer. */
    s->data = (uint8_t *)malloc((size_t)size);
    if (!s->data) {
        free(s);
        return 0;
    }
    memcpy(s->data, data, (size_t)size);
    s->size = (size_t)size;

    if (!EXR_OK(exr_reader_open_memory(s->data, s->size, NULL, &s->r))) {
        free(s->data);
        free(s);
        return 0;
    }
    if (exr_reader_parse_header(s->r) != EXR_SUCCESS) {
        exr_reader_close(s->r);
        free(s->data);
        free(s);
        return 0;
    }
    s->num_parts = exr_reader_num_parts(s->r);

    g_sessions[slot] = s;
    return slot + 1;
}

EXRV_EXPORT void exrv_close(int h) {
    exrv_session *s = session_from_handle(h);
    if (!s) return;
    session_reset_selection(s);
    if (s->r) exr_reader_close(s->r);
    free(s->data);
    free(s);
    g_sessions[h - 1] = NULL;
}

EXRV_EXPORT int exrv_num_parts(int h) {
    exrv_session *s = session_from_handle(h);
    return s ? s->num_parts : 0;
}

/* Build the structured-header JSON. Caller frees the returned string with
 * exrv_free. Returns NULL on error. */
EXRV_EXPORT char *exrv_header_json(int h) {
    exrv_session *s = session_from_handle(h);
    strbuf b;
    int part;
    if (!s) return NULL;

    memset(&b, 0, sizeof(b));
    sb_puts(&b, "{");
    sb_kv_int(&b, "numParts", s->num_parts);
    sb_puts(&b, ",\"parts\":[");

    for (part = 0; part < s->num_parts; ++part) {
        const exr_header *hd = exr_reader_part_header(s->r, part);
        int width, height, c;
        int is_deep;
        if (!hd) continue;
        width = hd->data_window.max_x - hd->data_window.min_x + 1;
        height = hd->data_window.max_y - hd->data_window.min_y + 1;
        is_deep = (hd->part_type == EXR_PART_DEEP_SCANLINE ||
                   hd->part_type == EXR_PART_DEEP_TILED);

        if (part) sb_puts(&b, ",");
        sb_puts(&b, "{");
        sb_puts(&b, "\"name\":");
        sb_json_str(&b, hd->name);
        sb_puts(&b, ",\"type\":");
        sb_json_str(&b, parttype_name(hd->part_type));
        sb_puts(&b, ",\"compression\":");
        sb_json_str(&b, comp_name(hd->compression));
        sb_puts(&b, ",\"lineOrder\":");
        sb_json_str(&b, lineorder_name(hd->line_order));
        sb_puts(&b, ",");
        sb_kv_int(&b, "width", width);
        sb_puts(&b, ",");
        sb_kv_int(&b, "height", height);
        sb_puts(&b, ",");
        sb_kv_int(&b, "deep", is_deep);

        sb_puts(&b, ",\"dataWindow\":{");
        sb_kv_int(&b, "minX", hd->data_window.min_x);
        sb_puts(&b, ",");
        sb_kv_int(&b, "minY", hd->data_window.min_y);
        sb_puts(&b, ",");
        sb_kv_int(&b, "maxX", hd->data_window.max_x);
        sb_puts(&b, ",");
        sb_kv_int(&b, "maxY", hd->data_window.max_y);
        sb_puts(&b, "}");

        sb_puts(&b, ",\"displayWindow\":{");
        sb_kv_int(&b, "minX", hd->display_window.min_x);
        sb_puts(&b, ",");
        sb_kv_int(&b, "minY", hd->display_window.min_y);
        sb_puts(&b, ",");
        sb_kv_int(&b, "maxX", hd->display_window.max_x);
        sb_puts(&b, ",");
        sb_kv_int(&b, "maxY", hd->display_window.max_y);
        sb_puts(&b, "}");

        {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), ",\"pixelAspect\":%g",
                     (double)hd->pixel_aspect_ratio);
            sb_puts(&b, tmp);
        }

        sb_puts(&b, ",");
        sb_kv_int(&b, "tiled", hd->tiled);
        if (hd->tiled) {
            sb_puts(&b, ",");
            sb_kv_int(&b, "tileXSize", (long)hd->tile_x_size);
            sb_puts(&b, ",");
            sb_kv_int(&b, "tileYSize", (long)hd->tile_y_size);
            sb_puts(&b, ",\"levelMode\":");
            sb_json_str(&b, levelmode_name(hd->level_mode));
        }

        /* Available (level_x, level_y) pairs, gathered from the block geometry
         * so the UI mip selector matches exactly what we can decode. */
        sb_puts(&b, ",\"levels\":[");
        if (hd->tiled && hd->level_mode != EXR_TILE_ONE_LEVEL) {
            uint32_t nb = 0, i;
            int first = 1;
            /* dedupe small level sets with an O(n^2) scan; level counts are tiny */
            int seen_lx[256], seen_ly[256], nseen = 0;
            if (EXR_OK(exr_reader_num_blocks(s->r, part, &nb))) {
                for (i = 0; i < nb; ++i) {
                    exr_block_info bi;
                    int k, dup = 0;
                    if (!EXR_OK(exr_reader_block_info(s->r, part, i, &bi)))
                        continue;
                    for (k = 0; k < nseen; ++k)
                        if (seen_lx[k] == bi.level_x && seen_ly[k] == bi.level_y) {
                            dup = 1;
                            break;
                        }
                    if (dup || nseen >= 256) continue;
                    seen_lx[nseen] = bi.level_x;
                    seen_ly[nseen] = bi.level_y;
                    nseen++;
                    if (!first) sb_puts(&b, ",");
                    first = 0;
                    {
                        char tmp[64];
                        snprintf(tmp, sizeof(tmp), "[%d,%d]", bi.level_x,
                                 bi.level_y);
                        sb_puts(&b, tmp);
                    }
                }
            }
            if (first) sb_puts(&b, "[0,0]");
        } else {
            sb_puts(&b, "[0,0]");
        }
        sb_puts(&b, "]");

        sb_puts(&b, ",\"channels\":[");
        for (c = 0; c < hd->num_channels; ++c) {
            const exr_channel *ch = &hd->channels[c];
            if (c) sb_puts(&b, ",");
            sb_puts(&b, "{\"name\":");
            sb_json_str(&b, ch->name);
            sb_puts(&b, ",\"type\":");
            sb_json_str(&b, pixeltype_name(ch->pixel_type));
            sb_puts(&b, ",");
            sb_kv_int(&b, "xSampling", ch->x_sampling);
            sb_puts(&b, ",");
            sb_kv_int(&b, "ySampling", ch->y_sampling);
            sb_puts(&b, "}");
        }
        sb_puts(&b, "]");

        sb_puts(&b, "}");
    }
    sb_puts(&b, "]}");

    if (b.oom) {
        free(b.buf);
        return NULL;
    }
    return b.buf;
}

/*
 * Select one (part, level) for decoding. Builds the list of global block
 * indices that belong to it, computes the level's pixel dimensions, maps the
 * R/G/B/A channels, and allocates a cleared RGBA accumulation buffer.
 * Returns the number of blocks to decode (>= 0), or -1 on error / unsupported
 * (e.g. a deep part).
 */
EXRV_EXPORT int exrv_select(int h, int part, int level_x, int level_y) {
    exrv_session *s = session_from_handle(h);
    const exr_header *hd;
    uint32_t nb = 0, i;
    int lw = 0, lh = 0, count = 0, k;
    static const char *names[4] = {"R", "G", "B", "A"};
    if (!s) return -1;
    if (part < 0 || part >= s->num_parts) return -1;

    hd = exr_reader_part_header(s->r, part);
    if (!hd) return -1;
    if (hd->part_type == EXR_PART_DEEP_SCANLINE ||
        hd->part_type == EXR_PART_DEEP_TILED)
        return -1; /* deep not supported by this viewer */

    session_reset_selection(s);
    s->sel_part = part;
    s->dw_min_x = hd->data_window.min_x;
    s->dw_min_y = hd->data_window.min_y;
    for (k = 0; k < 4; ++k) s->rmap[k] = find_channel(hd, names[k]);

    /* Fallbacks for parts without conventional R/G/B so they render instead of
     * showing black — common in multipart EXRs (depth Z, motion vectors,
     * disparity, masks) and luminance/multiview images. */
    if (s->rmap[0] < 0 && s->rmap[1] < 0 && s->rmap[2] < 0) {
        int y = find_channel(hd, "Y");
        if (y >= 0) {
            /* Luminance (multiview Fog.exr, luminance-chroma): Y -> grayscale.
             * Subsampled chroma (RY/BY) is not reconstructed. */
            s->rmap[0] = s->rmap[1] = s->rmap[2] = y;
        } else if (hd->num_channels == 1) {
            /* Single data channel (depth Z, mask): replicate to grayscale. */
            s->rmap[0] = s->rmap[1] = s->rmap[2] = 0;
        } else if (hd->num_channels >= 2) {
            /* Multi-channel data (motion vectors forward.u/v, disparity .x/.y):
             * map the leading channels to R, G and (if present) B. */
            s->rmap[0] = 0;
            s->rmap[1] = 1;
            s->rmap[2] = (hd->num_channels >= 3) ? 2 : -1;
        }
    }

    if (!EXR_OK(exr_reader_num_blocks(s->r, part, &nb))) return -1;

    s->blocks = (uint32_t *)malloc((nb ? nb : 1) * sizeof(uint32_t));
    if (!s->blocks) return -1;

    for (i = 0; i < nb; ++i) {
        exr_block_info bi;
        int dx, dy;
        if (!EXR_OK(exr_reader_block_info(s->r, part, i, &bi))) continue;
        if (bi.level_x != level_x || bi.level_y != level_y) continue;
        s->blocks[count++] = i;
        dx = bi.x0 - s->dw_min_x;
        dy = bi.y0 - s->dw_min_y;
        if (dx + bi.width > lw) lw = dx + bi.width;
        if (dy + bi.height > lh) lh = dy + bi.height;
    }
    s->nblocks = count;
    s->lw = lw;
    s->lh = lh;

    if (lw <= 0 || lh <= 0) {
        session_reset_selection(s);
        s->sel_part = part;
        return -1;
    }

    s->rgba = (float *)malloc((size_t)lw * (size_t)lh * 4 * sizeof(float));
    if (!s->rgba) {
        session_reset_selection(s);
        return -1;
    }
    /* default RGB = 0, A = 1 (so images lacking an alpha channel are opaque) */
    {
        size_t n = (size_t)lw * (size_t)lh, p;
        for (p = 0; p < n; ++p) {
            s->rgba[p * 4 + 0] = 0.0f;
            s->rgba[p * 4 + 1] = 0.0f;
            s->rgba[p * 4 + 2] = 0.0f;
            s->rgba[p * 4 + 3] = 1.0f;
        }
    }
    return count;
}

EXRV_EXPORT int exrv_level_width(int h) {
    exrv_session *s = session_from_handle(h);
    return s ? s->lw : 0;
}

EXRV_EXPORT int exrv_level_height(int h) {
    exrv_session *s = session_from_handle(h);
    return s ? s->lh : 0;
}

/* Scatter one channel's planar samples (native pixel type) into the RGBA
 * accumulation buffer at the block's offset. Only full-resolution channels
 * (x/y sampling == 1) are mapped; subsampled channels are skipped (rare for
 * R/G/B/A and not needed for a viewer). */
static void scatter_channel(exrv_session *s, const exr_block_info *bi,
                            int comp, const void *planar, exr_pixel_type type) {
    int row, col, bw = bi->width, bh = bi->height;
    int dx = bi->x0 - s->dw_min_x;
    int dy = bi->y0 - s->dw_min_y;
    const uint16_t *ph = (const uint16_t *)planar;
    const float *pf = (const float *)planar;
    const uint32_t *pu = (const uint32_t *)planar;

    for (row = 0; row < bh; ++row) {
        int yy = dy + row;
        if (yy < 0 || yy >= s->lh) continue;
        for (col = 0; col < bw; ++col) {
            int xx = dx + col;
            size_t si = (size_t)row * (size_t)bw + (size_t)col;
            size_t di;
            float v;
            if (xx < 0 || xx >= s->lw) continue;
            switch (type) {
            case EXR_PIXEL_HALF: {
                float f;
                exr_half_to_float(ph + si, &f, 1);
                v = f;
                break;
            }
            case EXR_PIXEL_FLOAT:
                v = pf[si];
                break;
            case EXR_PIXEL_UINT:
                v = (float)pu[si];
                break;
            default:
                v = 0.0f;
                break;
            }
            di = ((size_t)yy * (size_t)s->lw + (size_t)xx) * 4 + (size_t)comp;
            s->rgba[di] = v;
        }
    }
}

/*
 * Decode the i-th block of the current selection into the RGBA accumulation
 * buffer. Returns 1 on success, 0 if i is out of range, -1 on decode error.
 */
EXRV_EXPORT int exrv_decode_block(int h, int i) {
    exrv_session *s = session_from_handle(h);
    const exr_header *hd;
    exr_block_info bi;
    uint8_t *block = NULL;
    uint8_t *chan = NULL;
    size_t chan_cap;
    int k, rc_ret = -1;

    if (!s || !s->rgba || !s->blocks) return -1;
    if (i < 0 || i >= s->nblocks) return 0;
    hd = exr_reader_part_header(s->r, s->sel_part);
    if (!hd) return -1;

    if (!EXR_OK(exr_reader_block_info(s->r, s->sel_part, s->blocks[i], &bi)))
        return -1;

    block = (uint8_t *)malloc(bi.uncompressed_size ? bi.uncompressed_size : 1);
    if (!block) return -1;
    if (!EXR_OK(exr_reader_decode_block(s->r, s->sel_part, s->blocks[i], block,
                                        bi.uncompressed_size)))
        goto done;

    /* scratch for one extracted channel (full-res block: width*height elems) */
    chan_cap = (size_t)bi.width * (size_t)bi.height * 4; /* worst-case 4 bytes */
    chan = (uint8_t *)malloc(chan_cap ? chan_cap : 1);
    if (!chan) goto done;

    for (k = 0; k < 4; ++k) {
        int ci = s->rmap[k];
        exr_pixel_type type;
        if (ci < 0) continue;
        if (hd->channels[ci].x_sampling != 1 || hd->channels[ci].y_sampling != 1)
            continue; /* skip subsampled channels */
        type = hd->channels[ci].pixel_type;
        if (!EXR_OK(exr_block_extract_channel(hd, &bi, block,
                                              bi.uncompressed_size, ci, chan)))
            goto done;
        scatter_channel(s, &bi, k, chan, type);
    }
    rc_ret = 1;

done:
    free(block);
    free(chan);
    return rc_ret;
}

EXRV_EXPORT float *exrv_rgba(int h) {
    exrv_session *s = session_from_handle(h);
    return s ? s->rgba : NULL;
}

EXRV_EXPORT void exrv_free(void *p) { free(p); }
