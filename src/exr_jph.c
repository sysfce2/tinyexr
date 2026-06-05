/*
 * TinyEXR - HTJ2K/JPH codec front end.
 *
 * This file is clean-room C11 code for the OpenEXR HTJ2K chunk wrapper and
 * codestream profile validation. The HT block entropy decoder itself is not
 * implemented in this milestone; real compressed codestreams fail with
 * EXR_ERROR_UNSUPPORTED after all bounded header validation has completed.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr_internal.h"

#include <limits.h>

enum {
    JPH_MARK_SOC = 0xff4f,
    JPH_MARK_CAP = 0xff50,
    JPH_MARK_SIZ = 0xff51,
    JPH_MARK_COD = 0xff52,
    JPH_MARK_COC = 0xff53,
    JPH_MARK_TLM = 0xff55,
    JPH_MARK_PRF = 0xff56,
    JPH_MARK_PLM = 0xff57,
    JPH_MARK_PLT = 0xff58,
    JPH_MARK_CPF = 0xff59,
    JPH_MARK_QCD = 0xff5c,
    JPH_MARK_QCC = 0xff5d,
    JPH_MARK_RGN = 0xff5e,
    JPH_MARK_POC = 0xff5f,
    JPH_MARK_PPM = 0xff60,
    JPH_MARK_PPT = 0xff61,
    JPH_MARK_CRG = 0xff63,
    JPH_MARK_COM = 0xff64,
    JPH_MARK_DFS = 0xff72,
    JPH_MARK_ADS = 0xff73,
    JPH_MARK_NLT = 0xff76,
    JPH_MARK_ATK = 0xff79,
    JPH_MARK_SOT = 0xff90,
    JPH_MARK_SOD = 0xff93,
    JPH_MARK_EOC = 0xffd9
};

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} JphReader;

typedef struct {
    uint32_t xsiz, ysiz, xosiz, yosiz;
    uint32_t xtsiz, ytsiz, xtosiz, ytosiz;
    uint32_t psot;
    const uint8_t *sot_start;
    const uint8_t *tile_data;
    size_t tile_data_size;
    uint16_t rsiz, csiz;
    uint8_t *ssiz;
    uint8_t *xrsiz;
    uint8_t *yrsiz;
    uint8_t *nlt_type;
    uint8_t qcd_exp[16];
    uint8_t qcd_count;
    uint8_t qcd_guard_bits;
    uint8_t num_decomps;
    uint8_t mc_trans;
    int saw_siz, saw_cap, saw_cod, saw_qcd, saw_sot;
    int saw_sod, saw_qcc;
} JphProfile;

typedef struct {
    uint32_t w, h;
} JphSize;

typedef struct {
    uint32_t w, h;
    uint32_t cb_w, cb_h;
    uint32_t kmax;
    uint8_t exists;
} JphBandGeom;

typedef struct {
    uint32_t num_w, num_h;
    uint32_t cur_x, cur_y;
} JphPrecinctState;

typedef struct {
    uint32_t comp;
    uint32_t res;
    uint32_t band;
    uint32_t x0;
    uint32_t y0;
    uint32_t missing_msbs;
    uint32_t active_passes;
    uint32_t length0;
    uint32_t length1;
    uint32_t width;
    uint32_t height;
    const uint8_t *data;
    size_t data_size;
} JphCodeblockSeg;

typedef struct {
    JphCodeblockSeg *items;
    size_t count;
    size_t cap;
} JphCodeblockList;

typedef exr_result (*JphCodeblockCallback)(void *user,
                                           const JphCodeblockSeg *seg);

typedef struct {
    const uint8_t *magsgn;
    size_t magsgn_size;
    const uint8_t *mel;
    size_t mel_size;
    const uint8_t *vlc;
    size_t vlc_size;
    const uint8_t *refine;
    size_t refine_size;
    uint32_t scup;
} JphCodeblockStreams;

typedef struct {
    int32_t *data;
    uint32_t w;
    uint32_t h;
} JphComponentPlane;

static uint16_t jph_be16(const uint8_t *p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static uint32_t jph_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int jph_read_u8(JphReader *r, uint8_t *v) {
    if (r->p >= r->end) return 0;
    *v = *r->p++;
    return 1;
}

static int jph_read_be16(JphReader *r, uint16_t *v) {
    if ((size_t)(r->end - r->p) < 2) return 0;
    *v = jph_be16(r->p);
    r->p += 2;
    return 1;
}

static int jph_read_be32(JphReader *r, uint32_t *v) {
    if ((size_t)(r->end - r->p) < 4) return 0;
    *v = jph_be32(r->p);
    r->p += 4;
    return 1;
}

static exr_result jph_skip_segment(JphReader *r) {
    uint16_t len;
    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len < 2) return EXR_ERROR_CORRUPT;
    if ((size_t)(r->end - r->p) < (size_t)(len - 2)) return EXR_ERROR_CORRUPT;
    r->p += (size_t)(len - 2);
    return EXR_SUCCESS;
}

static exr_result jph_next_marker(JphReader *r, uint16_t *marker) {
    uint8_t b;
    while (jph_read_u8(r, &b)) {
        if (b == 0xff) {
            do {
                if (!jph_read_u8(r, &b)) return EXR_ERROR_CORRUPT;
            } while (b == 0xff);
            if (b == 0x00) continue;
            *marker = (uint16_t)(0xff00u | b);
            return EXR_SUCCESS;
        }
    }
    return EXR_ERROR_CORRUPT;
}

static exr_result jph_find_eoc(JphReader *r, const uint8_t **payload_end) {
    uint8_t b;
    while (jph_read_u8(r, &b)) {
        if (b == 0xff) {
            do {
                if (!jph_read_u8(r, &b)) return EXR_ERROR_CORRUPT;
            } while (b == 0xff);
            if (b == 0x00) continue;
            if ((uint16_t)(0xff00u | b) == JPH_MARK_EOC) {
                *payload_end = r->p - 2;
                return EXR_SUCCESS;
            }
            return EXR_ERROR_UNSUPPORTED;
        }
    }
    return EXR_ERROR_CORRUPT;
}

void exr_jph_bitreader_init(exr_jph_bitreader *br, const uint8_t *data,
                            size_t size) {
    if (!br) return;
    br->p = data;
    br->end = data ? data + size : data;
    br->bits = 0;
    br->bit_count = 0;
    br->prev_ff = 0;
}

static exr_result jph_bitreader_fill(exr_jph_bitreader *br, unsigned need) {
    while (br->bit_count < need) {
        uint8_t byte;
        unsigned nbits = 8;
        if (!br || !br->p || br->p >= br->end) return EXR_ERROR_CORRUPT;
        byte = *br->p++;
        if (br->prev_ff) {
            if (byte & 0x80u) return EXR_ERROR_CORRUPT;
            nbits = 7;
            byte = (uint8_t)(byte & 0x7fu);
        }
        if (br->bit_count + nbits > 64u) return EXR_ERROR_CORRUPT;
        br->bits = (br->bits << nbits) | (uint64_t)byte;
        br->bit_count += nbits;
        br->prev_ff = (byte == 0xffu && nbits == 8u);
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_bitreader_read(exr_jph_bitreader *br, unsigned nbits,
                                  uint32_t *out) {
    exr_result rc;
    uint64_t mask;
    if (!br || !out || nbits > 32u) return EXR_ERROR_INVALID_ARGUMENT;
    if (nbits == 0) {
        *out = 0;
        return EXR_SUCCESS;
    }
    rc = jph_bitreader_fill(br, nbits);
    if (rc != EXR_SUCCESS) return rc;
    mask = (nbits == 32u) ? UINT64_C(0xffffffff) : ((UINT64_C(1) << nbits) - 1u);
    *out = (uint32_t)((br->bits >> (br->bit_count - nbits)) & mask);
    br->bit_count -= nbits;
    if (br->bit_count == 0) br->bits = 0;
    else br->bits &= ((UINT64_C(1) << br->bit_count) - 1u);
    return EXR_SUCCESS;
}

void exr_jph_bitreader_align(exr_jph_bitreader *br) {
    unsigned drop;
    if (!br) return;
    drop = br->bit_count & 7u;
    if (drop) {
        br->bit_count -= drop;
        if (br->bit_count == 0) br->bits = 0;
        else br->bits &= ((UINT64_C(1) << br->bit_count) - 1u);
    }
}

void exr_jph_ht_forward_init(exr_jph_ht_forward_reader *r,
                             const uint8_t *data, size_t size,
                             uint8_t fill_byte) {
    if (!r) return;
    r->p = data;
    r->end = data ? data + size : data;
    r->bits = 0;
    r->bit_count = 0;
    r->fill_byte = fill_byte;
    r->prev_ff = 0;
}

static exr_result jph_ht_forward_fill(exr_jph_ht_forward_reader *r,
                                      unsigned need) {
    while (r->bit_count < need) {
        uint8_t byte;
        unsigned nbits = 8;
        int have_data = 0;
        if (!r) return EXR_ERROR_INVALID_ARGUMENT;
        if (r->p && r->p < r->end) {
            byte = *r->p++;
            have_data = 1;
        } else {
            byte = r->fill_byte;
        }
        if (r->prev_ff) {
            if (have_data && (byte & 0x80u)) return EXR_ERROR_CORRUPT;
            byte &= 0x7fu;
            nbits = 7;
        }
        if (r->bit_count + nbits > 64u) return EXR_ERROR_CORRUPT;
        r->bits |= (uint64_t)byte << r->bit_count;
        r->bit_count += nbits;
        r->prev_ff = (nbits == 8u && byte == 0xffu);
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_ht_forward_read(exr_jph_ht_forward_reader *r,
                                   unsigned nbits, uint32_t *out) {
    exr_result rc;
    uint64_t mask;
    if (!r || !out || nbits > 32u) return EXR_ERROR_INVALID_ARGUMENT;
    if (nbits == 0) {
        *out = 0;
        return EXR_SUCCESS;
    }
    rc = jph_ht_forward_fill(r, nbits);
    if (rc != EXR_SUCCESS) return rc;
    mask = (nbits == 32u) ? UINT64_C(0xffffffff) : ((UINT64_C(1) << nbits) - 1u);
    *out = (uint32_t)(r->bits & mask);
    r->bits >>= nbits;
    r->bit_count -= nbits;
    return EXR_SUCCESS;
}

void exr_jph_ht_reverse_init(exr_jph_ht_reverse_reader *r,
                             const uint8_t *data, size_t size,
                             int initial_unstuff, int zero_fill) {
    if (!r) return;
    r->start = data;
    r->p = data ? data + size : data;
    r->bits = 0;
    r->bit_count = 0;
    r->unstuff = initial_unstuff != 0;
    r->zero_fill = zero_fill != 0;
}

static exr_result jph_ht_reverse_fill(exr_jph_ht_reverse_reader *r,
                                      unsigned need) {
    while (r->bit_count < need) {
        uint8_t byte = 0;
        unsigned nbits = 8;
        if (!r) return EXR_ERROR_INVALID_ARGUMENT;
        if (r->p && r->p > r->start) byte = *--r->p;
        else if (!r->zero_fill) byte = 0;
        if (r->unstuff && ((byte & 0x7fu) == 0x7fu)) {
            if (byte & 0x80u) return EXR_ERROR_CORRUPT;
            nbits = 7;
        }
        if (r->bit_count + nbits > 64u) return EXR_ERROR_CORRUPT;
        r->bits |= (uint64_t)(byte & ((1u << nbits) - 1u)) << r->bit_count;
        r->bit_count += nbits;
        r->unstuff = byte > 0x8fu;
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_ht_reverse_read(exr_jph_ht_reverse_reader *r,
                                   unsigned nbits, uint32_t *out) {
    exr_result rc;
    uint64_t mask;
    if (!r || !out || nbits > 32u) return EXR_ERROR_INVALID_ARGUMENT;
    if (nbits == 0) {
        *out = 0;
        return EXR_SUCCESS;
    }
    rc = jph_ht_reverse_fill(r, nbits);
    if (rc != EXR_SUCCESS) return rc;
    mask = (nbits == 32u) ? UINT64_C(0xffffffff) : ((UINT64_C(1) << nbits) - 1u);
    *out = (uint32_t)(r->bits & mask);
    r->bits >>= nbits;
    r->bit_count -= nbits;
    return EXR_SUCCESS;
}

void exr_jph_mel_init(exr_jph_mel_reader *r, const uint8_t *data,
                      size_t size) {
    if (!r) return;
    r->p = data;
    r->end = data ? data + size : data;
    r->bits = 0;
    r->bit_count = 0;
    r->prev_ff = 0;
    r->k = 0;
}

static exr_result jph_mel_read_bits(exr_jph_mel_reader *r, unsigned nbits,
                                    uint32_t *out) {
    uint64_t mask;
    if (!r || !out || nbits > 32u) return EXR_ERROR_INVALID_ARGUMENT;
    while (r->bit_count < nbits) {
        uint8_t byte;
        unsigned add_bits = 8;
        if (!r->p || r->p >= r->end) return EXR_ERROR_CORRUPT;
        byte = *r->p++;
        if (r->prev_ff) {
            if (byte > 0x8fu) return EXR_ERROR_CORRUPT;
            byte &= 0x7fu;
            add_bits = 7;
        }
        if (r->bit_count + add_bits > 64u) return EXR_ERROR_CORRUPT;
        r->bits = (r->bits << add_bits) | (uint64_t)byte;
        r->bit_count += add_bits;
        r->prev_ff = (add_bits == 8u && byte == 0xffu);
    }
    if (nbits == 0) {
        *out = 0;
        return EXR_SUCCESS;
    }
    mask = (nbits == 32u) ? UINT64_C(0xffffffff) : ((UINT64_C(1) << nbits) - 1u);
    *out = (uint32_t)((r->bits >> (r->bit_count - nbits)) & mask);
    r->bit_count -= nbits;
    if (r->bit_count == 0) r->bits = 0;
    else r->bits &= ((UINT64_C(1) << r->bit_count) - 1u);
    return EXR_SUCCESS;
}

exr_result exr_jph_mel_get_run(exr_jph_mel_reader *r, uint32_t *zero_run,
                               int *has_one) {
    static const uint8_t mel_exp[13] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5
    };
    uint32_t bit = 0, v = 0;
    uint8_t eval;
    exr_result rc;
    if (!r || !zero_run || !has_one || r->k < 0 || r->k > 12)
        return EXR_ERROR_INVALID_ARGUMENT;
    eval = mel_exp[r->k];
    rc = jph_mel_read_bits(r, 1, &bit);
    if (rc != EXR_SUCCESS) return rc;
    if (bit) {
        *zero_run = (1u << eval) - 1u;
        *has_one = 0;
        if (r->k < 12) r->k++;
        return EXR_SUCCESS;
    }
    if (eval) {
        rc = jph_mel_read_bits(r, eval, &v);
        if (rc != EXR_SUCCESS) return rc;
    }
    *zero_run = v;
    *has_one = 1;
    if (r->k > 0) r->k--;
    return EXR_SUCCESS;
}

static uint32_t jph_bit_length_u32(uint32_t v) {
    uint32_t n = 0;
    while (v) {
        n++;
        v >>= 1u;
    }
    return n;
}

exr_result exr_jph_packet_read_pass_count(exr_jph_bitreader *br,
                                          uint32_t *raw_passes,
                                          uint32_t *active_passes,
                                          uint32_t *placeholder_groups) {
    uint32_t bit = 0, passes = 1, groups;
    exr_result rc;
    if (!br || !raw_passes || !active_passes || !placeholder_groups)
        return EXR_ERROR_INVALID_ARGUMENT;

    rc = exr_jph_bitreader_read(br, 1, &bit);
    if (rc != EXR_SUCCESS) return rc;
    if (bit) {
        passes = 2;
        rc = exr_jph_bitreader_read(br, 1, &bit);
        if (rc != EXR_SUCCESS) return rc;
        if (bit) {
            rc = exr_jph_bitreader_read(br, 2, &bit);
            if (rc != EXR_SUCCESS) return rc;
            passes = 3u + bit;
            if (bit == 3u) {
                rc = exr_jph_bitreader_read(br, 5, &bit);
                if (rc != EXR_SUCCESS) return rc;
                passes = 6u + bit;
                if (bit == 31u) {
                    rc = exr_jph_bitreader_read(br, 7, &bit);
                    if (rc != EXR_SUCCESS) return rc;
                    passes = 37u + bit;
                }
            }
        }
    }

    groups = (passes - 1u) / 3u;
    *raw_passes = passes;
    *placeholder_groups = groups;
    *active_passes = passes - groups * 3u;
    if (*active_passes == 0u || *active_passes > 3u) return EXR_ERROR_CORRUPT;
    return EXR_SUCCESS;
}

exr_result exr_jph_packet_read_pass_lengths(exr_jph_bitreader *br,
                                            uint32_t active_passes,
                                            uint32_t placeholder_groups,
                                            uint32_t lengths[2]) {
    uint32_t bit = 1, lblock = 3, nbits, v;
    exr_result rc;
    if (!br || !lengths) return EXR_ERROR_INVALID_ARGUMENT;
    if (active_passes == 0u || active_passes > 3u)
        return EXR_ERROR_INVALID_ARGUMENT;
    lengths[0] = 0;
    lengths[1] = 0;

    while (bit) {
        rc = exr_jph_bitreader_read(br, 1, &bit);
        if (rc != EXR_SUCCESS) return rc;
        if (bit) {
            lblock++;
            if (lblock > 32u) return EXR_ERROR_CORRUPT;
        }
    }

    nbits = lblock + jph_bit_length_u32(placeholder_groups + 1u) - 1u;
    if (nbits > 32u) return EXR_ERROR_CORRUPT;
    rc = exr_jph_bitreader_read(br, nbits, &v);
    if (rc != EXR_SUCCESS) return rc;
    if (v < 2u || v >= 65535u) return EXR_ERROR_CORRUPT;
    lengths[0] = v;

    if (active_passes > 1u) {
        nbits = lblock + (active_passes > 2u ? 1u : 0u);
        if (nbits > 32u) return EXR_ERROR_CORRUPT;
        rc = exr_jph_bitreader_read(br, nbits, &v);
        if (rc != EXR_SUCCESS) return rc;
        if (v >= 2047u) return EXR_ERROR_CORRUPT;
        lengths[1] = v;
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_tag_tree_init(const exr_allocator *a,
                                 exr_jph_tag_tree *tree, uint32_t width,
                                 uint32_t height) {
    uint32_t w = width, h = height, level = 0;
    size_t total = 0;
    if (!tree) return EXR_ERROR_INVALID_ARGUMENT;
    if (!a) a = exr_default_allocator();
    memset(tree, 0, sizeof(*tree));
    if (width == 0 || height == 0) return EXR_ERROR_INVALID_ARGUMENT;
    while (1) {
        size_t nodes;
        if (level >= EXR_JPH_TAGTREE_MAX_LEVELS) return EXR_ERROR_CORRUPT;
        tree->width[level] = w;
        tree->height[level] = h;
        tree->offset[level] = total;
        if (exr_mul_ovf((size_t)w, (size_t)h, &nodes) ||
            exr_add_ovf(total, nodes, &total))
            return EXR_ERROR_CORRUPT;
        level++;
        if (w == 1u && h == 1u) break;
        w = (w + 1u) / 2u;
        h = (h + 1u) / 2u;
    }
    tree->num_levels = level;
    tree->node_count = total;
    tree->value = (uint32_t *)exr_calloc(a, total, sizeof(uint32_t));
    tree->known = (uint8_t *)exr_calloc(a, total, 1);
    if (!tree->value || !tree->known) {
        exr_jph_tag_tree_free(a, tree);
        return EXR_ERROR_OUT_OF_MEMORY;
    }
    return EXR_SUCCESS;
}

void exr_jph_tag_tree_free(const exr_allocator *a, exr_jph_tag_tree *tree) {
    if (!tree) return;
    if (!a) a = exr_default_allocator();
    exr_free(a, tree->value);
    exr_free(a, tree->known);
    memset(tree, 0, sizeof(*tree));
}

exr_result exr_jph_tag_tree_decode(exr_jph_tag_tree *tree,
                                   exr_jph_bitreader *br, uint32_t leaf_x,
                                   uint32_t leaf_y, uint32_t threshold,
                                   uint32_t *out_value) {
    uint32_t xs[EXR_JPH_TAGTREE_MAX_LEVELS];
    uint32_t ys[EXR_JPH_TAGTREE_MAX_LEVELS];
    uint32_t level, low = 0;
    if (!tree || !br || !out_value || !tree->value || !tree->known)
        return EXR_ERROR_INVALID_ARGUMENT;
    if (tree->num_levels == 0 || leaf_x >= tree->width[0] ||
        leaf_y >= tree->height[0])
        return EXR_ERROR_INVALID_ARGUMENT;

    xs[0] = leaf_x;
    ys[0] = leaf_y;
    for (level = 1; level < tree->num_levels; ++level) {
        xs[level] = xs[level - 1u] >> 1u;
        ys[level] = ys[level - 1u] >> 1u;
    }

    for (level = tree->num_levels; level > 0; --level) {
        uint32_t li = level - 1u;
        size_t idx = tree->offset[li] +
                     (size_t)ys[li] * tree->width[li] + xs[li];
        if (idx >= tree->node_count) return EXR_ERROR_CORRUPT;
        if (tree->value[idx] < low) tree->value[idx] = low;
        while (tree->value[idx] < threshold && !tree->known[idx]) {
            uint32_t bit = 0;
            exr_result rc = exr_jph_bitreader_read(br, 1, &bit);
            if (rc != EXR_SUCCESS) return rc;
            if (bit) tree->known[idx] = 1;
            else tree->value[idx]++;
        }
        low = tree->value[idx];
    }
    *out_value = low;
    return EXR_SUCCESS;
}

static uint32_t jph_divceil_u32(uint32_t a, uint32_t b) {
    return b ? (a / b + ((a % b) != 0)) : 0;
}

static uint32_t jph_ceil_div_pow2_u32(uint32_t v, uint32_t shift) {
    while (shift--) v = (v + 1u) >> 1u;
    return v;
}

static int64_t jph_floor_div_pow2(int64_t v, unsigned shift) {
    int64_t d = (int64_t)1 << shift;
    if (v >= 0) return v / d;
    return -(((-v) + d - 1) / d);
}

static exr_result jph_i64_to_i32(int64_t v, int32_t *out) {
    if (v < (int64_t)INT32_MIN || v > (int64_t)INT32_MAX)
        return EXR_ERROR_CORRUPT;
    *out = (int32_t)v;
    return EXR_SUCCESS;
}

exr_result exr_jph_inverse_rct_i32(int32_t *c0, int32_t *c1, int32_t *c2,
                                   size_t count) {
    size_t i;
    if ((!c0 || !c1 || !c2) && count) return EXR_ERROR_INVALID_ARGUMENT;
    for (i = 0; i < count; ++i) {
        int64_t y = c0[i];
        int64_t db = c1[i];
        int64_t dr = c2[i];
        int64_t g = y - jph_floor_div_pow2(db + dr, 2);
        int64_t r = dr + g;
        int64_t b = db + g;
        exr_result rc = jph_i64_to_i32(r, &c0[i]);
        if (rc != EXR_SUCCESS) return rc;
        rc = jph_i64_to_i32(g, &c1[i]);
        if (rc != EXR_SUCCESS) return rc;
        rc = jph_i64_to_i32(b, &c2[i]);
        if (rc != EXR_SUCCESS) return rc;
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_apply_nlt_type3_i32(int32_t *data, size_t count,
                                       uint32_t bit_depth) {
    size_t i;
    int64_t bias;
    if (!data && count) return EXR_ERROR_INVALID_ARGUMENT;
    if (bit_depth == 0 || bit_depth > 32u) return EXR_ERROR_INVALID_ARGUMENT;
    bias = ((int64_t)1 << (bit_depth - 1u)) + 1;
    for (i = 0; i < count; ++i) {
        int64_t v = data[i];
        if (v < 0) {
            exr_result rc = jph_i64_to_i32(-v - bias, &data[i]);
            if (rc != EXR_SUCCESS) return rc;
        }
    }
    return EXR_SUCCESS;
}

exr_result exr_jph_inverse_53_i32(const int32_t *low, size_t low_count,
                                  const int32_t *high, size_t high_count,
                                  int32_t *out, size_t out_count) {
    size_t i;
    size_t expected_low = (out_count + 1u) / 2u;
    size_t expected_high = out_count / 2u;

    if (!out && out_count) return EXR_ERROR_INVALID_ARGUMENT;
    if (low_count != expected_low || high_count != expected_high)
        return EXR_ERROR_INVALID_ARGUMENT;
    if ((!low && low_count) || (!high && high_count))
        return EXR_ERROR_INVALID_ARGUMENT;
    if (out_count == 0) return EXR_SUCCESS;
    if (high_count == 0) {
        out[0] = low[0];
        return EXR_SUCCESS;
    }

    for (i = 0; i < low_count; ++i) {
        int64_t dl = high[i > 0 ? i - 1u : 0u];
        int64_t dr = high[i < high_count ? i : high_count - 1u];
        int64_t even = (int64_t)low[i] - jph_floor_div_pow2(dl + dr + 2, 2);
        exr_result rc = jph_i64_to_i32(even, &out[2u * i]);
        if (rc != EXR_SUCCESS) return rc;
    }
    for (i = 0; i < high_count; ++i) {
        int64_t e0 = out[2u * i];
        int64_t e1 = (i + 1u < low_count) ? out[2u * (i + 1u)] : e0;
        int64_t odd = (int64_t)high[i] + jph_floor_div_pow2(e0 + e1, 1);
        exr_result rc = jph_i64_to_i32(odd, &out[2u * i + 1u]);
        if (rc != EXR_SUCCESS) return rc;
    }
    return EXR_SUCCESS;
}

static size_t jph_ceil_div_pow2_size(size_t v, unsigned shift) {
    while (shift--) v = (v + 1u) / 2u;
    return v;
}

exr_result exr_jph_inverse_53_2d_i32(const exr_allocator *a, int32_t *data,
                                     size_t width, size_t height,
                                     unsigned levels) {
    unsigned level;
    if (!a) a = exr_default_allocator();
    if (!data && width && height) return EXR_ERROR_INVALID_ARGUMENT;
    if (levels > 32) return EXR_ERROR_INVALID_ARGUMENT;
    if (width == 0 || height == 0 || levels == 0) return EXR_SUCCESS;

    for (level = levels; level > 0; --level) {
        size_t rw = jph_ceil_div_pow2_size(width, level - 1u);
        size_t rh = jph_ceil_div_pow2_size(height, level - 1u);
        size_t lw = (rw + 1u) / 2u, hw = rw / 2u;
        size_t lh = (rh + 1u) / 2u, hh = rh / 2u;
        size_t temp_count, temp_bytes, col_bytes;
        int32_t *temp = NULL, *col_low = NULL, *col_high = NULL, *col_out = NULL;
        size_t y, x;
        exr_result rc = EXR_SUCCESS;

        if (rw == 0 || rh == 0) return EXR_ERROR_CORRUPT;
        if (exr_mul_ovf(rw, rh, &temp_count) ||
            exr_mul_ovf(temp_count, sizeof(int32_t), &temp_bytes) ||
            exr_mul_ovf(rh, sizeof(int32_t), &col_bytes))
            return EXR_ERROR_CORRUPT;

        temp = (int32_t *)exr_malloc(a, temp_bytes);
        col_low = (int32_t *)exr_malloc(a, col_bytes);
        col_high = (int32_t *)exr_malloc(a, col_bytes);
        col_out = (int32_t *)exr_malloc(a, col_bytes);
        if (!temp || !col_low || !col_high || !col_out) {
            exr_free(a, temp);
            exr_free(a, col_low);
            exr_free(a, col_high);
            exr_free(a, col_out);
            return EXR_ERROR_OUT_OF_MEMORY;
        }

        for (y = 0; y < rh; ++y) {
            const int32_t *row = data + y * width;
            rc = exr_jph_inverse_53_i32(row, lw, row + lw, hw,
                                        temp + y * rw, rw);
            if (rc != EXR_SUCCESS) goto done_level;
        }
        for (x = 0; x < rw; ++x) {
            for (y = 0; y < lh; ++y) col_low[y] = temp[y * rw + x];
            for (y = 0; y < hh; ++y) col_high[y] = temp[(lh + y) * rw + x];
            rc = exr_jph_inverse_53_i32(col_low, lh, col_high, hh, col_out, rh);
            if (rc != EXR_SUCCESS) goto done_level;
            for (y = 0; y < rh; ++y) data[y * width + x] = col_out[y];
        }

done_level:
        exr_free(a, temp);
        exr_free(a, col_low);
        exr_free(a, col_high);
        exr_free(a, col_out);
        if (rc != EXR_SUCCESS) return rc;
    }
    return EXR_SUCCESS;
}

static exr_result jph_validate_siz_component(const exr_codec_ctx *ctx,
                                             const JphProfile *jp,
                                             const uint16_t *map,
                                             uint16_t c) {
    const exr_channel *ch;
    uint16_t file_c = map[c];
    uint32_t bit_depth = (uint32_t)(jp->ssiz[c] & 0x7fu) + 1u;
    int is_signed = (jp->ssiz[c] & 0x80u) != 0;
    uint32_t want_bits, want_signed, recon_w, recon_h;

    if (file_c >= (uint16_t)ctx->num_channels) return EXR_ERROR_CORRUPT;
    ch = &ctx->channels[file_c];
    want_bits = (ch->pixel_type == EXR_PIXEL_HALF) ? 16u : 32u;
    want_signed = (ch->pixel_type == EXR_PIXEL_UINT) ? 0u : 1u;
    if (bit_depth != want_bits || (uint32_t)is_signed != want_signed)
        return EXR_ERROR_UNSUPPORTED;
    if (jp->xrsiz[c] == 0 || jp->yrsiz[c] == 0) return EXR_ERROR_CORRUPT;
    if ((int32_t)jp->xrsiz[c] != ch->x_sampling ||
        (int32_t)jp->yrsiz[c] != ch->y_sampling)
        return EXR_ERROR_UNSUPPORTED;

    recon_w = jph_divceil_u32(jp->xsiz, jp->xrsiz[c]) -
              jph_divceil_u32(jp->xosiz, jp->xrsiz[c]);
    recon_h = jph_divceil_u32(jp->ysiz, jp->yrsiz[c]) -
              jph_divceil_u32(jp->yosiz, jp->yrsiz[c]);
    if (recon_w != (uint32_t)exr_num_samples(ctx->x, ctx->x + ctx->width - 1,
                                             ch->x_sampling))
        return EXR_ERROR_CORRUPT;
    if (recon_h != (uint32_t)exr_num_samples(ctx->y, ctx->y + ctx->num_lines - 1,
                                             ch->y_sampling))
        return EXR_ERROR_CORRUPT;
    return EXR_SUCCESS;
}

static exr_result jph_parse_siz(const exr_allocator *a, JphReader *r,
                                JphProfile *jp) {
    uint16_t len, csiz;
    size_t i, remain;

    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len < 38) return EXR_ERROR_CORRUPT;
    if ((size_t)(r->end - r->p) < (size_t)(len - 2)) return EXR_ERROR_CORRUPT;
    remain = (size_t)(len - 2);

    if (remain < 36) return EXR_ERROR_CORRUPT;
    jp->rsiz = jph_be16(r->p); r->p += 2;
    jp->xsiz = jph_be32(r->p); r->p += 4;
    jp->ysiz = jph_be32(r->p); r->p += 4;
    jp->xosiz = jph_be32(r->p); r->p += 4;
    jp->yosiz = jph_be32(r->p); r->p += 4;
    jp->xtsiz = jph_be32(r->p); r->p += 4;
    jp->ytsiz = jph_be32(r->p); r->p += 4;
    jp->xtosiz = jph_be32(r->p); r->p += 4;
    jp->ytosiz = jph_be32(r->p); r->p += 4;
    csiz = jph_be16(r->p); r->p += 2;
    remain -= 36;

    if (csiz == 0) return EXR_ERROR_CORRUPT;
    if (remain != (size_t)csiz * 3u) return EXR_ERROR_CORRUPT;
    jp->ssiz = (uint8_t *)exr_malloc(a, csiz ? csiz : 1);
    jp->xrsiz = (uint8_t *)exr_malloc(a, csiz ? csiz : 1);
    jp->yrsiz = (uint8_t *)exr_malloc(a, csiz ? csiz : 1);
    jp->nlt_type = (uint8_t *)exr_calloc(a, csiz ? csiz : 1, 1);
    if (!jp->ssiz || !jp->xrsiz || !jp->yrsiz || !jp->nlt_type)
        return EXR_ERROR_OUT_OF_MEMORY;
    for (i = 0; i < csiz; ++i) {
        jp->ssiz[i] = r->p[0];
        jp->xrsiz[i] = r->p[1];
        jp->yrsiz[i] = r->p[2];
        r->p += 3;
    }
    jp->csiz = csiz;
    jp->saw_siz = 1;
    return EXR_SUCCESS;
}

static exr_result jph_parse_cap(JphReader *r, JphProfile *jp) {
    uint16_t len;
    uint32_t pcap;
    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len < 6 || ((len - 6u) & 1u)) return EXR_ERROR_CORRUPT;
    if ((size_t)(r->end - r->p) < (size_t)(len - 2)) return EXR_ERROR_CORRUPT;
    pcap = jph_be32(r->p);
    if ((pcap & 0x00020000u) == 0) return EXR_ERROR_UNSUPPORTED;
    r->p += (size_t)(len - 2);
    jp->saw_cap = 1;
    return EXR_SUCCESS;
}

static exr_result jph_parse_cod(JphReader *r, JphProfile *jp) {
    uint16_t len, layers;
    uint8_t scod, prog, mc, ndecomp, bw, bh, style, wavelet;

    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if ((size_t)(r->end - r->p) < (size_t)(len - 2)) return EXR_ERROR_CORRUPT;
    if (len < 12) return EXR_ERROR_CORRUPT;
    scod = r->p[0];
    prog = r->p[1];
    layers = jph_be16(r->p + 2);
    mc = r->p[4];
    ndecomp = r->p[5];
    bw = r->p[6];
    bh = r->p[7];
    style = r->p[8];
    wavelet = r->p[9];
    r->p += 10;
    if (len != (uint16_t)(12u + ((scod & 1u) ? (1u + ndecomp) : 0u)))
        return EXR_ERROR_CORRUPT;
    if (scod != 0u) return EXR_ERROR_UNSUPPORTED;
    if (prog != 2u || layers != 1u) return EXR_ERROR_UNSUPPORTED;
    if (!(mc == 0u || mc == 1u)) return EXR_ERROR_UNSUPPORTED;
    if (ndecomp != 5u || bw != 5u || bh != 3u) return EXR_ERROR_UNSUPPORTED;
    if (style != 0x40u || wavelet != 1u) return EXR_ERROR_UNSUPPORTED;
    jp->num_decomps = ndecomp;
    jp->mc_trans = mc;
    jp->saw_cod = 1;
    return EXR_SUCCESS;
}

static exr_result jph_parse_quant(JphReader *r, JphProfile *jp, int is_qcc) {
    uint16_t len;
    size_t payload, comp_bytes = 0, pos;
    uint8_t sq;

    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len < (is_qcc ? 4u : 3u)) return EXR_ERROR_CORRUPT;
    if ((size_t)(r->end - r->p) < (size_t)(len - 2)) return EXR_ERROR_CORRUPT;
    payload = (size_t)(len - 2);
    if (is_qcc) {
        uint16_t comp_idx;
        if (!jp->saw_siz) return EXR_ERROR_CORRUPT;
        comp_bytes = (jp->csiz < 257u) ? 1u : 2u;
        if (payload < comp_bytes + 1u) return EXR_ERROR_CORRUPT;
        comp_idx = comp_bytes == 1u ? (uint16_t)r->p[0] : jph_be16(r->p);
        if (comp_idx >= jp->csiz) return EXR_ERROR_CORRUPT;
        r->p += comp_bytes;
        payload -= comp_bytes;
        jp->saw_qcc = 1;
    }
    sq = r->p[0];
    r->p++;
    payload--;
    if ((sq & 0x1fu) != 0u) return EXR_ERROR_UNSUPPORTED;
    if ((sq >> 5) > 7u) return EXR_ERROR_CORRUPT;
    pos = 0;
    while (pos < payload) {
        uint8_t sp = r->p[pos];
        uint8_t expn = (uint8_t)(sp >> 3);
        if (expn == 0) return EXR_ERROR_CORRUPT;
        pos++;
    }
    if (!is_qcc) {
        if (payload == 0 || payload > sizeof(jp->qcd_exp))
            return EXR_ERROR_CORRUPT;
        memcpy(jp->qcd_exp, r->p, payload);
        jp->qcd_count = (uint8_t)payload;
        jp->qcd_guard_bits = (uint8_t)(sq >> 5);
    }
    r->p += payload;
    return EXR_SUCCESS;
}

static exr_result jph_parse_nlt(JphReader *r, const JphProfile *jp) {
    uint16_t len, comp;
    uint8_t bd, type;
    uint32_t bit_depth;
    int is_signed;
    uint16_t i;
    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len != 6u) return EXR_ERROR_UNSUPPORTED;
    if ((size_t)(r->end - r->p) < 4) return EXR_ERROR_CORRUPT;
    comp = jph_be16(r->p);
    bd = r->p[2];
    type = r->p[3];
    r->p += 4;
    if (!(type == 0u || type == 3u)) return EXR_ERROR_UNSUPPORTED;
    if (comp != 0xffffu && (!jp->saw_siz || comp >= jp->csiz))
        return EXR_ERROR_CORRUPT;
    if (!jp->saw_siz || !jp->nlt_type) return EXR_ERROR_CORRUPT;
    bit_depth = (uint32_t)(bd & 0x7fu) + 1u;
    is_signed = (bd & 0x80u) != 0;
    if (comp == 0xffffu) {
        for (i = 0; i < jp->csiz; ++i) {
            uint32_t siz_bits = (uint32_t)(jp->ssiz[i] & 0x7fu) + 1u;
            int siz_signed = (jp->ssiz[i] & 0x80u) != 0;
            if (bit_depth != siz_bits || is_signed != siz_signed)
                return EXR_ERROR_UNSUPPORTED;
            if (type == 3u && !is_signed) return EXR_ERROR_UNSUPPORTED;
            jp->nlt_type[i] = type;
        }
    } else {
        uint32_t siz_bits = (uint32_t)(jp->ssiz[comp] & 0x7fu) + 1u;
        int siz_signed = (jp->ssiz[comp] & 0x80u) != 0;
        if (bit_depth != siz_bits || is_signed != siz_signed)
            return EXR_ERROR_UNSUPPORTED;
        if (type == 3u && !is_signed) return EXR_ERROR_UNSUPPORTED;
        jp->nlt_type[comp] = type;
    }
    return EXR_SUCCESS;
}

static exr_result jph_parse_sot(JphReader *r, JphProfile *jp) {
    uint16_t len, isot;
    uint32_t psot;
    uint8_t tpsot, tnsot;
    if (!jph_read_be16(r, &len)) return EXR_ERROR_CORRUPT;
    if (len != 10u) return EXR_ERROR_CORRUPT;
    if (!jph_read_be16(r, &isot) || !jph_read_be32(r, &psot) ||
        !jph_read_u8(r, &tpsot) || !jph_read_u8(r, &tnsot))
        return EXR_ERROR_CORRUPT;
    if (isot != 0u || tpsot != 0u || tnsot > 1u) return EXR_ERROR_UNSUPPORTED;
    if (psot != 0u && psot < 14u) return EXR_ERROR_CORRUPT;
    jp->psot = psot;
    jp->saw_sot = 1;
    return EXR_SUCCESS;
}

static exr_result jph_finish_tile_payload(JphReader *r, JphProfile *jp) {
    const uint8_t *payload_start = r->p;
    const uint8_t *payload_end = NULL;
    JphReader er;
    uint16_t marker;

    if (jp->psot != 0u) {
        const uint8_t *tile_end;
        if (!jp->sot_start) return EXR_ERROR_CORRUPT;
        if ((size_t)(r->end - jp->sot_start) < (size_t)jp->psot)
            return EXR_ERROR_CORRUPT;
        tile_end = jp->sot_start + jp->psot;
        if (tile_end < payload_start) return EXR_ERROR_CORRUPT;
        jp->tile_data = payload_start;
        jp->tile_data_size = (size_t)(tile_end - payload_start);
        r->p = tile_end;
        er = *r;
        if (jph_next_marker(&er, &marker) != EXR_SUCCESS)
            return EXR_ERROR_CORRUPT;
        if (marker != JPH_MARK_EOC) return EXR_ERROR_UNSUPPORTED;
        r->p = er.p;
    } else {
        exr_result rc;
        er = *r;
        rc = jph_find_eoc(&er, &payload_end);
        if (rc != EXR_SUCCESS) return rc;
        if (!payload_end || payload_end < payload_start) return EXR_ERROR_CORRUPT;
        jp->tile_data = payload_start;
        jp->tile_data_size = (size_t)(payload_end - payload_start);
        r->p = er.p;
    }
    if (jp->tile_data_size == 0) return EXR_ERROR_CORRUPT;
    return EXR_SUCCESS;
}

static exr_result jph_bitreader_terminate_packet(exr_jph_bitreader *br) {
    exr_jph_bitreader_align(br);
    if (br->prev_ff) {
        uint8_t byte;
        if (!br->p || br->p >= br->end) return EXR_ERROR_CORRUPT;
        byte = *br->p++;
        if (byte & 0x80u) return EXR_ERROR_CORRUPT;
        br->prev_ff = 0;
    }
    br->bits = 0;
    br->bit_count = 0;
    return EXR_SUCCESS;
}

static exr_result jph_codeblock_list_append(const exr_allocator *a,
                                            JphCodeblockList *list,
                                            JphCodeblockSeg seg) {
    JphCodeblockSeg *new_items;
    size_t new_cap, new_bytes, old_bytes;
    if (!a || !list) return EXR_ERROR_INVALID_ARGUMENT;
    if (list->count == list->cap) {
        if (list->cap > (SIZE_MAX / 2u)) return EXR_ERROR_CORRUPT;
        new_cap = list->cap ? list->cap * 2u : 16u;
        if (exr_mul_ovf(new_cap, sizeof(*new_items), &new_bytes))
            return EXR_ERROR_CORRUPT;
        new_items = (JphCodeblockSeg *)exr_malloc(a, new_bytes);
        if (!new_items) return EXR_ERROR_OUT_OF_MEMORY;
        if (list->items) {
            if (exr_mul_ovf(list->count, sizeof(*list->items), &old_bytes)) {
                exr_free(a, new_items);
                return EXR_ERROR_CORRUPT;
            }
            memcpy(new_items, list->items, old_bytes);
            exr_free(a, list->items);
        }
        list->items = new_items;
        list->cap = new_cap;
    }
    list->items[list->count++] = seg;
    return EXR_SUCCESS;
}

static exr_result jph_validate_forward_stuffing(const uint8_t *data,
                                                size_t size,
                                                uint8_t max_after_ff) {
    size_t i;
    int prev_ff = 0;
    if (!data && size) return EXR_ERROR_INVALID_ARGUMENT;
    for (i = 0; i < size; ++i) {
        if (prev_ff && data[i] > max_after_ff) return EXR_ERROR_CORRUPT;
        prev_ff = (data[i] == 0xffu);
    }
    return EXR_SUCCESS;
}

static exr_result jph_validate_reverse_stuffing(const uint8_t *data,
                                                size_t size,
                                                int initial_unstuff) {
    size_t i;
    int unstuff = initial_unstuff;
    if (!data && size) return EXR_ERROR_INVALID_ARGUMENT;
    for (i = size; i > 0; --i) {
        uint8_t byte = data[i - 1u];
        if (unstuff && ((byte & 0x7fu) == 0x7fu) && (byte & 0x80u))
            return EXR_ERROR_CORRUPT;
        unstuff = byte > 0x8fu;
    }
    return EXR_SUCCESS;
}

static exr_result jph_split_ht_codeblock_streams(const uint8_t *data,
                                                 size_t data_size,
                                                 uint32_t length0,
                                                 uint32_t length1,
                                                 JphCodeblockStreams *out) {
    size_t total, scup_start;
    uint32_t scup;
    exr_result rc;
    if (!out || (!data && data_size)) return EXR_ERROR_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    if (length0 < 2u || length0 >= 65535u || length1 >= 2047u)
        return EXR_ERROR_CORRUPT;
    if (exr_add_ovf((size_t)length0, (size_t)length1, &total))
        return EXR_ERROR_CORRUPT;
    if (total != data_size) return EXR_ERROR_CORRUPT;

    scup = ((uint32_t)data[length0 - 1u] << 4u) |
           ((uint32_t)data[length0 - 2u] & 0x0fu);
    if (scup < 2u || scup > length0 || scup > 4079u)
        return EXR_ERROR_CORRUPT;

    scup_start = (size_t)length0 - (size_t)scup;
    out->magsgn = data;
    out->magsgn_size = scup_start;
    out->mel = data + scup_start;
    out->mel_size = (size_t)scup - 1u;
    out->vlc = data + scup_start + 1u;
    out->vlc_size = (size_t)scup - 2u;
    out->refine = data + length0;
    out->refine_size = length1;
    out->scup = scup;

    rc = jph_validate_forward_stuffing(out->magsgn, out->magsgn_size, 0x7fu);
    if (rc != EXR_SUCCESS) return rc;
    rc = jph_validate_forward_stuffing(out->mel, out->mel_size, 0x8fu);
    if (rc != EXR_SUCCESS) return rc;
    rc = jph_validate_reverse_stuffing(out->vlc, out->vlc_size,
                                       (data[length0 - 1u] | 0x0fu) > 0x8fu);
    if (rc != EXR_SUCCESS) return rc;
    if (out->refine_size) {
        rc = jph_validate_forward_stuffing(out->refine, out->refine_size,
                                           0x7fu);
        if (rc != EXR_SUCCESS) return rc;
        rc = jph_validate_reverse_stuffing(out->refine, out->refine_size, 1);
        if (rc != EXR_SUCCESS) return rc;
    }
    return EXR_SUCCESS;
}

static exr_result jph_validate_ht_codeblock_segments(const uint8_t *data,
                                                     size_t data_size,
                                                     uint32_t missing_msbs,
                                                     uint32_t active_passes,
                                                     uint32_t length0,
                                                     uint32_t length1,
                                                     uint32_t width,
                                                     uint32_t height) {
    JphCodeblockStreams streams;
    if (!data && data_size) return EXR_ERROR_INVALID_ARGUMENT;
    if (width == 0u || height == 0u || width > 128u || height > 32u)
        return EXR_ERROR_CORRUPT;
    if (active_passes == 0u || active_passes > 3u)
        return EXR_ERROR_CORRUPT;
    {
        exr_result rc = jph_split_ht_codeblock_streams(data, data_size,
                                                       length0, length1,
                                                       &streams);
        if (rc != EXR_SUCCESS) return rc;
    }

    if (active_passes > 1u && length1 == 0u) active_passes = 1u;
    if (missing_msbs > 30u) return EXR_SUCCESS;
    if (missing_msbs == 30u && active_passes > 1u) return EXR_ERROR_CORRUPT;
    if (missing_msbs == 29u && active_passes > 1u) active_passes = 1u;
    (void)streams;
    return EXR_SUCCESS;
}

static uint32_t jph_band_quant_index(uint32_t res, uint32_t band) {
    return res ? (res - 1u) * 3u + band : 0u;
}

static exr_result jph_band_kmax(const JphProfile *jp, uint32_t res,
                                uint32_t band, uint32_t *out_kmax) {
    uint32_t idx, expn, bits;
    if (!jp || !out_kmax || jp->qcd_count == 0) return EXR_ERROR_CORRUPT;
    idx = jph_band_quant_index(res, band);
    if (idx >= jp->qcd_count) idx = (uint32_t)jp->qcd_count - 1u;
    expn = (uint32_t)(jp->qcd_exp[idx] >> 3);
    bits = expn ? expn - 1u : 0u;
    *out_kmax = bits + (uint32_t)jp->qcd_guard_bits;
    return EXR_SUCCESS;
}

static JphSize jph_component_size(const JphProfile *jp, uint32_t comp) {
    JphSize s;
    s.w = jph_divceil_u32(jp->xsiz, jp->xrsiz[comp]) -
          jph_divceil_u32(jp->xosiz, jp->xrsiz[comp]);
    s.h = jph_divceil_u32(jp->ysiz, jp->yrsiz[comp]) -
          jph_divceil_u32(jp->yosiz, jp->yrsiz[comp]);
    return s;
}

static JphSize jph_resolution_size(JphSize comp_size, uint32_t num_decomps,
                                   uint32_t res) {
    uint32_t shift = num_decomps - res;
    JphSize s;
    s.w = jph_ceil_div_pow2_u32(comp_size.w, shift);
    s.h = jph_ceil_div_pow2_u32(comp_size.h, shift);
    return s;
}

static exr_result jph_build_band_geoms(const JphProfile *jp, uint32_t comp,
                                       uint32_t res, JphBandGeom bands[4],
                                       JphSize *out_res_size) {
    JphSize comp_size = jph_component_size(jp, comp);
    JphSize rs = jph_resolution_size(comp_size, jp->num_decomps, res);
    uint32_t b;
    memset(bands, 0, sizeof(JphBandGeom) * 4u);
    if (out_res_size) *out_res_size = rs;

    if (res == 0u) {
        bands[0].w = rs.w;
        bands[0].h = rs.h;
        bands[0].cb_w = 128u;
        bands[0].cb_h = 32u;
        bands[0].exists = (rs.w != 0u && rs.h != 0u);
        return jph_band_kmax(jp, 0, 0, &bands[0].kmax);
    }

    bands[1].w = rs.w >> 1u;
    bands[1].h = (rs.h + 1u) >> 1u;
    bands[2].w = (rs.w + 1u) >> 1u;
    bands[2].h = rs.h >> 1u;
    bands[3].w = rs.w >> 1u;
    bands[3].h = rs.h >> 1u;
    for (b = 1; b < 4; ++b) {
        bands[b].cb_w = 128u;
        bands[b].cb_h = 32u;
        bands[b].exists = (bands[b].w != 0u && bands[b].h != 0u);
        if (bands[b].exists) {
            exr_result rc = jph_band_kmax(jp, res, b, &bands[b].kmax);
            if (rc != EXR_SUCCESS) return rc;
        }
    }
    return EXR_SUCCESS;
}

static uint32_t jph_subband_project(uint32_t v, uint32_t offset) {
    if (v <= offset) return 0;
    return (v - offset + 1u) >> 1u;
}

static exr_result jph_parse_precinct_packet(const exr_allocator *a,
                                            const JphProfile *jp,
                                            uint32_t comp, uint32_t res,
                                            uint32_t precinct_x,
                                            uint32_t precinct_y,
                                            exr_jph_bitreader *br,
                                            size_t *out_codeblocks,
                                            JphCodeblockCallback cb,
                                            void *cb_user) {
    enum { JPH_PREC_LOG = 15 };
    JphBandGeom bands[4];
    JphSize rs;
    JphCodeblockList codeblocks;
    int saw_nonempty_band = 0;
    uint32_t b;
    exr_result rc;

    if (!a) a = exr_default_allocator();
    if (out_codeblocks) *out_codeblocks = 0;
    memset(&codeblocks, 0, sizeof(codeblocks));
    rc = jph_build_band_geoms(jp, comp, res, bands, &rs);
    if (rc != EXR_SUCCESS) return rc;

    for (b = 0; b < 4; ++b) {
        uint32_t p0x, p1x, p0y, p1y, bx0, bx1, by0, by1;
        uint32_t cbx0, cbx1, cby0, cby1, cbw, cbh, x, y;
        exr_jph_tag_tree inc_tree, mmsb_tree;
        int trees_ready = 0;

        if (!bands[b].exists) continue;
        p0x = precinct_x << JPH_PREC_LOG;
        p0y = precinct_y << JPH_PREC_LOG;
        p1x = p0x + (1u << JPH_PREC_LOG);
        p1y = p0y + (1u << JPH_PREC_LOG);
        if (p0x > rs.w) p0x = rs.w;
        if (p0y > rs.h) p0y = rs.h;
        if (p1x > rs.w) p1x = rs.w;
        if (p1y > rs.h) p1y = rs.h;

        if (res == 0u) {
            bx0 = p0x;
            bx1 = p1x;
            by0 = p0y;
            by1 = p1y;
        } else {
            bx0 = jph_subband_project(p0x, b & 1u);
            bx1 = jph_subband_project(p1x, b & 1u);
            by0 = jph_subband_project(p0y, b >> 1u);
            by1 = jph_subband_project(p1y, b >> 1u);
        }
        if (bx1 > bands[b].w) bx1 = bands[b].w;
        if (by1 > bands[b].h) by1 = bands[b].h;
        if (bx1 <= bx0 || by1 <= by0) continue;

        cbx0 = bx0 / bands[b].cb_w;
        cbx1 = jph_divceil_u32(bx1, bands[b].cb_w);
        cby0 = by0 / bands[b].cb_h;
        cby1 = jph_divceil_u32(by1, bands[b].cb_h);
        cbw = cbx1 - cbx0;
        cbh = cby1 - cby0;
        if (cbw == 0u || cbh == 0u) continue;

        if (!saw_nonempty_band) {
            uint32_t packet_present = 0;
            rc = exr_jph_bitreader_read(br, 1, &packet_present);
            if (rc != EXR_SUCCESS) return rc;
            if (!packet_present) return jph_bitreader_terminate_packet(br);
            saw_nonempty_band = 1;
        }

        memset(&inc_tree, 0, sizeof(inc_tree));
        memset(&mmsb_tree, 0, sizeof(mmsb_tree));
        rc = exr_jph_tag_tree_init(a, &inc_tree, cbw, cbh);
        if (rc != EXR_SUCCESS) goto band_done;
        rc = exr_jph_tag_tree_init(a, &mmsb_tree, cbw, cbh);
        if (rc != EXR_SUCCESS) goto band_done;
        trees_ready = 1;

        for (y = 0; y < cbh; ++y) {
            for (x = 0; x < cbw; ++x) {
                uint32_t inc = 0, missing = 0, raw = 0, active = 0, groups = 0;
                uint32_t lengths[2] = {0, 0};
                uint32_t cb_global_x, cb_global_y;
                uint64_t cb_px0, cb_px1, cb_py0, cb_py1;
                JphCodeblockSeg seg;

                rc = exr_jph_tag_tree_decode(&inc_tree, br, x, y, 1, &inc);
                if (rc != EXR_SUCCESS) goto band_done;
                if (inc >= 1u) continue;

                rc = exr_jph_tag_tree_decode(&mmsb_tree, br, x, y,
                                             bands[b].kmax + 1u, &missing);
                if (rc != EXR_SUCCESS) goto band_done;
                if (missing > bands[b].kmax) {
                    rc = EXR_ERROR_CORRUPT;
                    goto band_done;
                }
                rc = exr_jph_packet_read_pass_count(br, &raw, &active, &groups);
                (void)raw;
                if (rc != EXR_SUCCESS) goto band_done;
                if (groups > bands[b].kmax - missing) {
                    rc = EXR_ERROR_CORRUPT;
                    goto band_done;
                }
                rc = exr_jph_packet_read_pass_lengths(br, active, groups,
                                                      lengths);
                if (rc != EXR_SUCCESS) goto band_done;

                cb_global_x = cbx0 + x;
                cb_global_y = cby0 + y;
                cb_px0 = (uint64_t)cb_global_x * bands[b].cb_w;
                cb_py0 = (uint64_t)cb_global_y * bands[b].cb_h;
                cb_px1 = cb_px0 + bands[b].cb_w;
                cb_py1 = cb_py0 + bands[b].cb_h;
                if (cb_px0 >= bands[b].w || cb_py0 >= bands[b].h) {
                    rc = EXR_ERROR_CORRUPT;
                    goto band_done;
                }
                if (cb_px1 > bands[b].w) cb_px1 = bands[b].w;
                if (cb_py1 > bands[b].h) cb_py1 = bands[b].h;
                if (cb_px1 <= cb_px0 || cb_py1 <= cb_py0 ||
                    cb_px1 - cb_px0 > UINT32_MAX ||
                    cb_py1 - cb_py0 > UINT32_MAX) {
                    rc = EXR_ERROR_CORRUPT;
                    goto band_done;
                }
                seg.missing_msbs = missing;
                seg.active_passes = active;
                seg.length0 = lengths[0];
                seg.length1 = lengths[1];
                seg.comp = comp;
                seg.res = res;
                seg.band = b;
                seg.x0 = (uint32_t)cb_px0;
                seg.y0 = (uint32_t)cb_py0;
                seg.width = (uint32_t)(cb_px1 - cb_px0);
                seg.height = (uint32_t)(cb_py1 - cb_py0);
                seg.data = NULL;
                seg.data_size = 0;
                rc = jph_codeblock_list_append(a, &codeblocks, seg);
                if (rc != EXR_SUCCESS) goto band_done;
            }
        }

band_done:
        if (trees_ready) {
            exr_jph_tag_tree_free(a, &inc_tree);
            exr_jph_tag_tree_free(a, &mmsb_tree);
        } else {
            exr_jph_tag_tree_free(a, &inc_tree);
        }
        if (rc != EXR_SUCCESS) goto done;
    }

    if (!saw_nonempty_band) {
        uint32_t bit = 0;
        rc = exr_jph_bitreader_read(br, 1, &bit);
        if (rc != EXR_SUCCESS) goto done;
    }
    rc = jph_bitreader_terminate_packet(br);
    if (rc != EXR_SUCCESS) goto done;
    for (b = 0; b < codeblocks.count; ++b) {
        size_t total;
        JphCodeblockSeg *seg = &codeblocks.items[b];
        if (exr_add_ovf((size_t)seg->length0, (size_t)seg->length1, &total)) {
            rc = EXR_ERROR_CORRUPT;
            goto done;
        }
        if ((size_t)(br->end - br->p) < total) {
            rc = EXR_ERROR_CORRUPT;
            goto done;
        }
        seg->data = br->p;
        seg->data_size = total;
        rc = jph_validate_ht_codeblock_segments(br->p, total,
                                                seg->missing_msbs,
                                                seg->active_passes,
                                                seg->length0, seg->length1,
                                                seg->width, seg->height);
        if (rc != EXR_SUCCESS) goto done;
        if (cb) {
            rc = cb(cb_user, seg);
            if (rc != EXR_SUCCESS) goto done;
        }
        br->p += total;
    }
    rc = EXR_SUCCESS;
    if (out_codeblocks) *out_codeblocks = codeblocks.count;

done:
    exr_free(a, codeblocks.items);
    return rc;
}

static exr_result jph_init_precinct_states(const JphProfile *jp,
                                           JphPrecinctState *states) {
    enum { JPH_PREC_LOG = 15 };
    uint32_t c, r;
    if (!jp || !states) return EXR_ERROR_INVALID_ARGUMENT;
    for (c = 0; c < jp->csiz; ++c) {
        JphSize comp_size = jph_component_size(jp, c);
        for (r = 0; r <= jp->num_decomps; ++r) {
            JphSize rs = jph_resolution_size(comp_size, jp->num_decomps, r);
            JphPrecinctState *st = states + (size_t)c * (jp->num_decomps + 1u) + r;
            st->num_w = jph_divceil_u32(rs.w, 1u << JPH_PREC_LOG);
            st->num_h = jph_divceil_u32(rs.h, 1u << JPH_PREC_LOG);
            st->cur_x = 0;
            st->cur_y = 0;
        }
    }
    return EXR_SUCCESS;
}

static int jph_precinct_top_left(const JphProfile *jp, uint32_t comp,
                                 uint32_t res, const JphPrecinctState *st,
                                 uint64_t *out_x, uint64_t *out_y) {
    enum { JPH_PREC_LOG = 15 };
    uint64_t down_x, down_y;
    if (!st || st->cur_y >= st->num_h || st->cur_x >= st->num_w) return 0;
    down_x = (uint64_t)jp->xrsiz[comp] << (jp->num_decomps - res);
    down_y = (uint64_t)jp->yrsiz[comp] << (jp->num_decomps - res);
    *out_x = down_x * ((uint64_t)st->cur_x << JPH_PREC_LOG);
    *out_y = down_y * ((uint64_t)st->cur_y << JPH_PREC_LOG);
    return 1;
}

static void jph_advance_precinct(JphPrecinctState *st) {
    if (++st->cur_x >= st->num_w) {
        st->cur_x = 0;
        st->cur_y++;
    }
}

typedef struct {
    const exr_allocator *a;
    const JphProfile *jp;
    JphComponentPlane *planes;
    uint16_t num_planes;
    size_t codeblocks;
} JphDecodeState;

static void jph_free_component_planes(const exr_allocator *a,
                                      JphComponentPlane *planes,
                                      uint16_t num_planes) {
    uint16_t c;
    if (!a) a = exr_default_allocator();
    if (!planes) return;
    for (c = 0; c < num_planes; ++c) exr_free(a, planes[c].data);
    exr_free(a, planes);
}

static exr_result jph_alloc_component_planes(const exr_allocator *a,
                                             const JphProfile *jp,
                                             JphComponentPlane **out_planes) {
    JphComponentPlane *planes;
    uint16_t c;
    if (!out_planes || !jp || jp->csiz == 0) return EXR_ERROR_INVALID_ARGUMENT;
    if (!a) a = exr_default_allocator();
    *out_planes = NULL;
    planes = (JphComponentPlane *)exr_calloc(a, jp->csiz, sizeof(*planes));
    if (!planes) return EXR_ERROR_OUT_OF_MEMORY;
    for (c = 0; c < jp->csiz; ++c) {
        JphSize s = jph_component_size(jp, c);
        size_t count;
        planes[c].w = s.w;
        planes[c].h = s.h;
        if (exr_mul_ovf((size_t)s.w, (size_t)s.h, &count) ||
            exr_mul_ovf(count, sizeof(int32_t), &count)) {
            jph_free_component_planes(a, planes, jp->csiz);
            return EXR_ERROR_CORRUPT;
        }
        planes[c].data = (int32_t *)exr_calloc(a, count ? count : 1u, 1);
        if (!planes[c].data) {
            jph_free_component_planes(a, planes, jp->csiz);
            return EXR_ERROR_OUT_OF_MEMORY;
        }
    }
    *out_planes = planes;
    return EXR_SUCCESS;
}

static exr_result jph_find_component_for_channel(const JphProfile *jp,
                                                 const uint16_t *map,
                                                 uint16_t channel,
                                                 uint16_t *out_comp) {
    uint16_t c;
    if (!jp || !map || !out_comp) return EXR_ERROR_INVALID_ARGUMENT;
    for (c = 0; c < jp->csiz; ++c) {
        if (map[c] == channel) {
            *out_comp = c;
            return EXR_SUCCESS;
        }
    }
    return EXR_ERROR_CORRUPT;
}

static exr_result jph_store_sample(uint8_t *dst, size_t dst_size, size_t *off,
                                   exr_pixel_type pixel_type, int32_t v) {
    if (!dst || !off) return EXR_ERROR_INVALID_ARGUMENT;
    if (pixel_type == EXR_PIXEL_HALF) {
        uint16_t bits;
        if (*off > dst_size || dst_size - *off < 2u) return EXR_ERROR_CORRUPT;
        bits = (uint16_t)v;
        dst[*off] = (uint8_t)bits;
        dst[*off + 1u] = (uint8_t)(bits >> 8);
        *off += 2u;
    } else {
        uint32_t bits;
        if (*off > dst_size || dst_size - *off < 4u) return EXR_ERROR_CORRUPT;
        if (pixel_type == EXR_PIXEL_UINT && v < 0) return EXR_ERROR_CORRUPT;
        bits = (uint32_t)v;
        exr_wr_u32(dst + *off, bits);
        *off += 4u;
    }
    return EXR_SUCCESS;
}

static exr_result jph_store_component_planes_to_block(
    const exr_codec_ctx *ctx, const JphProfile *jp, const uint16_t *map,
    const JphComponentPlane *planes, uint8_t *dst, size_t dst_size) {
    size_t off = 0;
    int32_t line, file_c;
    if (!ctx || !jp || !map || !planes || !dst) return EXR_ERROR_INVALID_ARGUMENT;
    for (line = 0; line < ctx->num_lines; ++line) {
        int32_t yy = ctx->y + line;
        for (file_c = 0; file_c < ctx->num_channels; ++file_c) {
            const exr_channel *ch = &ctx->channels[file_c];
            int32_t xs = ch->x_sampling <= 0 ? 1 : ch->x_sampling;
            int32_t ys = ch->y_sampling <= 0 ? 1 : ch->y_sampling;
            uint16_t comp = 0;
            int nx, x;
            int row_i;
            uint32_t row;
            exr_result rc;
            if ((yy % ys) != 0) continue;
            rc = jph_find_component_for_channel(jp, map, (uint16_t)file_c,
                                                &comp);
            if (rc != EXR_SUCCESS) return rc;
            nx = exr_num_samples(ctx->x, ctx->x + ctx->width - 1, xs);
            if (nx < 0) nx = 0;
            row_i = exr_num_samples(ctx->y, yy, ys) - 1;
            if (row_i < 0) return EXR_ERROR_CORRUPT;
            row = (uint32_t)row_i;
            if (row >= planes[comp].h || (uint32_t)nx > planes[comp].w)
                return EXR_ERROR_CORRUPT;
            for (x = 0; x < nx; ++x) {
                size_t idx = (size_t)row * planes[comp].w + (uint32_t)x;
                rc = jph_store_sample(dst, dst_size, &off, ch->pixel_type,
                                      planes[comp].data[idx]);
                if (rc != EXR_SUCCESS) return rc;
            }
        }
    }
    return off == dst_size ? EXR_SUCCESS : EXR_ERROR_CORRUPT;
}

static exr_result jph_postprocess_component_planes(const exr_allocator *a,
                                                   const JphProfile *jp,
                                                   JphComponentPlane *planes) {
    uint16_t c;
    if (!jp || !planes) return EXR_ERROR_INVALID_ARGUMENT;
    for (c = 0; c < jp->csiz; ++c) {
        exr_result rc;
        if (!planes[c].data) return EXR_ERROR_CORRUPT;
        rc = exr_jph_inverse_53_2d_i32(a, planes[c].data, planes[c].w,
                                       planes[c].h, jp->num_decomps);
        if (rc != EXR_SUCCESS) return rc;
    }
    if (jp->mc_trans) {
        size_t count;
        if (jp->csiz < 3u) return EXR_ERROR_CORRUPT;
        if (planes[0].w != planes[1].w || planes[0].w != planes[2].w ||
            planes[0].h != planes[1].h || planes[0].h != planes[2].h)
            return EXR_ERROR_UNSUPPORTED;
        if (exr_mul_ovf((size_t)planes[0].w, (size_t)planes[0].h, &count))
            return EXR_ERROR_CORRUPT;
        {
            exr_result rc = exr_jph_inverse_rct_i32(planes[0].data,
                                                    planes[1].data,
                                                    planes[2].data, count);
            if (rc != EXR_SUCCESS) return rc;
        }
    }
    for (c = 0; c < jp->csiz; ++c) {
        uint32_t bit_depth = (uint32_t)(jp->ssiz[c] & 0x7fu) + 1u;
        if (jp->nlt_type && jp->nlt_type[c] == 3u) {
            size_t count;
            exr_result rc;
            if (exr_mul_ovf((size_t)planes[c].w, (size_t)planes[c].h, &count))
                return EXR_ERROR_CORRUPT;
            rc = exr_jph_apply_nlt_type3_i32(planes[c].data, count,
                                             bit_depth);
            if (rc != EXR_SUCCESS) return rc;
        } else if (jp->nlt_type && jp->nlt_type[c] != 0u) {
            return EXR_ERROR_UNSUPPORTED;
        }
    }
    return EXR_SUCCESS;
}

static exr_result jph_decode_codeblock_stub(void *user,
                                            const JphCodeblockSeg *seg) {
    JphDecodeState *st = (JphDecodeState *)user;
    JphBandGeom bands[4];
    exr_result rc;
    if (!st || !seg || !seg->data || seg->data_size == 0)
        return EXR_ERROR_CORRUPT;
    if (!st->jp || seg->comp >= st->num_planes || !st->planes ||
        !st->planes[seg->comp].data)
        return EXR_ERROR_CORRUPT;
    rc = jph_build_band_geoms(st->jp, seg->comp, seg->res, bands, NULL);
    if (rc != EXR_SUCCESS) return rc;
    if (seg->band >= 4u || !bands[seg->band].exists)
        return EXR_ERROR_CORRUPT;
    if (seg->x0 > bands[seg->band].w || seg->y0 > bands[seg->band].h ||
        seg->width > bands[seg->band].w - seg->x0 ||
        seg->height > bands[seg->band].h - seg->y0)
        return EXR_ERROR_CORRUPT;
    if (exr_add_ovf(st->codeblocks, 1u, &st->codeblocks))
        return EXR_ERROR_CORRUPT;
    return EXR_SUCCESS;
}

static exr_result jph_parse_tile_packets(const exr_codec_ctx *ctx,
                                         const JphProfile *jp,
                                         size_t *out_codeblocks,
                                         JphCodeblockCallback cb,
                                         void *cb_user) {
    const exr_allocator *a = ctx->alloc ? ctx->alloc : exr_default_allocator();
    exr_jph_bitreader br;
    JphPrecinctState *states = NULL;
    size_t state_count, total_codeblocks = 0;
    uint32_t r;
    exr_result rc;

    if (out_codeblocks) *out_codeblocks = 0;
    if (jp->saw_qcc) return EXR_ERROR_UNSUPPORTED;
    if (jp->num_decomps > 31u) return EXR_ERROR_CORRUPT;
    if (exr_mul_ovf((size_t)jp->csiz, (size_t)jp->num_decomps + 1u,
                    &state_count))
        return EXR_ERROR_CORRUPT;
    states = (JphPrecinctState *)exr_calloc(a, state_count, sizeof(*states));
    if (!states) return EXR_ERROR_OUT_OF_MEMORY;
    rc = jph_init_precinct_states(jp, states);
    if (rc != EXR_SUCCESS) goto done;

    exr_jph_bitreader_init(&br, jp->tile_data, jp->tile_data_size);
    for (r = 0; r <= jp->num_decomps; ++r) {
        while (1) {
            uint32_t c, best_c = 0;
            uint64_t best_x = UINT64_MAX, best_y = UINT64_MAX;
            int found = 0;
            for (c = 0; c < jp->csiz; ++c) {
                JphPrecinctState *st =
                    states + (size_t)c * (jp->num_decomps + 1u) + r;
                uint64_t x = 0, y = 0;
                if (!jph_precinct_top_left(jp, c, r, st, &x, &y)) continue;
                if (!found || y < best_y || (y == best_y && x < best_x)) {
                    found = 1;
                    best_c = c;
                    best_x = x;
                    best_y = y;
                }
            }
            if (!found) break;
            if (br.p >= br.end) {
                rc = EXR_ERROR_CORRUPT;
                goto done;
            }
            {
                JphPrecinctState *st =
                    states + (size_t)best_c * (jp->num_decomps + 1u) + r;
                size_t packet_codeblocks = 0;
                rc = jph_parse_precinct_packet(a, jp, best_c, r, st->cur_x,
                                               st->cur_y, &br,
                                               &packet_codeblocks, cb,
                                               cb_user);
                if (rc != EXR_SUCCESS) goto done;
                if (exr_add_ovf(total_codeblocks, packet_codeblocks,
                                &total_codeblocks)) {
                    rc = EXR_ERROR_CORRUPT;
                    goto done;
                }
                jph_advance_precinct(st);
            }
        }
    }
    if (br.p != br.end) rc = EXR_ERROR_CORRUPT;
    else if (out_codeblocks) *out_codeblocks = total_codeblocks;

done:
    exr_free(a, states);
    return rc;
}

static exr_result jph_decode_tile_payload(const exr_codec_ctx *ctx,
                                          const JphProfile *jp,
                                          const uint16_t *map,
                                          uint8_t *dst, size_t dst_size) {
    size_t expected = 0;
    size_t codeblocks = 0;
    JphDecodeState decode_state;
    exr_result rc;

    (void)map;
    memset(&decode_state, 0, sizeof(decode_state));
    if (!jp->tile_data || jp->tile_data_size == 0) return EXR_ERROR_CORRUPT;
    rc = exr_block_uncompressed_size(ctx->channels, ctx->num_channels, ctx->x,
                                     ctx->y, ctx->width, ctx->num_lines,
                                     &expected);
    if (rc != EXR_SUCCESS) return rc;
    if (expected != dst_size) return EXR_ERROR_CORRUPT;
    if (jp->mc_trans != 0u && ctx->num_channels < 3) return EXR_ERROR_CORRUPT;
    decode_state.a = ctx->alloc ? ctx->alloc : exr_default_allocator();
    decode_state.jp = jp;
    decode_state.num_planes = jp->csiz;
    rc = jph_alloc_component_planes(decode_state.a, jp, &decode_state.planes);
    if (rc != EXR_SUCCESS) return rc;
    rc = jph_parse_tile_packets(ctx, jp, &codeblocks, jph_decode_codeblock_stub,
                                &decode_state);
    if (rc != EXR_SUCCESS) goto done;
    if (decode_state.codeblocks != codeblocks) {
        rc = EXR_ERROR_CORRUPT;
        goto done;
    }
    if (codeblocks == 0) {
        rc = jph_postprocess_component_planes(decode_state.a, jp,
                                              decode_state.planes);
        if (rc != EXR_SUCCESS) goto done;
        rc = jph_store_component_planes_to_block(ctx, jp, map,
                                                 decode_state.planes, dst,
                                                 dst_size);
        goto done;
    }
    rc = EXR_ERROR_UNSUPPORTED;

done:
    jph_free_component_planes(decode_state.a, decode_state.planes,
                              decode_state.num_planes);
    return rc;
}

static exr_result jph_validate_profile(const exr_codec_ctx *ctx,
                                       const uint16_t *map,
                                       const uint8_t *src, size_t src_size,
                                       uint8_t *dst, size_t dst_size) {
    const exr_allocator *a = ctx->alloc ? ctx->alloc : exr_default_allocator();
    JphReader r;
    JphProfile jp;
    uint16_t marker;
    exr_result rc = EXR_SUCCESS;
    int done = 0;
    uint16_t c;

    memset(&jp, 0, sizeof(jp));
    r.p = src;
    r.end = src + src_size;

    rc = jph_next_marker(&r, &marker);
    if (rc != EXR_SUCCESS) goto done;
    if (marker != JPH_MARK_SOC) { rc = EXR_ERROR_CORRUPT; goto done; }
    rc = jph_next_marker(&r, &marker);
    if (rc != EXR_SUCCESS) goto done;
    if (marker != JPH_MARK_SIZ) { rc = EXR_ERROR_CORRUPT; goto done; }
    rc = jph_parse_siz(a, &r, &jp);
    if (rc != EXR_SUCCESS) goto done;

    while (!done) {
        rc = jph_next_marker(&r, &marker);
        if (rc != EXR_SUCCESS) goto done;
        switch (marker) {
        case JPH_MARK_CAP:
            rc = jph_parse_cap(&r, &jp);
            break;
        case JPH_MARK_PRF:
        case JPH_MARK_CPF:
        case JPH_MARK_TLM:
        case JPH_MARK_PLM:
        case JPH_MARK_PLT:
        case JPH_MARK_CRG:
        case JPH_MARK_COM:
            rc = jph_skip_segment(&r);
            break;
        case JPH_MARK_COD:
            rc = jph_parse_cod(&r, &jp);
            break;
        case JPH_MARK_QCD:
            rc = jph_parse_quant(&r, &jp, 0);
            if (rc == EXR_SUCCESS) jp.saw_qcd = 1;
            break;
        case JPH_MARK_QCC:
            if (!jp.saw_siz) rc = EXR_ERROR_CORRUPT;
            else rc = jph_parse_quant(&r, &jp, 1);
            break;
        case JPH_MARK_NLT:
            rc = jph_parse_nlt(&r, &jp);
            break;
        case JPH_MARK_SOT:
            jp.sot_start = r.p - 2;
            rc = jph_parse_sot(&r, &jp);
            if (rc != EXR_SUCCESS) break;
            while (1) {
                rc = jph_next_marker(&r, &marker);
                if (rc != EXR_SUCCESS) break;
                if (marker == JPH_MARK_SOD) {
                    jp.saw_sod = 1;
                    rc = jph_finish_tile_payload(&r, &jp);
                    if (rc != EXR_SUCCESS) break;
                    done = 1;
                    break;
                }
                if (marker == JPH_MARK_PLT || marker == JPH_MARK_COM) {
                    rc = jph_skip_segment(&r);
                    if (rc != EXR_SUCCESS) break;
                    continue;
                }
                rc = EXR_ERROR_UNSUPPORTED;
                break;
            }
            break;
        case JPH_MARK_COC:
        case JPH_MARK_RGN:
        case JPH_MARK_POC:
        case JPH_MARK_PPM:
        case JPH_MARK_PPT:
        case JPH_MARK_DFS:
        case JPH_MARK_ADS:
        case JPH_MARK_ATK:
            rc = EXR_ERROR_UNSUPPORTED;
            break;
        default:
            rc = EXR_ERROR_UNSUPPORTED;
            break;
        }
        if (rc != EXR_SUCCESS) goto done;
    }

    if (!jp.saw_cod || !jp.saw_qcd || !jp.saw_sot || !jp.saw_sod) {
        rc = EXR_ERROR_CORRUPT;
        goto done;
    }
    if (jp.qcd_count != (uint8_t)(1u + 3u * jp.num_decomps)) {
        rc = EXR_ERROR_CORRUPT;
        goto done;
    }
    if (jp.csiz != (uint16_t)ctx->num_channels) { rc = EXR_ERROR_CORRUPT; goto done; }
    if ((jp.rsiz & 0x4000u) == 0) { rc = EXR_ERROR_UNSUPPORTED; goto done; }
    if (jp.xosiz != 0u || jp.yosiz != 0u || jp.xtosiz != 0u || jp.ytosiz != 0u)
    { rc = EXR_ERROR_UNSUPPORTED; goto done; }
    if (jp.xsiz != (uint32_t)ctx->width || jp.ysiz != (uint32_t)ctx->num_lines)
    { rc = EXR_ERROR_CORRUPT; goto done; }
    if (jp.xtsiz != jp.xsiz || jp.ytsiz != jp.ysiz)
    { rc = EXR_ERROR_UNSUPPORTED; goto done; }
    for (c = 0; c < jp.csiz; ++c) {
        rc = jph_validate_siz_component(ctx, &jp, map, c);
        if (rc != EXR_SUCCESS) goto done;
    }
    rc = jph_decode_tile_payload(ctx, &jp, map, dst, dst_size);

done:
    exr_free(a, jp.ssiz);
    exr_free(a, jp.xrsiz);
    exr_free(a, jp.yrsiz);
    exr_free(a, jp.nlt_type);
    return rc;
}

static exr_result jph_parse_ht_header(const exr_codec_ctx *ctx,
                                      const uint8_t *src, size_t src_size,
                                      uint16_t **out_map,
                                      size_t *out_codestream_off) {
    const exr_allocator *a = ctx->alloc ? ctx->alloc : exr_default_allocator();
    size_t payload_size, prefix_size = 6, header_size, need_map_bytes;
    uint16_t nch, i;
    uint8_t *seen = NULL;
    uint16_t *map = NULL;

    *out_map = NULL;
    *out_codestream_off = 0;
    if (ctx->num_channels <= 0 || ctx->num_channels > EXR_MAX_CHANNELS)
        return EXR_ERROR_CORRUPT;
    if (src_size < 8) return EXR_ERROR_CORRUPT;
    if (jph_be16(src) != 0x4854u) return EXR_ERROR_CORRUPT;
    payload_size = (size_t)jph_be32(src + 2);
    if (exr_add_ovf(prefix_size, payload_size, &header_size))
        return EXR_ERROR_CORRUPT;
    if (header_size > src_size) return EXR_ERROR_CORRUPT;
    if (payload_size < 2) return EXR_ERROR_CORRUPT;
    nch = jph_be16(src + 6);
    if (nch != (uint16_t)ctx->num_channels) return EXR_ERROR_CORRUPT;
    if (exr_mul_ovf((size_t)nch, 2u, &need_map_bytes))
        return EXR_ERROR_CORRUPT;
    if (payload_size < 2u + need_map_bytes) return EXR_ERROR_CORRUPT;

    map = (uint16_t *)exr_malloc(a, (size_t)nch * sizeof(*map));
    seen = (uint8_t *)exr_calloc(a, (size_t)nch, 1);
    if (!map || !seen) {
        exr_free(a, map);
        exr_free(a, seen);
        return EXR_ERROR_OUT_OF_MEMORY;
    }
    for (i = 0; i < nch; ++i) {
        uint16_t file_i = jph_be16(src + 8u + (size_t)i * 2u);
        if (file_i >= nch || seen[file_i]) {
            exr_free(a, map);
            exr_free(a, seen);
            return EXR_ERROR_CORRUPT;
        }
        map[i] = file_i;
        seen[file_i] = 1;
    }
    exr_free(a, seen);
    *out_map = map;
    *out_codestream_off = header_size;
    return EXR_SUCCESS;
}

exr_result exr_jph_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size) {
    uint16_t *map = NULL;
    size_t codestream_off = 0;
    exr_result rc;

    if (!ctx || !src) return EXR_ERROR_INVALID_ARGUMENT;
    rc = jph_parse_ht_header(ctx, src, src_size, &map, &codestream_off);
    if (rc != EXR_SUCCESS) return rc;
    if (codestream_off >= src_size) {
        exr_free(ctx->alloc ? ctx->alloc : exr_default_allocator(), map);
        return EXR_ERROR_CORRUPT;
    }
    rc = jph_validate_profile(ctx, map, src + codestream_off,
                              src_size - codestream_off, dst, dst_size);
    exr_free(ctx->alloc ? ctx->alloc : exr_default_allocator(), map);
    return rc;
}

exr_result exr_jph_compress(const exr_codec_ctx *ctx, const uint8_t *block,
                            size_t n, uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    (void)block;
    (void)n;
    if (out_data) *out_data = NULL;
    if (out_size) *out_size = 0;
    return EXR_ERROR_UNSUPPORTED;
}
