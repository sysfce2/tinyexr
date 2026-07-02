/*
 * TinyEXR texcomp - BC7 encoder
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "texcomp.h"
#include "texcomp_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

const uint32_t tc_bc7_weights4[16] = {0,  4,  9,  13, 17, 21, 26, 30,
                                             34, 38, 43, 47, 51, 55, 60, 64};
static const uint32_t tc_bc7_weights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
static const uint32_t tc_bc7_weights2[4] = {0, 21, 43, 64};

static const uint8_t tc_bc7_num_subsets[8] = {3, 2, 3, 2, 1, 1, 1, 2};
static const uint8_t tc_bc7_partition_bits[8] = {4, 6, 6, 6, 0, 0, 0, 6};
static const uint8_t tc_bc7_color_index_bits[8] = {3, 3, 2, 2, 2, 2, 4, 2};
static const uint8_t tc_bc7_alpha_index_bits[8] = {0, 0, 0, 0, 3, 2, 4, 2};
static const uint8_t tc_bc7_color_precision[8] = {4, 6, 5, 7, 5, 7, 7, 5};
static const uint8_t tc_bc7_alpha_precision[8] = {0, 0, 0, 0, 6, 8, 7, 5};
static const uint8_t tc_bc7_has_pbits[8] = {1, 1, 0, 1, 0, 0, 1, 1};
static const uint8_t tc_bc7_shared_pbits[8] = {0, 1, 0, 0, 0, 0, 0, 0};
static const uint8_t tc_bc7_sep_alpha[8] = {0, 0, 0, 0, 1, 1, 0, 0};

static const uint8_t tc_bc7_partition1[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t tc_bc7_partition2[64 * 16] = {
    0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,    0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,    0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,    0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,    0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,    0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,    0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,    0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
    0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,    0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,    0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,    0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,    0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,    0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,    0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
    0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,    0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,    0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,    0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,    0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,    0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,    0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,    0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,    0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,    0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,    0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,    0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,    0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,    0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
    0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,    0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,    0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,    0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,    0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,    0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,    0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
    0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,    0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,    0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,    0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,    0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,    0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,    0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,    0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
    0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,    0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,    0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,    0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,    0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,    0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,    0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,    0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};
static const uint8_t tc_bc7_partition3_0[16] = {0, 0, 1, 1, 0, 0, 1, 1,
                                                0, 2, 2, 1, 2, 2, 2, 2};
static const uint8_t tc_bc7_anchor2[64] = {
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,15,15,15,15,15, 2, 2,15
};
static const uint8_t tc_bc7_anchor3_0[2] = {3, 15};

static const uint8_t tc_bc7_part_lut[8][21] = {
    {5, 9, 12, 27, 28, 47, 49, 51, 53},
    {34, 38, 39, 44},
    {2, 26, 32, 37, 46, 63},
    {16, 21, 25, 31, 43, 55, 59},
    {14, 29, 33, 36, 45, 60},
    {17, 19, 23, 30, 40, 54, 57},
    {48, 52, 56, 58},
    {0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 15, 18, 20, 22, 24, 35, 41, 42, 50, 61, 62}
};
static const uint8_t tc_bc7_alpha_part_lut[8][12] = {
    {4, 7, 8, 11, 52, 60, 63},
    {27, 28, 30, 31, 34, 35, 36, 37, 40, 41, 42, 43},
    {0, 1, 2, 3, 23, 32, 39, 45, 58, 59},
    {18, 21, 22, 25, 54, 62},
    {10, 13, 14, 15, 16, 33, 38, 46, 56, 57},
    {17, 19, 20, 24, 55, 61},
    {5, 6, 9, 12, 53},
    {26, 29, 44, 47, 48, 49, 50, 51}
};

void tc_bc7_options_init(tc_bc7_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
    opt->quality = TC_BC7_QUALITY_QUICKBC7;
    opt->perceptual = 1;
    opt->quick = 1;
    opt->threads = 1;
    opt->mode_mask = 0xffu;
}

size_t tc_bc7_compressed_size(uint32_t width, uint32_t height) {
    size_t bx, by, blocks;
    if (!width || !height) return 0;
    bx = ((size_t)width + 3u) >> 2;
    by = ((size_t)height + 3u) >> 2;
    if (tc_mul_ovf_size(bx, by, &blocks)) return 0;
    if (blocks > SIZE_MAX / 16u) return 0;
    return blocks * 16u;
}

static uint8_t tc_unquant(uint32_t q, uint32_t bits) {
    uint32_t maxv;
    if (bits >= 8u) return (uint8_t)q;
    maxv = (1u << bits) - 1u;
    return (uint8_t)((q * 255u + maxv / 2u) / maxv);
}

static uint32_t tc_quant(uint32_t v, uint32_t bits) {
    uint32_t maxv = (1u << bits) - 1u;
    return (v * maxv + 127u) / 255u;
}

static uint32_t tc_block_quick_mask(const uint8_t pix[16][4], uint32_t user_mask) {
    uint32_t i;
    uint32_t lmin = UINT_MAX, lmax = 0, amin = UINT_MAX, amax = 0;
    uint32_t has_alpha = 0, lstate, astate, mask = 0;
    for (i = 0; i < 16u; ++i) {
        uint32_t y = tc_luma_u8(pix[i]) >> 7;
        uint32_t a = pix[i][3];
        if (y < lmin) lmin = y;
        if (y > lmax) lmax = y;
        if (a < amin) amin = a;
        if (a > amax) amax = a;
        if (a < 255u) has_alpha = 1;
    }

    lstate = (lmax - lmin <= 19u) ? 0u : ((lmax - lmin <= 48u) ? 1u : 2u);
    if (!has_alpha) {
        if (lstate == 0u) mask = 1u << 6;
        else if (lstate == 1u) mask = (1u << 1) | (1u << 6);
        else mask = 1u << 1;
    } else {
        uint32_t adiff = amax - amin;
        astate = (adiff < 10u) ? 0u : ((adiff <= 21u) ? 1u : 2u);
        if (lstate <= 1u && astate <= 1u) mask |= 1u << 6;
        if (lstate <= 1u && astate >= 1u) mask |= 1u << 5;
        if (lstate >= 1u) mask |= 1u << 7;
        if (!mask) mask = 1u << 6;
    }

    mask &= user_mask;
    return mask ? mask : user_mask;
}

static uint32_t tc_err4(const uint8_t *a, uint8_t r, uint8_t g, uint8_t b,
                        uint8_t al, int has_alpha) {
    int dr = (int)a[0] - (int)r;
    int dg = (int)a[1] - (int)g;
    int db = (int)a[2] - (int)b;
    int da = has_alpha ? ((int)a[3] - (int)al) : 0;
    return (uint32_t)(dr * dr + dg * dg + db * db + da * da);
}

static uint32_t tc_err3(const uint8_t *a, uint8_t r, uint8_t g, uint8_t b) {
    int dr = (int)a[0] - (int)r;
    int dg = (int)a[1] - (int)g;
    int db = (int)a[2] - (int)b;
    return (uint32_t)(dr * dr + dg * dg + db * db);
}

static uint32_t tc_err1(uint8_t a, uint8_t b) {
    int d = (int)a - (int)b;
    return (uint32_t)(d * d);
}

typedef struct tc_bc7_candidate {
    uint8_t mode;
    uint8_t partition;
    uint8_t index_selector;
    uint8_t rotation;
    uint8_t lo[3][4];
    uint8_t hi[3][4];
    uint8_t pbits[3][2];
    uint8_t selectors[16];
    uint8_t alpha_selectors[16];
} tc_bc7_candidate;

static const uint8_t *tc_partition_for(uint32_t mode, uint32_t partition) {
    if (tc_bc7_num_subsets[mode] == 1u) return tc_bc7_partition1;
    if (tc_bc7_num_subsets[mode] == 2u)
        return &tc_bc7_partition2[(partition & 63u) * 16u];
    return tc_bc7_partition3_0;
}

static uint8_t tc_anchor_for_subset(uint32_t mode, uint32_t partition,
                                    uint32_t subset) {
    if (subset == 0u) return 0;
    if (tc_bc7_num_subsets[mode] == 3u) return tc_bc7_anchor3_0[subset - 1u];
    return tc_bc7_anchor2[partition & 63u];
}

static const uint32_t *tc_weights_for_bits(uint32_t bits) {
    if (bits == 2u) return tc_bc7_weights2;
    if (bits == 3u) return tc_bc7_weights3;
    return tc_bc7_weights4;
}

static uint8_t tc_decode_endpoint(uint8_t q, uint8_t p, uint32_t precision,
                                  uint32_t has_pbit) {
    if (has_pbit) return tc_unquant(((uint32_t)q << 1u) | p, precision + 1u);
    return tc_unquant(q, precision);
}

static void tc_quant_endpoint(uint8_t v, uint32_t precision, uint32_t has_pbit,
                              uint32_t shared_pbit, uint8_t shared_p,
                              uint8_t *q, uint8_t *p) {
    if (has_pbit) {
        uint32_t total_bits = precision + 1u;
        uint32_t u = tc_quant(v, total_bits);
        if (shared_pbit) {
            *p = shared_p;
            *q = (uint8_t)(u >> 1u);
        } else {
            *p = (uint8_t)(u & 1u);
            *q = (uint8_t)(u >> 1u);
        }
    } else {
        *p = 0;
        *q = (uint8_t)tc_quant(v, precision);
    }
}

static void tc_recon_color(const tc_bc7_candidate *cand, uint32_t mode,
                           uint32_t subset, uint32_t sel, uint32_t asel,
                           uint8_t out[4]) {
    uint32_t c;
    uint32_t cbits = tc_bc7_color_index_bits[mode] + cand->index_selector;
    uint32_t abits = tc_bc7_alpha_index_bits[mode] - cand->index_selector;
    const uint32_t *cw = tc_weights_for_bits(cbits);
    const uint32_t *aw = tc_weights_for_bits(abits ? abits : cbits);
    uint32_t w = cw[sel];
    uint32_t awv = aw[asel];

    for (c = 0; c < 3u; ++c) {
        uint32_t a = tc_decode_endpoint(cand->lo[subset][c], cand->pbits[subset][0],
                                        tc_bc7_color_precision[mode],
                                        tc_bc7_has_pbits[mode]);
        uint32_t b = tc_decode_endpoint(cand->hi[subset][c], cand->pbits[subset][1],
                                        tc_bc7_color_precision[mode],
                                        tc_bc7_has_pbits[mode]);
        out[c] = (uint8_t)(((64u - w) * a + w * b + 32u) >> 6);
    }
    if (mode < 4u) {
        out[3] = 255u;
    } else {
        uint32_t ap = tc_bc7_has_pbits[mode] && !tc_bc7_sep_alpha[mode];
        uint32_t a = tc_decode_endpoint(cand->lo[subset][3], cand->pbits[subset][0],
                                        tc_bc7_alpha_precision[mode], ap);
        uint32_t b = tc_decode_endpoint(cand->hi[subset][3], cand->pbits[subset][1],
                                        tc_bc7_alpha_precision[mode], ap);
        if (!tc_bc7_sep_alpha[mode]) awv = w;
        out[3] = (uint8_t)(((64u - awv) * a + awv * b + 32u) >> 6);
    }
}

static void tc_fill_palette(const tc_bc7_candidate *cand, uint32_t mode,
                            uint8_t pal[3][16][4]) {
    uint32_t subset, sel, max_sel = 1u << (tc_bc7_color_index_bits[mode] + cand->index_selector);
    uint32_t max_asel = tc_bc7_alpha_index_bits[mode] > cand->index_selector
                            ? (1u << (tc_bc7_alpha_index_bits[mode] - cand->index_selector))
                            : max_sel;
    uint32_t n = max_sel > max_asel ? max_sel : max_asel;
    for (subset = 0; subset < tc_bc7_num_subsets[mode]; ++subset) {
        for (sel = 0; sel < n; ++sel) {
            uint32_t cs = sel < max_sel ? sel : max_sel - 1u;
            uint32_t as = sel < max_asel ? sel : max_asel - 1u;
            tc_recon_color(cand, mode, subset, cs, as, pal[subset][sel]);
        }
    }
}

static void tc_pack_candidate(const tc_bc7_candidate *src, uint8_t out[16]) {
    tc_bc7_candidate cand = *src;
    const uint8_t *part = tc_partition_for(cand.mode, cand.partition);
    uint32_t subsets = tc_bc7_num_subsets[cand.mode];
    uint32_t bitpos = 0, subset, comp, i;
    uint8_t anchor[3] = {0, 0, 0};

    for (subset = 0; subset < subsets; ++subset) {
        uint32_t anchor_index = tc_anchor_for_subset(cand.mode, cand.partition, subset);
        uint32_t cidx_bits = tc_bc7_color_index_bits[cand.mode] + cand.index_selector;
        uint32_t ncolor = 1u << cidx_bits;
        anchor[subset] = (uint8_t)anchor_index;
        if (cand.selectors[anchor_index] & (ncolor >> 1u)) {
            uint8_t tq[4], tp;
            for (i = 0; i < 16u; ++i)
                if (part[i] == subset) cand.selectors[i] = (uint8_t)((ncolor - 1u) - cand.selectors[i]);
            for (comp = 0; comp < (tc_bc7_sep_alpha[cand.mode] ? 3u : 4u); ++comp) {
                tq[comp] = cand.lo[subset][comp];
                cand.lo[subset][comp] = cand.hi[subset][comp];
                cand.hi[subset][comp] = tq[comp];
            }
            if (!tc_bc7_shared_pbits[cand.mode]) {
                tp = cand.pbits[subset][0];
                cand.pbits[subset][0] = cand.pbits[subset][1];
                cand.pbits[subset][1] = tp;
            }
        }
        if (tc_bc7_sep_alpha[cand.mode]) {
            uint32_t aidx_bits = tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector;
            uint32_t nalpha = 1u << aidx_bits;
            if (cand.alpha_selectors[anchor_index] & (nalpha >> 1u)) {
                uint8_t tq = cand.lo[subset][3];
                for (i = 0; i < 16u; ++i)
                    if (part[i] == subset)
                        cand.alpha_selectors[i] = (uint8_t)((nalpha - 1u) - cand.alpha_selectors[i]);
                cand.lo[subset][3] = cand.hi[subset][3];
                cand.hi[subset][3] = tq;
            }
        }
    }

    memset(out, 0, 16);
    tc_set_bits(out, &bitpos, 1u << cand.mode, cand.mode + 1u);
    if (cand.mode == 4u || cand.mode == 5u) tc_set_bits(out, &bitpos, cand.rotation, 2);
    if (cand.mode == 4u) tc_set_bits(out, &bitpos, cand.index_selector, 1);
    if (tc_bc7_partition_bits[cand.mode])
        tc_set_bits(out, &bitpos, cand.partition, tc_bc7_partition_bits[cand.mode]);

    for (comp = 0; comp < (cand.mode >= 4u ? 4u : 3u); ++comp) {
        uint32_t prec = comp == 3u ? tc_bc7_alpha_precision[cand.mode]
                                   : tc_bc7_color_precision[cand.mode];
        for (subset = 0; subset < subsets; ++subset) {
            tc_set_bits(out, &bitpos, cand.lo[subset][comp], prec);
            tc_set_bits(out, &bitpos, cand.hi[subset][comp], prec);
        }
    }
    if (tc_bc7_has_pbits[cand.mode]) {
        for (subset = 0; subset < subsets; ++subset) {
            tc_set_bits(out, &bitpos, cand.pbits[subset][0], 1);
            if (!tc_bc7_shared_pbits[cand.mode]) tc_set_bits(out, &bitpos, cand.pbits[subset][1], 1);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t n = cand.index_selector ? (tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector)
                                         : (tc_bc7_color_index_bits[cand.mode] + cand.index_selector);
        if (i == anchor[0] || i == anchor[1] || i == anchor[2]) n--;
        tc_set_bits(out, &bitpos,
                    cand.index_selector ? cand.alpha_selectors[i] : cand.selectors[i], n);
    }
    if (tc_bc7_sep_alpha[cand.mode]) {
        for (i = 0; i < 16u; ++i) {
            uint32_t n = cand.index_selector ? (tc_bc7_color_index_bits[cand.mode] + cand.index_selector)
                                             : (tc_bc7_alpha_index_bits[cand.mode] - cand.index_selector);
            if (i == anchor[0] || i == anchor[1] || i == anchor[2]) n--;
            tc_set_bits(out, &bitpos,
                        cand.index_selector ? cand.selectors[i] : cand.alpha_selectors[i], n);
        }
    }
}

static uint64_t tc_build_candidate(uint32_t mode, uint32_t partition,
                                   const uint8_t pix[16][4],
                                   tc_bc7_candidate *cand) {
    const uint8_t *part = tc_partition_for(mode, partition);
    uint32_t subsets = tc_bc7_num_subsets[mode];
    uint32_t subset, i, c;
    uint64_t total_err = 0;
    memset(cand, 0, sizeof(*cand));
    cand->mode = (uint8_t)mode;
    cand->partition = (uint8_t)partition;
    cand->index_selector = 0;
    cand->rotation = 0;

    for (subset = 0; subset < subsets; ++subset) {
        uint32_t min_l = UINT_MAX, max_l = 0, min_i = 0, max_i = 0;
        for (i = 0; i < 16u; ++i) {
            if (part[i] == subset) {
                uint32_t y = tc_luma_u8(pix[i]);
                if (y < min_l) {
                    min_l = y;
                    min_i = i;
                }
                if (y >= max_l) {
                    max_l = y;
                    max_i = i;
                }
            }
        }
        for (c = 0; c < 4u; ++c) {
            uint8_t qp0, qp1;
            uint32_t prec = c == 3u && mode >= 4u ? tc_bc7_alpha_precision[mode]
                                                   : tc_bc7_color_precision[mode];
            uint32_t has_p = tc_bc7_has_pbits[mode] && (c < 3u || !tc_bc7_sep_alpha[mode]);
            uint8_t shared = (uint8_t)(((pix[min_i][c] + pix[max_i][c]) >> 1u) & 1u);
            if (mode < 4u && c == 3u) {
                cand->lo[subset][c] = 0;
                cand->hi[subset][c] = 0;
                continue;
            }
            tc_quant_endpoint(pix[min_i][c], prec, has_p, tc_bc7_shared_pbits[mode],
                              shared, &cand->lo[subset][c], &qp0);
            tc_quant_endpoint(pix[max_i][c], prec, has_p, tc_bc7_shared_pbits[mode],
                              shared, &cand->hi[subset][c], &qp1);
            if (tc_bc7_shared_pbits[mode]) {
                cand->pbits[subset][0] = shared;
                cand->pbits[subset][1] = shared;
            } else if (has_p) {
                cand->pbits[subset][0] = qp0;
                cand->pbits[subset][1] = qp1;
            }
        }
    }

    {
        uint8_t pal[3][16][4];
        uint32_t cbits = tc_bc7_color_index_bits[mode] + cand->index_selector;
        uint32_t abits = tc_bc7_alpha_index_bits[mode] - cand->index_selector;
        uint32_t nc = 1u << cbits;
        uint32_t na = abits ? (1u << abits) : nc;
        tc_fill_palette(cand, mode, pal);
        for (i = 0; i < 16u; ++i) {
            uint32_t subset = part[i], s, best_s = 0, best_a = 0;
            uint32_t best = UINT_MAX;
            uint32_t best_color_err = 0;
            uint32_t best_alpha_err = 0;
            for (s = 0; s < nc; ++s) {
                const uint8_t *r = pal[subset][s];
                uint32_t e;
                if (tc_bc7_sep_alpha[mode] || mode < 4u)
                    e = tc_err3(pix[i], r[0], r[1], r[2]);
                else
                    e = tc_err4(pix[i], r[0], r[1], r[2], r[3], 1);
                if (e < best) {
                    best = e;
                    best_s = s;
                }
            }
            best_color_err = best;
            if (tc_bc7_sep_alpha[mode]) {
                best = UINT_MAX;
                for (s = 0; s < na; ++s) {
                    const uint8_t *r = pal[subset][s];
                    uint32_t e = tc_err1(pix[i][3], r[3]);
                    if (e < best) {
                        best = e;
                        best_a = s;
                    }
                }
                best_alpha_err = best;
            } else {
                best_a = best_s;
                best_alpha_err = 0;
            }
            cand->selectors[i] = (uint8_t)best_s;
            cand->alpha_selectors[i] = (uint8_t)best_a;
            total_err += (uint64_t)best_color_err + (uint64_t)best_alpha_err;
        }
    }
    return total_err;
}

static uint8_t tc_lookup_index_from_mask(uint32_t mask) {
    switch (mask) {
        case 11u: return 0;
        case 12u: return 1;
        case 18u: return 2;
        case 21u: return 3;
        case 33u: return 4;
        case 38u: return 5;
        case 56u: return 6;
        case 63u: return 7;
        default: return 0;
    }
}

static uint8_t tc_dominant_rgb_channel(const uint8_t *p) {
    if (p[0] > p[1]) return p[0] > p[2] ? 0u : 2u;
    return p[1] > p[2] ? 1u : 2u;
}

static uint8_t tc_rgb_partition_lut_index(const uint8_t pix[16][4]) {
    const uint8_t ids[4] = {0, 1, 4, 5};
    uint8_t ch[4];
    uint32_t i, j, bit = 0, mask = 0;
    for (i = 0; i < 4u; ++i) ch[i] = tc_dominant_rgb_channel(pix[ids[i]]);
    for (i = 0; i < 4u; ++i) {
        for (j = i + 1u; j < 4u; ++j) {
            if (ch[i] == ch[j]) mask |= 1u << bit;
            ++bit;
        }
    }
    return tc_lookup_index_from_mask(mask);
}

static uint8_t tc_alpha_partition_lut_index(const uint8_t pix[16][4]) {
    const uint8_t ids[4] = {0, 3, 12, 15};
    uint32_t i, j, bit = 0, mask = 0;
    uint8_t cls[4], amin = 255, amax = 0, median;
    for (i = 0; i < 16u; ++i) {
        if (pix[i][3] < amin) amin = pix[i][3];
        if (pix[i][3] > amax) amax = pix[i][3];
    }
    median = (uint8_t)(((uint32_t)amin + (uint32_t)amax) >> 1u);
    for (i = 0; i < 4u; ++i) cls[i] = pix[ids[i]][3] > median ? 1u : 0u;
    for (i = 0; i < 4u; ++i) {
        for (j = i + 1u; j < 4u; ++j) {
            if (cls[i] == cls[j]) mask |= 1u << bit;
            ++bit;
        }
    }
    return tc_lookup_index_from_mask(mask);
}

static uint64_t tc_partition_cluster_score(const uint8_t pix[16][4],
                                           uint32_t partition,
                                           uint32_t include_alpha) {
    const uint8_t *part = &tc_bc7_partition2[(partition & 63u) * 16u];
    uint32_t sums[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    uint32_t total[4] = {0, 0, 0, 0};
    uint32_t counts[2] = {0, 0};
    uint32_t i, c, comps = include_alpha ? 4u : 3u;
    uint64_t score = 0;
    for (i = 0; i < 16u; ++i) {
        uint32_t s = part[i];
        ++counts[s];
        for (c = 0; c < comps; ++c) {
            sums[s][c] += pix[i][c];
            total[c] += pix[i][c];
        }
    }
    for (i = 0; i < 2u; ++i) {
        if (!counts[i]) continue;
        for (c = 0; c < comps; ++c) {
            int32_t d = (int32_t)(16u * sums[i][c]) -
                        (int32_t)(counts[i] * total[c]);
            score += ((uint64_t)(uint32_t)(d < 0 ? -d : d) *
                      (uint64_t)(uint32_t)(d < 0 ? -d : d)) /
                     counts[i];
        }
    }
    return score;
}

static uint32_t tc_select_partition2(const uint8_t pix[16][4], uint32_t mode,
                                     uint32_t quick) {
    uint32_t best_partition = 0, i, count;
    uint64_t best_score = 0;
    if (!quick) {
        count = 64u;
        for (i = 0; i < count; ++i) {
            uint64_t score = tc_partition_cluster_score(pix, i, mode == 7u);
            if (score > best_score || i == 0u) {
                best_score = score;
                best_partition = i;
            }
        }
    } else if (mode == 7u) {
        uint8_t lut = tc_alpha_partition_lut_index(pix);
        (void)best_score;
        best_partition = tc_bc7_alpha_part_lut[lut][0];
    } else {
        uint8_t lut = tc_rgb_partition_lut_index(pix);
        (void)best_score;
        best_partition = tc_bc7_part_lut[lut][0];
    }
    return best_partition;
}

static void tc_encode_bc7_all_modes_block(const uint8_t pix[16][4],
                                          const tc_bc7_options *opt,
                                          uint8_t out[16]) {
    uint32_t mode;
    uint64_t best_err = UINT64_MAX;
    uint8_t best_block[16];
    uint32_t mask = opt && opt->mode_mask ? opt->mode_mask : 0xffu;
    if (opt && opt->quick) mask = tc_block_quick_mask(pix, mask);
    memset(best_block, 0, sizeof(best_block));
    for (mode = 0; mode < 8u; ++mode) {
        tc_bc7_candidate cand;
        uint64_t err;
        if ((mask & (1u << mode)) == 0u) continue;
        err = tc_build_candidate(mode,
                                 (mode == 1u || mode == 7u)
                                     ? tc_select_partition2(pix, mode,
                                                            opt && opt->quick)
                                     : 0u,
                                 pix, &cand);
        if (err < best_err) {
            best_err = err;
            tc_pack_candidate(&cand, best_block);
        }
    }
    memcpy(out, best_block, 16);
}

tc_result tc_bc7_compress_rgba8(const uint8_t *rgba, uint32_t width,
                                uint32_t height, size_t stride,
                                const tc_bc7_options *opt, uint8_t *out_bc7,
                                size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    uint8_t block[16][4];
    size_t need, off = 0;
    tc_bc7_options defopt;

    if (!rgba || !out_bc7 || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride < (size_t)width * 4u) return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc7_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;
    if (!opt) {
        tc_bc7_options_init(&defopt);
        opt = &defopt;
    }

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const uint8_t *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = rgba + (size_t)y * stride + (size_t)x * 4u;
                    memcpy(block[yy * 4u + xx], src, 4);
                }
            }
            tc_encode_bc7_all_modes_block(block, opt, out_bc7 + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}
