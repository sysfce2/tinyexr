/*
 * TinyEXR texcomp - BC6H encoder
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "texcomp.h"
#include "texcomp_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void tc_bc6h_options_init(tc_bc6h_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof(*opt));
}

size_t tc_bc6h_compressed_size(uint32_t width, uint32_t height) {
    return tc_bc7_compressed_size(width, height);
}

uint16_t tc_float_to_half_bits(float fv) {
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign, mant, exp;
    v.f = fv;
    sign = (v.u >> 16) & 0x8000u;
    exp = (v.u >> 23) & 0xffu;
    mant = v.u & 0x7fffffu;
    if (exp == 255u) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    if (exp > 142u) return (uint16_t)(sign | 0x7c00u);
    if (exp < 113u) {
        uint32_t m;
        if (exp < 103u) return (uint16_t)sign;
        m = mant | 0x800000u;
        m >>= 125u - exp;
        m = (m + 0x1000u) >> 13;
        return (uint16_t)(sign | m);
    }
    exp = exp - 112u;
    mant = (mant + 0x1000u) >> 13;
    if (mant & 0x400u) {
        mant = 0;
        ++exp;
    }
    if (exp >= 31u) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3ffu));
}

static uint32_t tc_bc6h_quant_uf16(float f) {
    uint16_t h;
    uint32_t mag;
    if (!(f > 0.0f)) return 0;
    h = tc_float_to_half_bits(f);
    mag = h & 0x7fffu;
    if (mag > 0x7bffu) mag = 0x7bffu;
    return (mag * 1023u + 15871u) / 31743u;
}

static int32_t tc_bc6h_quant_sf16(float f) {
    uint16_t h;
    uint32_t mag;
    int32_t q;
    if (f == 0.0f) return 0;
    h = tc_float_to_half_bits(f);
    mag = h & 0x7fffu;
    if (mag > 0x7bffu) mag = 0x7bffu;
    q = (int32_t)((mag * 511u + 15871u) / 31743u);
    if (q > 511) q = 511;
    return (h & 0x8000u) ? -q : q;
}

static uint32_t tc_bc6h_unquant_uf16_to_mag(uint32_t q) {
    uint32_t unq;
    if (q == 0u) unq = 0u;
    else if (q >= 1023u) unq = 0xffffu;
    else unq = ((q << 16) + 0x8000u) >> 10;
    return (unq * 31u) >> 6;
}

static int32_t tc_bc6h_unquant_sf16_to_smag(int32_t q) {
    int32_t sign = 0, unq;
    if (q < 0) {
        sign = 1;
        q = -q;
    }
    if (q == 0) unq = 0;
    else if (q >= 511) unq = 0x7fff;
    else unq = (int32_t)(((uint32_t)q << 15) + 0x4000u) >> 9;
    if (sign) unq = -unq;
    if (unq < 0) return -(((-unq) * 31) >> 5);
    return (unq * 31) >> 5;
}

static uint32_t tc_bc6h_err3_mag(const uint32_t a[3], uint32_t r, uint32_t g,
                                 uint32_t b) {
    int32_t dr = (int32_t)a[0] - (int32_t)r;
    int32_t dg = (int32_t)a[1] - (int32_t)g;
    int32_t db = (int32_t)a[2] - (int32_t)b;
    return (uint32_t)(dr * dr + dg * dg + db * db);
}

static uint32_t tc_bc6h_err3_smag(const int32_t a[3], int32_t r, int32_t g,
                                  int32_t b) {
    int64_t dr = (int64_t)a[0] - (int64_t)r;
    int64_t dg = (int64_t)a[1] - (int64_t)g;
    int64_t db = (int64_t)a[2] - (int64_t)b;
    uint64_t e = (uint64_t)(dr * dr + dg * dg + db * db);
    return e > UINT_MAX ? UINT_MAX : (uint32_t)e;
}

static uint32_t tc_bc6h_pack_signed10(int32_t q) {
    if (q < -512) q = -512;
    if (q > 511) q = 511;
    return (uint32_t)q & 1023u;
}

static uint64_t tc_bc6h_choose_selectors_uf16(const uint32_t target[16][3],
                                              const uint32_t lo[3],
                                              const uint32_t hi[3],
                                              uint8_t sel[16]) {
    uint32_t pal[16][3];
    uint64_t err = 0;
    uint32_t i, c, s;
    for (s = 0; s < 16u; ++s) {
        uint32_t w = tc_bc7_weights4[s];
        for (c = 0; c < 3u; ++c) {
            uint32_t qv = ((64u - w) * lo[c] + w * hi[c] + 32u) >> 6;
            pal[s][c] = tc_bc6h_unquant_uf16_to_mag(qv);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t best = 0, best_err = UINT_MAX;
        for (s = 0; s < 16u; ++s) {
            uint32_t e = tc_bc6h_err3_mag(target[i], pal[s][0], pal[s][1], pal[s][2]);
            if (e < best_err) {
                best_err = e;
                best = s;
            }
        }
        sel[i] = (uint8_t)best;
        err += best_err;
    }
    return err;
}

static uint64_t tc_bc6h_choose_selectors_sf16(const int32_t target[16][3],
                                              const int32_t lo[3],
                                              const int32_t hi[3],
                                              uint8_t sel[16]) {
    int32_t pal[16][3];
    uint64_t err = 0;
    uint32_t i, c, s;
    for (s = 0; s < 16u; ++s) {
        uint32_t w = tc_bc7_weights4[s];
        for (c = 0; c < 3u; ++c) {
            int32_t qv = (int32_t)(((int64_t)(64u - w) * lo[c] +
                                    (int64_t)w * hi[c] + 32) >>
                                   6);
            pal[s][c] = tc_bc6h_unquant_sf16_to_smag(qv);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t best = 0, best_err = UINT_MAX;
        for (s = 0; s < 16u; ++s) {
            uint32_t e = tc_bc6h_err3_smag(target[i], pal[s][0], pal[s][1], pal[s][2]);
            if (e < best_err) {
                best_err = e;
                best = s;
            }
        }
        sel[i] = (uint8_t)best;
        err += best_err;
    }
    return err;
}

/* Sign-aware rounded integer division (den > 0). */
static int64_t tc_bc6h_rdiv(int64_t num, int64_t den) {
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
}

/* Least-squares endpoint refinement for mode 11: given the current selectors,
 * re-solve each channel's two 10-bit endpoints (2x2 normal equations in the
 * quantized domain) so the interpolated palette best fits the block, recompute
 * the selectors, and keep the result while the (magnitude-domain) error drops.
 * `maxq` is the endpoint clamp (1023 for uf16, +/-511 for sf16 handled by the
 * signed variant). Returns the best error found. */
static uint64_t tc_bc6h_refine_uf16(const uint32_t q[16][3],
                                    const uint32_t target[16][3], uint32_t lo[3],
                                    uint32_t hi[3], uint8_t sel[16]) {
    uint64_t best = tc_bc6h_choose_selectors_uf16(target, lo, hi, sel);
    int round;
    for (round = 0; round < 3; ++round) {
        uint32_t nlo[3], nhi[3];
        uint8_t nsel[16];
        uint64_t e;
        uint32_t c, i;
        for (c = 0; c < 3u; ++c) {
            int64_t saa = 0, sab = 0, sbb = 0, sap = 0, sbp = 0, det, l, h;
            for (i = 0; i < 16u; ++i) {
                int64_t b = tc_bc7_weights4[sel[i]], a = 64 - b, p = q[i][c];
                saa += a * a;
                sab += a * b;
                sbb += b * b;
                sap += a * p;
                sbp += b * p;
            }
            det = saa * sbb - sab * sab;
            if (det <= 0) {
                nlo[c] = lo[c];
                nhi[c] = hi[c];
                continue;
            }
            l = tc_bc6h_rdiv((sap * sbb - sbp * sab) * 64, det);
            h = tc_bc6h_rdiv((sbp * saa - sap * sab) * 64, det);
            if (l < 0) l = 0;
            if (l > 1023) l = 1023;
            if (h < 0) h = 0;
            if (h > 1023) h = 1023;
            nlo[c] = (uint32_t)l;
            nhi[c] = (uint32_t)h;
        }
        e = tc_bc6h_choose_selectors_uf16(target, nlo, nhi, nsel);
        if (e < best) {
            best = e;
            memcpy(lo, nlo, sizeof(nlo));
            memcpy(hi, nhi, sizeof(nhi));
            memcpy(sel, nsel, 16u);
        } else {
            break;
        }
    }
    return best;
}

/* Signed (sf16) endpoint refinement, clamped to the [-511,511] mode-11 range. */
static uint64_t tc_bc6h_refine_sf16(const int32_t q[16][3],
                                    const int32_t target[16][3], int32_t lo[3],
                                    int32_t hi[3], uint8_t sel[16]) {
    uint64_t best = tc_bc6h_choose_selectors_sf16(target, lo, hi, sel);
    int round;
    for (round = 0; round < 3; ++round) {
        int32_t nlo[3], nhi[3];
        uint8_t nsel[16];
        uint64_t e;
        uint32_t c, i;
        for (c = 0; c < 3u; ++c) {
            int64_t saa = 0, sab = 0, sbb = 0, sap = 0, sbp = 0, det, l, h;
            for (i = 0; i < 16u; ++i) {
                int64_t b = tc_bc7_weights4[sel[i]], a = 64 - b, p = q[i][c];
                saa += a * a;
                sab += a * b;
                sbb += b * b;
                sap += a * p;
                sbp += b * p;
            }
            det = saa * sbb - sab * sab;
            if (det <= 0) {
                nlo[c] = lo[c];
                nhi[c] = hi[c];
                continue;
            }
            l = tc_bc6h_rdiv((sap * sbb - sbp * sab) * 64, det);
            h = tc_bc6h_rdiv((sbp * saa - sap * sab) * 64, det);
            if (l < -511) l = -511;
            if (l > 511) l = 511;
            if (h < -511) h = -511;
            if (h > 511) h = 511;
            nlo[c] = (int32_t)l;
            nhi[c] = (int32_t)h;
        }
        e = tc_bc6h_choose_selectors_sf16(target, nlo, nhi, nsel);
        if (e < best) {
            best = e;
            memcpy(lo, nlo, sizeof(nlo));
            memcpy(hi, nhi, sizeof(nhi));
            memcpy(sel, nsel, 16u);
        } else {
            break;
        }
    }
    return best;
}

static void tc_encode_bc6h_block_uf16(const float pix[16][3], uint8_t out[16]) {
    uint32_t q[16][3], target[16][3], lo[3], hi[3], luma_lo[3], luma_hi[3];
    uint32_t i, c, bitpos = 0, min_l = UINT_MAX, max_l = 0, min_i = 0, max_i = 0;
    uint8_t sel[16], box_sel[16], luma_sel[16];
    memset(out, 0, 16);
    for (c = 0; c < 3u; ++c) {
        lo[c] = UINT_MAX;
        hi[c] = 0;
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t l;
        for (c = 0; c < 3u; ++c) {
            q[i][c] = tc_bc6h_quant_uf16(pix[i][c]);
            target[i][c] = tc_bc6h_unquant_uf16_to_mag(q[i][c]);
            if (q[i][c] < lo[c]) lo[c] = q[i][c];
            if (q[i][c] > hi[c]) hi[c] = q[i][c];
        }
        l = target[i][0] * 38u + target[i][1] * 76u + target[i][2] * 14u;
        if (l < min_l) {
            min_l = l;
            min_i = i;
        }
        if (l >= max_l) {
            max_l = l;
            max_i = i;
        }
    }
    for (c = 0; c < 3u; ++c) {
        luma_lo[c] = q[min_i][c];
        luma_hi[c] = q[max_i][c];
    }
    if (tc_bc6h_choose_selectors_uf16(target, luma_lo, luma_hi, luma_sel) <
        tc_bc6h_choose_selectors_uf16(target, lo, hi, box_sel)) {
        memcpy(lo, luma_lo, sizeof(lo));
        memcpy(hi, luma_hi, sizeof(hi));
        memcpy(sel, luma_sel, sizeof(sel));
    } else {
        memcpy(sel, box_sel, sizeof(sel));
    }
    (void)tc_bc6h_refine_uf16(q, target, lo, hi, sel);
    if (sel[0] & 8u) {
        uint32_t t;
        for (c = 0; c < 3u; ++c) {
            t = lo[c];
            lo[c] = hi[c];
            hi[c] = t;
        }
        for (i = 0; i < 16u; ++i) sel[i] = (uint8_t)(15u - sel[i]);
    }

    tc_set_bits(out, &bitpos, 0x03u, 5); /* BC6H mode 11: bit pattern 00011 */
    tc_set_bits(out, &bitpos, lo[0], 10);
    tc_set_bits(out, &bitpos, lo[1], 10);
    tc_set_bits(out, &bitpos, lo[2], 10);
    tc_set_bits(out, &bitpos, hi[0], 10);
    tc_set_bits(out, &bitpos, hi[1], 10);
    tc_set_bits(out, &bitpos, hi[2], 10);
    tc_set_bits(out, &bitpos, sel[0], 3);
    for (i = 1; i < 16u; ++i) tc_set_bits(out, &bitpos, sel[i], 4);
}

static void tc_encode_bc6h_block_sf16(const float pix[16][3], uint8_t out[16]) {
    int32_t q[16][3], target[16][3], lo[3], hi[3], luma_lo[3], luma_hi[3];
    uint32_t i, c, bitpos = 0, min_i = 0, max_i = 0;
    int32_t min_l = INT_MAX, max_l = INT_MIN;
    uint8_t sel[16], box_sel[16], luma_sel[16];
    memset(out, 0, 16);
    for (c = 0; c < 3u; ++c) {
        lo[c] = INT_MAX;
        hi[c] = INT_MIN;
    }
    for (i = 0; i < 16u; ++i) {
        int32_t l;
        for (c = 0; c < 3u; ++c) {
            q[i][c] = tc_bc6h_quant_sf16(pix[i][c]);
            target[i][c] = tc_bc6h_unquant_sf16_to_smag(q[i][c]);
            if (q[i][c] < lo[c]) lo[c] = q[i][c];
            if (q[i][c] > hi[c]) hi[c] = q[i][c];
        }
        l = target[i][0] * 38 + target[i][1] * 76 + target[i][2] * 14;
        if (l < min_l) {
            min_l = l;
            min_i = i;
        }
        if (l >= max_l) {
            max_l = l;
            max_i = i;
        }
    }
    for (c = 0; c < 3u; ++c) {
        luma_lo[c] = q[min_i][c];
        luma_hi[c] = q[max_i][c];
    }
    if (tc_bc6h_choose_selectors_sf16(target, luma_lo, luma_hi, luma_sel) <
        tc_bc6h_choose_selectors_sf16(target, lo, hi, box_sel)) {
        memcpy(lo, luma_lo, sizeof(lo));
        memcpy(hi, luma_hi, sizeof(hi));
        memcpy(sel, luma_sel, sizeof(sel));
    } else {
        memcpy(sel, box_sel, sizeof(sel));
    }
    (void)tc_bc6h_refine_sf16(q, target, lo, hi, sel);
    if (sel[0] & 8u) {
        int32_t t;
        for (c = 0; c < 3u; ++c) {
            t = lo[c];
            lo[c] = hi[c];
            hi[c] = t;
        }
        for (i = 0; i < 16u; ++i) sel[i] = (uint8_t)(15u - sel[i]);
    }

    tc_set_bits(out, &bitpos, 0x03u, 5);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[0]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[1]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(lo[2]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[0]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[1]), 10);
    tc_set_bits(out, &bitpos, tc_bc6h_pack_signed10(hi[2]), 10);
    tc_set_bits(out, &bitpos, sel[0], 3);
    for (i = 1; i < 16u; ++i) tc_set_bits(out, &bitpos, sel[i], 4);
}

tc_result tc_bc6h_compress_rgb32f(const float *rgb, uint32_t width,
                                  uint32_t height, size_t stride_bytes,
                                  const tc_bc6h_options *opt,
                                  uint8_t *out_bc6h, size_t out_size) {
    uint32_t bx, by, x, y, xx, yy;
    float block[16][3];
    size_t need, off = 0;
    (void)opt;

    if (!rgb || !out_bc6h || !width || !height) return TC_ERROR_INVALID_ARGUMENT;
    if (stride_bytes < (size_t)width * 3u * sizeof(float))
        return TC_ERROR_INVALID_ARGUMENT;
    need = tc_bc6h_compressed_size(width, height);
    if (!need || out_size < need) return TC_ERROR_INVALID_ARGUMENT;

    for (by = 0; by < height; by += 4) {
        for (bx = 0; bx < width; bx += 4) {
            for (yy = 0; yy < 4; ++yy) {
                y = by + yy;
                if (y >= height) y = height - 1u;
                for (xx = 0; xx < 4; ++xx) {
                    const float *src;
                    x = bx + xx;
                    if (x >= width) x = width - 1u;
                    src = (const float *)((const uint8_t *)rgb +
                                          (size_t)y * stride_bytes) +
                          (size_t)x * 3u;
                    block[yy * 4u + xx][0] = src[0];
                    block[yy * 4u + xx][1] = src[1];
                    block[yy * 4u + xx][2] = src[2];
                }
            }
            if (opt && opt->signed_float)
                tc_encode_bc6h_block_sf16(block, out_bc6h + off);
            else
                tc_encode_bc6h_block_uf16(block, out_bc6h + off);
            off += 16u;
        }
    }

    return TC_SUCCESS;
}
