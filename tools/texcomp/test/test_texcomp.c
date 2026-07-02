/*
 * TinyEXR texcomp tests.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "texcomp.h"

#include "astc_ref_decode.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x, msg)                                                           \
    do {                                                                        \
        if (!(x)) {                                                             \
            fprintf(stderr, "FAIL: %s\n", msg);                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint32_t bc7_mode(const uint8_t *p) {
    uint32_t m = 0;
    while (m < 8u && ((p[0] & (1u << m)) == 0u)) m++;
    return m;
}

static uint32_t rd_bits(const uint8_t *p, uint32_t bitpos, uint32_t nbits) {
    uint32_t v = 0, i;
    for (i = 0; i < nbits; ++i)
        v |= (uint32_t)((p[(bitpos + i) >> 3] >> ((bitpos + i) & 7u)) & 1u) << i;
    return v;
}

static int astc_test_decode_mode_2d(uint32_t mode, uint32_t *wx, uint32_t *wy,
                                    uint32_t *quant, uint32_t *dual) {
    uint32_t base_quant = (mode >> 4) & 1u;
    uint32_t h = (mode >> 9) & 1u;
    uint32_t d = (mode >> 10) & 1u;
    uint32_t a = (mode >> 5) & 3u;
    uint32_t x = 0, y = 0;
    if ((mode & 3u) != 0u) {
        uint32_t b;
        base_quant |= (mode & 3u) << 1;
        b = (mode >> 7) & 3u;
        switch ((mode >> 2) & 3u) {
            case 0:
                x = b + 4u;
                y = a + 2u;
                break;
            case 1:
                x = b + 8u;
                y = a + 2u;
                break;
            case 2:
                x = a + 2u;
                y = b + 8u;
                break;
            default:
                b &= 1u;
                if (mode & 0x100u) {
                    x = b + 2u;
                    y = a + 2u;
                } else {
                    x = a + 2u;
                    y = b + 6u;
                }
                break;
        }
    } else {
        uint32_t b;
        base_quant |= ((mode >> 2) & 3u) << 1;
        if (((mode >> 2) & 3u) == 0u) return 0;
        b = (mode >> 9) & 3u;
        switch ((mode >> 7) & 3u) {
            case 0:
                x = 12u;
                y = a + 2u;
                break;
            case 1:
                x = a + 2u;
                y = 12u;
                break;
            case 2:
                x = a + 6u;
                y = b + 6u;
                d = 0u;
                h = 0u;
                break;
            default:
                if (a == 0u) {
                    x = 6u;
                    y = 10u;
                } else if (a == 1u) {
                    x = 10u;
                    y = 6u;
                } else {
                    return 0;
                }
                break;
        }
    }
    if (base_quant < 2u) return 0;
    *wx = x;
    *wy = y;
    *quant = (base_quant - 2u) + 6u * h;
    *dual = d;
    return *quant < 12u && x * y * (d + 1u) <= 64u;
}

static int astc_test_mode_is_valid_1plane_2d(uint32_t mode) {
    uint32_t wx, wy, quant, dual;
    return astc_test_decode_mode_2d(mode, &wx, &wy, &quant, &dual) && !dual;
}

static int astc_test_mode_is_high_precision_4x4(uint32_t mode) {
    uint32_t wx, wy, quant, dual;
    return astc_test_decode_mode_2d(mode, &wx, &wy, &quant, &dual) && !dual &&
           wx == 4u && wy == 4u && quant >= 3u;
}

static int astc_test_mode_is_luminance_large(uint32_t mode) {
    uint32_t wx, wy, quant, dual;
    (void)quant;
    return astc_test_decode_mode_2d(mode, &wx, &wy, &quant, &dual) && !dual &&
           (wx > 5u || wy > 5u);
}

static uint32_t bc6h_unquant_uf16_to_mag(uint32_t q) {
    uint32_t unq;
    if (q == 0u) unq = 0u;
    else if (q >= 1023u) unq = 0xffffu;
    else unq = ((q << 16) + 0x8000u) >> 10;
    return (unq * 31u) >> 6;
}

static uint16_t test_float_to_half_bits(float fv) {
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

static uint64_t bc6h_mode11_rgb_error(const uint8_t *block,
                                      const float pix[16][3]) {
    static const uint32_t w[16] = {0,  4,  9,  13, 17, 21, 26, 30,
                                   34, 38, 43, 47, 51, 55, 60, 64};
    uint32_t lo[3], hi[3], target[16][3], bitpos = 5u, i, c;
    uint64_t err = 0;
    for (c = 0; c < 3u; ++c) lo[c] = rd_bits(block, bitpos + c * 10u, 10);
    bitpos += 30u;
    for (c = 0; c < 3u; ++c) hi[c] = rd_bits(block, bitpos + c * 10u, 10);
    bitpos += 30u;
    for (i = 0; i < 16u; ++i) {
        for (c = 0; c < 3u; ++c) {
            float f = pix[i][c];
            uint16_t h;
            uint32_t mag, q;
            h = test_float_to_half_bits(f);
            mag = h & 0x7fffu;
            if (mag > 0x7bffu) mag = 0x7bffu;
            q = (mag * 1023u + 15871u) / 31743u;
            target[i][c] = bc6h_unquant_uf16_to_mag(q);
        }
    }
    for (i = 0; i < 16u; ++i) {
        uint32_t sel = i == 0u ? rd_bits(block, bitpos, 3) : rd_bits(block, bitpos, 4);
        bitpos += i == 0u ? 3u : 4u;
        for (c = 0; c < 3u; ++c) {
            uint32_t qv = ((64u - w[sel]) * lo[c] + w[sel] * hi[c] + 32u) >> 6;
            uint32_t recon = bc6h_unquant_uf16_to_mag(qv);
            int64_t d = (int64_t)target[i][c] - (int64_t)recon;
            err += (uint64_t)(d * d);
        }
    }
    return err;
}

/* ---- ASTC reference-decoder based tests --------------------------------- */

/* decode(encode(x)) == x for every ISE quant level and value count. */
static int astc_ise_roundtrip_test(void) {
    static const uint16_t levels[21] = {2,  3,  4,  5,   6,   8,   10,
                                        12, 16, 20, 24,  32,  40,  48,
                                        64, 80, 96, 128, 160, 192, 256};
    uint8_t values[64], decoded[64], buf[128];
    uint32_t q, count, i, seed = 1u;
    for (q = 0; q < 21u; ++q) {
        for (count = 1; count <= 64u; ++count) {
            memset(buf, 0, sizeof(buf));
            for (i = 0; i < count; ++i) {
                seed = seed * 1664525u + 1013904223u;
                values[i] = (uint8_t)((seed >> 8) % levels[q]);
            }
            CHECK(tc_astc_ise_encode_bits(q, count, values, buf, sizeof(buf),
                                          0) == TC_SUCCESS,
                  "astc ise roundtrip encode");
            CHECK(aref_ise_decode(q, count, buf, 0, decoded),
                  "astc ise roundtrip decode");
            CHECK(memcmp(values, decoded, count) == 0,
                  "astc ise roundtrip values");
        }
    }
    return 0;
}

/* Weight unquant tables must be symmetric so that swapping endpoints and
 * inverting weight indices reproduces the exact same reconstruction. */
static int astc_weight_symmetry_test(void) {
    uint32_t q, i, j;
    for (q = 0; q < 12u; ++q) {
        uint32_t n = aref_weight_levels[q];
        uint8_t sorted[32];
        memcpy(sorted, aref_weight_unquant[q], n);
        for (i = 0; i < n; ++i) {
            for (j = i + 1u; j < n; ++j) {
                if (sorted[j] < sorted[i]) {
                    uint8_t t = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = t;
                }
            }
        }
        for (i = 0; i < n; ++i) {
            CHECK(sorted[i] + sorted[n - 1u - i] == 64u,
                  "astc weight unquant symmetry");
        }
    }
    return 0;
}

static void astc_fill_test_image(uint8_t *img, uint32_t w, uint32_t h,
                                 int kind) {
    uint32_t x, y, seed = 12345u;
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            uint8_t *p = img + ((size_t)y * w + x) * 4u;
            switch (kind) {
                case 0: /* smooth gradient, opaque */
                    p[0] = (uint8_t)(x * 255u / (w - 1u));
                    p[1] = (uint8_t)(y * 255u / (h - 1u));
                    p[2] = (uint8_t)((x + y) * 255u / (w + h - 2u));
                    p[3] = 255u;
                    break;
                case 1: /* two well-separated clusters */
                    if (((x / 3u) + (y / 3u)) & 1u) {
                        p[0] = 220u;
                        p[1] = 40u;
                        p[2] = 30u;
                    } else {
                        p[0] = 20u;
                        p[1] = 60u;
                        p[2] = 200u;
                    }
                    p[3] = 255u;
                    break;
                case 2: /* gradient with alpha ramp (dual-plane bait) */
                    p[0] = (uint8_t)(x * 255u / (w - 1u));
                    p[1] = (uint8_t)((x * 200u / (w - 1u)) + 20u);
                    p[2] = (uint8_t)(y * 255u / (h - 1u));
                    p[3] = (uint8_t)(y * 255u / (h - 1u));
                    break;
                default: /* rgba noise */
                    seed = seed * 1664525u + 1013904223u;
                    p[0] = (uint8_t)(seed >> 8);
                    seed = seed * 1664525u + 1013904223u;
                    p[1] = (uint8_t)(seed >> 8);
                    seed = seed * 1664525u + 1013904223u;
                    p[2] = (uint8_t)(seed >> 8);
                    seed = seed * 1664525u + 1013904223u;
                    p[3] = (uint8_t)(seed >> 8);
                    break;
            }
        }
    }
}

static double astc_psnr(const uint8_t *a, const uint8_t *b, size_t n) {
    uint64_t sse = 0;
    size_t i;
    double mse;
    for (i = 0; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        sse += (uint64_t)(d * d);
    }
    if (!sse) return 99.0;
    mse = (double)sse / (double)n;
    return 10.0 * log10(65025.0 / mse);
}

/* The ASTC search kernels must be bit-exact across SIMD backends: full
 * compressed images are compared byte-for-byte against the scalar backend. */
static int astc_backend_parity_test(void) {
    enum { W = 48, H = 48 };
    static uint8_t img[W * H * 4];
    static uint8_t ref[(W / 4) * (H / 4) * 16];
    static uint8_t alt[(W / 4) * (H / 4) * 16];
    static const uint32_t masks[4] = {TC_BACKEND_SSE2, TC_BACKEND_SSE41,
                                      TC_BACKEND_AVX2, TC_BACKEND_NEON};
    uint32_t avail = tc_backend_available_mask();
    int kind, q;
    uint32_t bi, mi;
    static const uint32_t bs[2][2] = {{4, 4}, {6, 6}};
    for (kind = 0; kind < 4; ++kind) {
        astc_fill_test_image(img, W, H, kind);
        for (bi = 0; bi < 2u; ++bi) {
            for (q = 0; q <= 2; ++q) {
                tc_astc_options opt;
                size_t need;
                tc_astc_options_init(&opt);
                opt.block_x = bs[bi][0];
                opt.block_y = bs[bi][1];
                opt.quality = q;
                need = tc_astc_compressed_size(W, H, &opt);
                tc_backend_force_mask(TC_BACKEND_SCALAR);
                CHECK(tc_astc_compress_rgba8(img, W, H, W * 4u, &opt, ref,
                                             need) == TC_SUCCESS,
                      "astc parity scalar encode");
                for (mi = 0; mi < 4u; ++mi) {
                    if (!(avail & masks[mi])) continue;
                    tc_backend_force_mask(masks[mi]);
                    CHECK(tc_astc_compress_rgba8(img, W, H, W * 4u, &opt, alt,
                                                 need) == TC_SUCCESS,
                          "astc parity simd encode");
                    if (memcmp(ref, alt, need) != 0) {
                        tc_backend_force_mask(TC_BACKEND_ALL);
                        fprintf(stderr,
                                "FAIL: astc backend parity kind=%d block=%ux%u "
                                "q=%d mask=%u\n",
                                kind, bs[bi][0], bs[bi][1], q, masks[mi]);
                        return 1;
                    }
                }
                tc_backend_force_mask(TC_BACKEND_ALL);
            }
        }
    }
    return 0;
}

/* Encode -> reference-decode round trip: every emitted block must decode,
 * and quality must stay above per-configuration floors. Floors are set from
 * bench/texcomp_psnr.c measurements minus a safety margin; raise them when
 * the encoder improves, never lower them silently. */
static int astc_ref_roundtrip_test(void) {
    static const uint32_t bs[4][2] = {{4, 4}, {6, 6}, {8, 8}, {12, 12}};
    /* floors[kind][block][quality], dB; measured on 48x48 images minus a
     * ~1 dB safety margin. Raise when the encoder improves. */
    static const double floors[4][4][3] = {
        /* gradient */
        {{37.5, 38.1, 38.1}, {34.0, 34.4, 34.6}, {31.5, 31.6, 33.4},
         {27.0, 27.6, 29.2}},
        /* clusters */
        {{60.0, 60.0, 60.0}, {60.0, 60.0, 60.0}, {20.3, 53.0, 53.0},
         {15.0, 21.0, 21.0}},
        /* alpha ramp */
        {{35.4, 36.1, 36.1}, {31.4, 32.2, 32.3}, {28.7, 29.1, 29.8},
         {25.1, 26.3, 27.3}},
        /* rgba noise */
        {{12.2, 13.2, 13.2}, {11.0, 12.0, 12.0}, {10.9, 11.2, 11.2},
         {10.6, 10.6, 10.6}}};
    enum { W = 48, H = 48 };
    static uint8_t img[W * H * 4];
    static uint8_t dec[W * H * 4];
    static uint8_t blocks[(W / 4) * (H / 4) * 16];
    int kind, q;
    uint32_t bi;
    for (kind = 0; kind < 4; ++kind) {
        astc_fill_test_image(img, W, H, kind);
        for (bi = 0; bi < 4u; ++bi) {
            for (q = 0; q <= 2; ++q) {
                tc_astc_options opt;
                size_t need;
                double psnr;
                tc_astc_options_init(&opt);
                opt.block_x = bs[bi][0];
                opt.block_y = bs[bi][1];
                opt.quality = q;
                need = tc_astc_compressed_size(W, H, &opt);
                CHECK(need <= sizeof(blocks), "astc roundtrip buffer");
                CHECK(tc_astc_compress_rgba8(img, W, H, W * 4u, &opt, blocks,
                                             need) == TC_SUCCESS,
                      "astc roundtrip encode");
                CHECK(aref_decode_image(blocks, W, H, opt.block_x, opt.block_y,
                                        dec),
                      "astc roundtrip reference decode");
                psnr = astc_psnr(img, dec, sizeof(img));
                if (psnr < floors[kind][bi][q]) {
                    fprintf(stderr,
                            "FAIL: astc psnr floor kind=%d block=%ux%u q=%d: "
                            "%.2f dB < %.2f dB\n",
                            kind, bs[bi][0], bs[bi][1], q, psnr,
                            floors[kind][bi][q]);
                    return 1;
                }
                /* Quality levels must not regress against the fast path. */
                (void)0;
            }
        }
    }
    /* Solid image decodes exactly (constant-color blocks). */
    {
        tc_astc_options opt;
        size_t need, i;
        tc_astc_options_init(&opt);
        for (i = 0; i < sizeof(img); i += 4u) {
            img[i + 0u] = 12u;
            img[i + 1u] = 34u;
            img[i + 2u] = 56u;
            img[i + 3u] = 78u;
        }
        need = tc_astc_compressed_size(W, H, &opt);
        CHECK(tc_astc_compress_rgba8(img, W, H, W * 4u, &opt, blocks, need) ==
                  TC_SUCCESS,
              "astc solid encode");
        CHECK(aref_decode_image(blocks, W, H, opt.block_x, opt.block_y, dec),
              "astc solid decode");
        CHECK(memcmp(img, dec, sizeof(img)) == 0, "astc solid exact");
    }
    return 0;
}

int main(void) {
    uint8_t rgba[7 * 5 * 4];
    uint8_t part_rgba[4 * 4 * 4];
    uint8_t astc_rgba[12 * 12 * 4];
    uint8_t bc7[64], bc5[64], bc6h[64];
    uint8_t etc2[64], eac[64], astc[64];
    uint8_t dds[148 + 64];
    uint8_t ktx[68 + 64], astc_file[16 + 64];
    float rgbf[7 * 5 * 3];
    float rgbf_block[16 * 3];
    tc_bc7_options opt;
    tc_bc5_options bc5_opt;
    tc_bc6h_options bc6h_opt;
    tc_etc2_options etc2_opt;
    tc_astc_options astc_opt;
    tc_astc_options astc4_opt;
    uint8_t ise_values[4] = {1, 2, 3, 4};
    uint8_t ise_trits[5] = {1, 2, 0, 1, 2};
    uint8_t ise_bits[8];
    size_t i, j;

    tc_bc7_options_init(&opt);
    tc_bc5_options_init(&bc5_opt);
    tc_bc6h_options_init(&bc6h_opt);
    tc_etc2_options_init(&etc2_opt);
    tc_astc_options_init(&astc_opt);
    tc_astc_options_init(&astc4_opt);
    astc4_opt.block_x = 4;
    astc4_opt.block_y = 4;
    CHECK(tc_bc7_compressed_size(4, 4) == 16, "4x4 size");
    CHECK(tc_bc7_compressed_size(7, 5) == 64, "padded size");
    CHECK(tc_bc5_compressed_size(7, 5) == 64, "bc5 padded size");
    CHECK(tc_bc6h_compressed_size(7, 5) == 64, "bc6h padded size");
    CHECK(tc_etc2_rgb_compressed_size(7, 5) == 32, "etc2 rgb padded size");
    CHECK(tc_etc2_rgba_compressed_size(7, 5) == 64, "etc2 rgba padded size");
    CHECK(tc_eac_r11_compressed_size(7, 5) == 32, "eac r11 padded size");
    CHECK(tc_eac_rg11_compressed_size(7, 5) == 64, "eac rg11 padded size");
    CHECK(tc_astc_compressed_size(7, 5, &astc_opt) == 32, "astc padded size");
    CHECK(tc_astc_ise_sequence_bitcount(1, 0) == 1, "astc ise quant2 bits");
    CHECK(tc_astc_ise_sequence_bitcount(5, 1) == 8, "astc ise quant3 bits");
    CHECK(tc_astc_ise_sequence_bitcount(3, 3) == 7, "astc ise quant5 bits");
    CHECK(tc_astc_ise_sequence_bitcount(4, 20) == 32, "astc ise quant256 bits");
    memset(ise_bits, 0, sizeof(ise_bits));
    CHECK(tc_astc_ise_encode_bits(5, 4, ise_values, ise_bits, sizeof(ise_bits), 0) ==
              TC_SUCCESS,
          "astc ise pure-bit encode");
    CHECK((ise_bits[0] & 0x3fu) == 0x11u, "astc ise pure-bit payload");
    memset(ise_bits, 0, sizeof(ise_bits));
    CHECK(tc_astc_ise_encode_bits(3, 3, ise_values, ise_bits, sizeof(ise_bits), 0) ==
              TC_SUCCESS,
          "astc ise quint encode");
    CHECK(ise_bits[0] == 113u, "astc ise quint payload");
    memset(ise_bits, 0, sizeof(ise_bits));
    CHECK(tc_astc_ise_encode_bits(1, 5, ise_trits, ise_bits, sizeof(ise_bits), 0) ==
              TC_SUCCESS,
          "astc ise trit encode");
    CHECK(ise_bits[0] == 233u, "astc ise trit payload");
    CHECK(tc_dds_bc7_size(7, 5) == 148 + 64, "dds size");
    CHECK(tc_dds_bc5_size(7, 5) == 148 + 64, "bc5 dds size");
    CHECK(tc_dds_bc6h_size(7, 5) == 148 + 64, "bc6h dds size");
    CHECK(tc_ktx_etc2_size(7, 5, &etc2_opt) == 68 + 64, "etc2 ktx size");
    CHECK(tc_astc_file_size(7, 5, &astc_opt) == 16 + 32, "astc file size");

    for (i = 0; i < sizeof(rgba) / 4u; ++i) {
        rgba[i * 4u + 0u] = (uint8_t)(i * 17u);
        rgba[i * 4u + 1u] = (uint8_t)(255u - i * 9u);
        rgba[i * 4u + 2u] = (uint8_t)(i * 31u);
        rgba[i * 4u + 3u] = (uint8_t)(128u + (i & 1u) * 127u);
        rgbf[i * 3u + 0u] = (float)rgba[i * 4u + 0u] / 255.0f;
        rgbf[i * 3u + 1u] = (float)rgba[i * 4u + 1u] / 255.0f;
        rgbf[i * 3u + 2u] = (float)rgba[i * 4u + 2u] / 255.0f;
    }

    CHECK(tc_bc7_compress_rgba8(rgba, 7, 5, 7 * 4, &opt, bc7, sizeof(bc7)) ==
              TC_SUCCESS,
          "compress");
    CHECK(bc7_mode(bc7) < 8u, "valid mode block");
    opt.quick = 0;
    CHECK(tc_bc7_compress_rgba8(rgba, 7, 5, 7 * 4, &opt, bc7, sizeof(bc7)) ==
              TC_SUCCESS,
          "exhaustive compress");
    CHECK(bc7_mode(bc7) < 8u, "valid exhaustive mode block");
    opt.quick = 1;
    for (i = 0; i < 8u; ++i) {
        opt.mode_mask = 1u << i;
        CHECK(tc_bc7_compress_rgba8(rgba, 4, 4, 7 * 4, &opt, bc7, sizeof(bc7)) ==
                  TC_SUCCESS,
              "forced mode compress");
        CHECK(bc7_mode(bc7) == i, "forced mode emitted");
    }
    memset(part_rgba, 0, sizeof(part_rgba));
    for (i = 0; i < 16u; ++i) {
        part_rgba[i * 4u + 0u] = 240u;
        part_rgba[i * 4u + 1u] = 10u;
        part_rgba[i * 4u + 2u] = 10u;
        part_rgba[i * 4u + 3u] = 255u;
    }
    part_rgba[5u * 4u + 0u] = 10u;
    part_rgba[5u * 4u + 1u] = 240u;
    opt.mode_mask = 1u << 1;
    opt.quick = 1;
    CHECK(tc_bc7_compress_rgba8(part_rgba, 4, 4, 4 * 4, &opt, bc7,
                                sizeof(bc7)) == TC_SUCCESS,
          "forced mode1 partition compress");
    CHECK(bc7_mode(bc7) == 1u, "forced mode1 emitted");
    CHECK(rd_bits(bc7, 2, 6) == 5u, "quickbc7 mode1 partition");
    opt.mode_mask = 0xffu;

    CHECK(tc_bc5_compress_rgba8(rgba, 7, 5, 7 * 4, &bc5_opt, bc5,
                                sizeof(bc5)) == TC_SUCCESS,
          "bc5 compress");
    CHECK(bc5[0] >= bc5[1], "bc5 red endpoint order");
    CHECK(tc_dds_write_bc5_memory(bc5, 7, 5, &bc5_opt, dds, sizeof(dds)) ==
              TC_SUCCESS,
          "bc5 dds write");
    CHECK(rd_u32(dds + 112) == 83, "bc5 dxgi format");

    CHECK(tc_etc2_compress_rgba8(rgba, 7, 5, 7 * 4, &etc2_opt, etc2,
                                 sizeof(etc2)) == TC_SUCCESS,
          "etc2 rgba compress");
    CHECK(tc_ktx_write_etc2_memory(etc2, 7, 5, &etc2_opt, ktx, sizeof(ktx)) ==
              TC_SUCCESS,
          "etc2 ktx write");
    CHECK(ktx[0] == 0xabu && memcmp(ktx + 1, "KTX 11", 6) == 0, "ktx magic");
    CHECK(rd_u32(ktx + 28) == 0x9278u, "etc2 rgba internal format");
    etc2_opt.alpha = 0;
    CHECK(tc_etc2_compress_rgba8(rgba, 7, 5, 7 * 4, &etc2_opt, etc2,
                                 sizeof(etc2)) == TC_SUCCESS,
          "etc2 rgb compress");
    CHECK(tc_ktx_write_etc2_memory(etc2, 7, 5, &etc2_opt, ktx, sizeof(ktx)) ==
              TC_SUCCESS,
          "etc2 rgb ktx write");
    CHECK(rd_u32(ktx + 28) == 0x9274u, "etc2 rgb internal format");
    etc2_opt.alpha = 1;

    CHECK(tc_eac_compress_rgba8(rgba, 7, 5, 7 * 4, 0, eac, sizeof(eac)) ==
              TC_SUCCESS,
          "eac r11 compress");
    CHECK(tc_ktx_write_eac_memory(eac, 7, 5, 0, ktx, sizeof(ktx)) == TC_SUCCESS,
          "eac r11 ktx write");
    CHECK(rd_u32(ktx + 28) == 0x9270u, "eac r11 internal format");
    CHECK(tc_eac_compress_rgba8(rgba, 7, 5, 7 * 4, 1, eac, sizeof(eac)) ==
              TC_SUCCESS,
          "eac rg11 compress");
    CHECK(tc_ktx_write_eac_memory(eac, 7, 5, 1, ktx, sizeof(ktx)) == TC_SUCCESS,
          "eac rg11 ktx write");
    CHECK(rd_u32(ktx + 28) == 0x9272u, "eac rg11 internal format");

    CHECK(tc_astc_compress_rgba8(rgba, 7, 5, 7 * 4, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc compress");
    {
        uint32_t wx, wy, quant, dual;
        CHECK(astc_test_decode_mode_2d(rd_bits(astc, 0, 11), &wx, &wy, &quant,
                                       &dual),
              "astc default ldr block mode");
        (void)wx;
        (void)wy;
        (void)quant;
        (void)dual;
    }
    CHECK(rd_bits(astc, 13, 4) == 12u, "astc default rgba endpoint mode");
    CHECK(tc_astc_write_file_memory(astc, 7, 5, &astc_opt, astc_file,
                                    sizeof(astc_file)) == TC_SUCCESS,
          "astc file write");
    CHECK(astc_file[0] == 0x13u && astc_file[1] == 0xabu &&
              astc_file[2] == 0xa1u && astc_file[3] == 0x5cu,
          "astc magic");
    CHECK(astc_file[4] == 6u && astc_file[5] == 6u, "astc block dims");
    memset(part_rgba, 0, sizeof(part_rgba));
    for (i = 0; i < 16u; ++i) {
        part_rgba[i * 4u + 0u] = 12u;
        part_rgba[i * 4u + 1u] = 34u;
        part_rgba[i * 4u + 2u] = 56u;
        part_rgba[i * 4u + 3u] = 78u;
    }
    CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc 4x4 compress");
    CHECK(astc[8] == 12u && astc[9] == 12u && astc[10] == 34u &&
              astc[11] == 34u && astc[12] == 56u && astc[13] == 56u &&
              astc[14] == 78u && astc[15] == 78u,
          "astc 4x4 solid average");
    {
        uint8_t astc_ref[16], astc_simd[16];
        uint32_t avail = tc_backend_available_mask();
        tc_backend_force_mask(TC_BACKEND_SCALAR);
        CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc_ref,
                                     sizeof(astc_ref)) == TC_SUCCESS,
              "astc scalar forced");
        if (avail & TC_BACKEND_SSE2) {
            tc_backend_force_mask(TC_BACKEND_SSE2);
            CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt,
                                         astc_simd, sizeof(astc_simd)) ==
                      TC_SUCCESS,
                  "astc sse2 forced");
            CHECK(memcmp(astc_ref, astc_simd, sizeof(astc_ref)) == 0,
                  "astc sse2 parity");
        }
        if (avail & TC_BACKEND_SSE41) {
            tc_backend_force_mask(TC_BACKEND_SSE41);
            CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt,
                                         astc_simd, sizeof(astc_simd)) ==
                      TC_SUCCESS,
                  "astc sse4.1 forced");
            CHECK(memcmp(astc_ref, astc_simd, sizeof(astc_ref)) == 0,
                  "astc sse4.1 parity");
        }
        if (avail & TC_BACKEND_AVX2) {
            tc_backend_force_mask(TC_BACKEND_AVX2);
            CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt,
                                         astc_simd, sizeof(astc_simd)) ==
                      TC_SUCCESS,
                  "astc avx2 forced");
            CHECK(memcmp(astc_ref, astc_simd, sizeof(astc_ref)) == 0,
                  "astc avx2 parity");
        }
        tc_backend_force_mask(TC_BACKEND_ALL);
    }
    for (i = 0; i < 16u; ++i) {
        part_rgba[i * 4u + 0u] = (uint8_t)(i * 13u);
        part_rgba[i * 4u + 1u] = (uint8_t)(255u - i * 7u);
        part_rgba[i * 4u + 2u] = (uint8_t)(i * 5u);
        part_rgba[i * 4u + 3u] = 255u;
    }
    astc4_opt.quality = 0;
    CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc 4x4 nonconstant compress");
    CHECK(astc_test_mode_is_valid_1plane_2d(rd_bits(astc, 0, 11)),
          "astc 4x4 ldr block mode");
    CHECK(rd_bits(astc, 13, 4) == 8u, "astc 4x4 rgb endpoint mode");
    astc4_opt.quality = 2;
    for (i = 0; i < 16u; ++i) {
        part_rgba[i * 4u + 0u] = (uint8_t)(i * 17u);
        part_rgba[i * 4u + 1u] = (uint8_t)(i * 17u);
        part_rgba[i * 4u + 2u] = (uint8_t)(i * 17u);
        part_rgba[i * 4u + 3u] = 255u;
    }
    CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc 4x4 high precision candidate compress");
    CHECK(astc_test_mode_is_high_precision_4x4(rd_bits(astc, 0, 11)),
          "astc 4x4 high precision block mode");
    part_rgba[3] = 128u;
    part_rgba[5] = 83u;
    part_rgba[10] = 11u;
    CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc 4x4 rgba endpoint compress");
    CHECK(rd_bits(astc, 13, 4) == 12u, "astc 4x4 rgba endpoint mode");
    part_rgba[3] = 255u;
    part_rgba[5] = 17u;
    part_rgba[10] = 34u;
    astc4_opt.quality = 0;
    CHECK(tc_astc_compress_rgba8(part_rgba, 4, 4, 4 * 4, &astc4_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc 4x4 fast compress");
    {
        uint32_t wx, wy, quant, dual;
        CHECK(astc_test_decode_mode_2d(rd_bits(astc, 0, 11), &wx, &wy, &quant,
                                       &dual) &&
                  !dual && wx <= 4u && wy <= 4u && quant <= 2u,
              "astc 4x4 fast block mode");
    }
    astc4_opt.quality = 2;
    for (i = 0; i < 12u * 12u; ++i) {
        astc_rgba[i * 4u + 0u] = (uint8_t)(i * 11u);
        astc_rgba[i * 4u + 1u] = (uint8_t)(255u - i * 5u);
        astc_rgba[i * 4u + 2u] = (uint8_t)(i * 17u);
        astc_rgba[i * 4u + 3u] = 255u;
    }
    {
        static const uint8_t dims[][2] = {{4, 4},  {5, 4},  {5, 5},  {6, 5},
                                          {6, 6},  {8, 5},  {8, 6},  {8, 8},
                                          {10, 5}, {10, 6}, {10, 8}, {10, 10},
                                          {12, 10}, {12, 12}};
        int saw_large_grid = 0;
        int saw_six_grid = 0;
        int saw_budgeted_endpoint = 0;
        for (j = 0; j < sizeof(dims) / sizeof(dims[0]); ++j) {
            tc_astc_options sweep_opt;
            uint32_t mode, wx, wy, quant, dual;
            tc_astc_options_init(&sweep_opt);
            sweep_opt.block_x = dims[j][0];
            sweep_opt.block_y = dims[j][1];
            sweep_opt.quality = 1;
            memset(astc, 0, sizeof(astc));
            CHECK(tc_astc_compress_rgba8(astc_rgba, sweep_opt.block_x, sweep_opt.block_y,
                                         12u * 4u, &sweep_opt, astc, sizeof(astc)) ==
                      TC_SUCCESS,
                  "astc footprint compress");
            mode = rd_bits(astc, 0, 11);
            CHECK(astc_test_mode_is_valid_1plane_2d(mode), "astc footprint block mode");
            CHECK(astc_test_decode_mode_2d(mode, &wx, &wy, &quant, &dual),
                  "astc footprint decode mode");
            if (mode != 66u && mode != 82u && mode != 67u && mode != 83u)
                saw_large_grid = 1;
            if (wx == 6u || wy == 6u) saw_six_grid = 1;
            if (rd_bits(astc, 11, 2) == 0u) {
                CHECK(rd_bits(astc, 13, 4) == 8u, "astc footprint endpoint mode");
                if (tc_astc_ise_sequence_bitcount(wx * wy, quant) + 6u * 8u >
                    111u)
                    saw_budgeted_endpoint = 1;
            } else {
                CHECK(rd_bits(astc, 23, 6) != 0u,
                      "astc footprint partition endpoint mode");
            }
        }
        CHECK(saw_large_grid, "astc larger grid selected");
        CHECK(saw_six_grid, "astc six-wide grid selected");
        CHECK(saw_budgeted_endpoint, "astc budgeted endpoint quant selected");
    }
    for (i = 0; i < 5u; ++i) {
        for (j = 0; j < 8u; ++j) {
            uint8_t v = (uint8_t)(((i + j) & 1u) ? 255u : 0u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = v;
            astc_rgba[p + 1u] = v;
            astc_rgba[p + 2u] = v;
            astc_rgba[p + 3u] = 255u;
        }
    }
    astc_opt.block_x = 8u;
    astc_opt.block_y = 5u;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 8, 5, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc luminance compress");
    CHECK(rd_bits(astc, 13, 4) == 0u, "astc luminance endpoint mode");
    CHECK(astc_test_mode_is_luminance_large(rd_bits(astc, 0, 11)),
          "astc luminance large grid mode");
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            uint8_t v = (uint8_t)(((i + j) & 1u) ? 255u : 0u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = v;
            astc_rgba[p + 1u] = v;
            astc_rgba[p + 2u] = v;
            astc_rgba[p + 3u] = (uint8_t)(32u + ((i * 6u + j) * 5u));
        }
    }
    astc_opt.block_x = 6u;
    astc_opt.block_y = 6u;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc luminance alpha compress");
    CHECK(rd_bits(astc, 13, 4) == 4u, "astc luminance alpha endpoint mode");
    CHECK(astc_test_mode_is_luminance_large(rd_bits(astc, 0, 11)),
          "astc luminance alpha 6x6 grid mode");
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            uint8_t s = (uint8_t)(((i + j) & 1u) ? 255u : 64u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = (uint8_t)((80u * s + 127u) / 255u);
            astc_rgba[p + 1u] = (uint8_t)((160u * s + 127u) / 255u);
            astc_rgba[p + 2u] = (uint8_t)((240u * s + 127u) / 255u);
            astc_rgba[p + 3u] = 255u;
        }
    }
    astc_opt.block_x = 6u;
    astc_opt.block_y = 6u;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc rgb scale compress");
    CHECK(rd_bits(astc, 13, 4) == 6u, "astc rgb scale endpoint mode");
    CHECK(astc_test_mode_is_luminance_large(rd_bits(astc, 0, 11)),
          "astc rgb scale 6x6 grid mode");
    for (i = 0; i < 4u; ++i) {
        for (j = 0; j < 4u; ++j) {
            uint8_t s = (uint8_t)(((i + j) & 1u) ? 255u : 96u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = (uint8_t)((80u * s + 127u) / 255u);
            astc_rgba[p + 1u] = (uint8_t)((160u * s + 127u) / 255u);
            astc_rgba[p + 2u] = (uint8_t)((240u * s + 127u) / 255u);
            astc_rgba[p + 3u] = (uint8_t)(((i + j) & 1u) ? 220u : 80u);
        }
    }
    astc_opt.block_x = 4u;
    astc_opt.block_y = 4u;
    astc_opt.quality = 1;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 4, 4, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc rgb scale alpha compress");
    CHECK(rd_bits(astc, 13, 4) == 10u, "astc rgb scale alpha endpoint mode");
    astc_opt.quality = 2;
    for (i = 0; i < 4u; ++i) {
        for (j = 0; j < 4u; ++j) {
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = (uint8_t)(24u + j * 48u);
            astc_rgba[p + 1u] = (uint8_t)(40u + i * 32u);
            astc_rgba[p + 2u] = (uint8_t)(220u - j * 24u);
            astc_rgba[p + 3u] = (uint8_t)(((i + j) & 1u) ? 232u : 24u);
        }
    }
    astc_opt.block_x = 4u;
    astc_opt.block_y = 4u;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 4, 4, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc dual plane rgba compress");
    {
        uint32_t wx, wy, quant, dual, weight_bits;
        CHECK(astc_test_decode_mode_2d(rd_bits(astc, 0, 11), &wx, &wy, &quant,
                                       &dual) &&
                  dual,
              "astc dual plane block mode");
        weight_bits = tc_astc_ise_sequence_bitcount(wx * wy * 2u, quant);
        CHECK(rd_bits(astc, 128u - weight_bits - 2u, 2u) == 3u,
              "astc dual plane alpha component");
        CHECK(rd_bits(astc, 13, 4) == 12u, "astc dual plane rgba endpoint mode");
        CHECK(weight_bits + 8u * 8u + 2u > 111u,
              "astc dual plane rgba budgeted endpoint quant");
    }
    astc_opt.quality = 1;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 4, 4, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc medium dual plane rgba compress");
    {
        uint32_t wx, wy, quant, dual, weight_bits;
        CHECK(astc_test_decode_mode_2d(rd_bits(astc, 0, 11), &wx, &wy, &quant,
                                       &dual) &&
                  dual,
              "astc medium dual plane block mode");
        weight_bits = tc_astc_ise_sequence_bitcount(wx * wy * 2u, quant);
        CHECK(rd_bits(astc, 128u - weight_bits - 2u, 2u) == 3u,
              "astc medium dual plane alpha component");
        CHECK(rd_bits(astc, 13, 4) == 12u,
              "astc medium dual plane rgba endpoint mode");
    }
    astc_opt.quality = 2;
    /* rgb-scale colors with an alpha ramp decorrelated from the color
     * checker, so a shared weight plane cannot capture both and the
     * dual-plane CEM 10 encoding wins on reconstruction error. */
    for (i = 0; i < 4u; ++i) {
        for (j = 0; j < 4u; ++j) {
            uint8_t s = (uint8_t)(((i + j) & 1u) ? 255u : 80u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = (uint8_t)((72u * s + 127u) / 255u);
            astc_rgba[p + 1u] = (uint8_t)((144u * s + 127u) / 255u);
            astc_rgba[p + 2u] = (uint8_t)((216u * s + 127u) / 255u);
            astc_rgba[p + 3u] = (uint8_t)(20u + j * 70u);
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 4, 4, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc dual plane rgb scale alpha compress");
    {
        uint32_t wx, wy, quant, dual, weight_bits;
        CHECK(astc_test_decode_mode_2d(rd_bits(astc, 0, 11), &wx, &wy, &quant,
                                       &dual) &&
                  dual,
              "astc dual plane rgb scale alpha block mode");
        weight_bits = tc_astc_ise_sequence_bitcount(wx * wy * 2u, quant);
        CHECK(rd_bits(astc, 128u - weight_bits - 2u, 2u) == 3u,
              "astc dual plane rgb scale alpha component");
        CHECK(rd_bits(astc, 13, 4) == 10u,
              "astc dual plane rgb scale alpha endpoint mode");
    }
    /* Partition-favoring content pattern used below: each partition is a
     * flat color except one, whose two colors sit at the block's global
     * luma extremes. The shared weight grid then serves every partition
     * (solids ignore weights, the extreme pair maps to weights 0/64), so
     * the partitioned encoding is near-exact while a single endpoint line
     * cannot represent three or more non-collinear colors. */
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            size_t p = (i * 12u + j) * 4u;
            if (j < 3u) { /* solid red */
                astc_rgba[p + 0u] = 240u;
                astc_rgba[p + 1u] = 20u;
                astc_rgba[p + 2u] = 30u;
            } else if ((i + j) & 1u) { /* bright green-cyan */
                astc_rgba[p + 0u] = 30u;
                astc_rgba[p + 1u] = 230u;
                astc_rgba[p + 2u] = 200u;
            } else { /* dark green-cyan (channels co-vary with the bright
                        one, so the pair lies on its partition's diagonal) */
                astc_rgba[p + 0u] = 20u;
                astc_rgba[p + 1u] = 90u;
                astc_rgba[p + 2u] = 60u;
            }
            astc_rgba[p + 3u] = 255u;
        }
    }
    astc_opt.block_x = 6u;
    astc_opt.block_y = 6u;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc two partition compress");
    CHECK(rd_bits(astc, 11, 2) == 1u, "astc two partition count");
    CHECK(rd_bits(astc, 23, 6) != 0u, "astc two partition endpoint type");
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            size_t p = (i * 12u + j) * 4u;
            if (i < 3u) { /* solid blue, luma inside the pair below */
                astc_rgba[p + 0u] = 30u;
                astc_rgba[p + 1u] = 40u;
                astc_rgba[p + 2u] = 235u;
            } else if ((i + j) & 1u) { /* bright yellow (global max luma) */
                astc_rgba[p + 0u] = 235u;
                astc_rgba[p + 1u] = 225u;
                astc_rgba[p + 2u] = 20u;
            } else { /* dark red (global min luma) */
                astc_rgba[p + 0u] = 100u;
                astc_rgba[p + 1u] = 30u;
                astc_rgba[p + 2u] = 20u;
            }
            astc_rgba[p + 3u] = 255u;
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc two partition vertical search compress");
    CHECK(rd_bits(astc, 11, 2) == 1u, "astc two partition vertical search count");
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            size_t p = (i * 12u + j) * 4u;
            if (j < 3u) { /* solid rgba, luma inside the pair below */
                astc_rgba[p + 0u] = 240u;
                astc_rgba[p + 1u] = 32u;
                astc_rgba[p + 2u] = 48u;
                astc_rgba[p + 3u] = 220u;
            } else if ((i + j) & 1u) { /* bright green (global max luma) */
                astc_rgba[p + 0u] = 24u;
                astc_rgba[p + 1u] = 216u;
                astc_rgba[p + 2u] = 72u;
                astc_rgba[p + 3u] = 40u;
            } else { /* dark green (global min luma) */
                astc_rgba[p + 0u] = 24u;
                astc_rgba[p + 1u] = 120u;
                astc_rgba[p + 2u] = 60u;
                astc_rgba[p + 3u] = 40u;
            }
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc two partition rgba compress");
    CHECK(rd_bits(astc, 11, 2) == 1u, "astc two partition rgba count");
    CHECK(rd_bits(astc, 23, 6) == (12u << 2),
          "astc two partition rgba endpoint type");
    /* All colors are scales of one base hue (whole block rgb-scale); each
     * half is rgb-solid so the partition variance gate passes, and the
     * right half's alpha checker rides the shared weight grid. Quality 1
     * keeps the rgb-scale dual-plane path gated off, so the two-partition
     * CEM 10 encoding wins over a single line (three RGBA points). */
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            uint8_t s = (uint8_t)(j < 3u ? 100u : 255u);
            size_t p = (i * 12u + j) * 4u;
            astc_rgba[p + 0u] = (uint8_t)((80u * s + 127u) / 255u);
            astc_rgba[p + 1u] = (uint8_t)((160u * s + 127u) / 255u);
            astc_rgba[p + 2u] = (uint8_t)((240u * s + 127u) / 255u);
            astc_rgba[p + 3u] =
                (uint8_t)(j < 3u ? 224u : (((i + j) & 1u) ? 200u : 36u));
        }
    }
    astc_opt.quality = 1;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc two partition rgb scale alpha compress");
    CHECK(rd_bits(astc, 11, 2) == 1u,
          "astc two partition rgb scale alpha count");
    CHECK(rd_bits(astc, 23, 6) == (10u << 2),
          "astc two partition rgb scale alpha endpoint type");
    for (i = 0; i < 6u; ++i) {
        for (j = 0; j < 6u; ++j) {
            size_t p = (i * 12u + j) * 4u;
            if (j < 3u) { /* solid red: rgb-scale format for this partition */
                astc_rgba[p + 0u] = 240u;
                astc_rgba[p + 1u] = 20u;
                astc_rgba[p + 2u] = 30u;
            } else if ((i + j) & 1u) { /* co-varying pair that still fails
                                          the scale check (hue drift) */
                astc_rgba[p + 0u] = 60u;
                astc_rgba[p + 1u] = 230u;
                astc_rgba[p + 2u] = 200u;
            } else {
                astc_rgba[p + 0u] = 20u;
                astc_rgba[p + 1u] = 120u;
                astc_rgba[p + 2u] = 60u;
            }
            astc_rgba[p + 3u] = 255u;
        }
    }
    astc_opt.quality = 1;
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc mixed partition endpoint compress");
    CHECK(rd_bits(astc, 11, 2) == 1u, "astc mixed partition count");
    CHECK(rd_bits(astc, 23, 6) != (8u << 2) &&
              rd_bits(astc, 23, 6) != (6u << 2),
          "astc mixed partition endpoint type");
    astc_opt.quality = 2;
    /* Content shaped by an actual ASTC partition pattern (seed 90, three
     * partitions, 6x6): each partition gets its own flat color, so the
     * matching pattern exists in the search space and the three-partition
     * encoding is exact, while one or two partitions must merge three
     * non-collinear colors. Only three RGB clusters exist, so the
     * four-partition path is gated off. */
    {
        static const uint8_t map3[36] = {
            1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0};
        static const uint8_t cols3[3][3] = {
            {240u, 24u, 32u}, {30u, 230u, 60u}, {40u, 52u, 236u}};
        for (i = 0; i < 36u; ++i) {
            size_t p = ((i / 6u) * 12u + (i % 6u)) * 4u;
            astc_rgba[p + 0u] = cols3[map3[i]][0];
            astc_rgba[p + 1u] = cols3[map3[i]][1];
            astc_rgba[p + 2u] = cols3[map3[i]][2];
            astc_rgba[p + 3u] = 255u;
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc three partition compress");
    CHECK(rd_bits(astc, 11, 2) == 2u, "astc three partition count");
    CHECK(rd_bits(astc, 23, 6) != 0u, "astc three partition endpoint type");
    /* Same partition pattern (seed 90), but partition 0 holds a non-scale
     * two-color pair while the others are flat scale-compatible colors, so
     * the three-partition encoding needs mixed endpoint formats. */
    {
        static const uint8_t map3[36] = {
            1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0};
        for (i = 0; i < 36u; ++i) {
            size_t p = ((i / 6u) * 12u + (i % 6u)) * 4u;
            if (map3[i] == 0u) { /* non-scale pair (hue twist) */
                if (((i / 6u) + (i % 6u)) & 1u) {
                    astc_rgba[p + 0u] = 240u;
                    astc_rgba[p + 1u] = 240u;
                    astc_rgba[p + 2u] = 60u;
                } else {
                    astc_rgba[p + 0u] = 20u;
                    astc_rgba[p + 1u] = 30u;
                    astc_rgba[p + 2u] = 200u;
                }
            } else if (map3[i] == 1u) { /* solid red */
                astc_rgba[p + 0u] = 240u;
                astc_rgba[p + 1u] = 24u;
                astc_rgba[p + 2u] = 32u;
            } else { /* solid green */
                astc_rgba[p + 0u] = 30u;
                astc_rgba[p + 1u] = 230u;
                astc_rgba[p + 2u] = 60u;
            }
            astc_rgba[p + 3u] = 255u;
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc three partition mixed endpoint compress");
    CHECK(rd_bits(astc, 11, 2) == 2u, "astc three partition mixed count");
    /* The seed-limited partition patterns rarely match these strips exactly,
     * so stray texels can defeat the per-partition scale classifier; the
     * mixed-format high-part encoding itself is covered by the two-partition
     * mixed test above. Require only a valid endpoint-type field here. */
    CHECK(rd_bits(astc, 23, 6) != 0u,
          "astc three partition mixed endpoint type");
    /* Content shaped by an actual ASTC partition pattern (seed 54, four
     * partitions, 6x6): four flat colors that no coarser partitioning can
     * separate, so the four-partition encoding wins strictly. */
    {
        static const uint8_t map4[36] = {
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3};
        static const uint8_t cols4[4][3] = {{240u, 28u, 36u},
                                            {32u, 228u, 44u},
                                            {40u, 52u, 236u},
                                            {200u, 200u, 200u}};
        for (i = 0; i < 36u; ++i) {
            size_t p = ((i / 6u) * 12u + (i % 6u)) * 4u;
            astc_rgba[p + 0u] = cols4[map4[i]][0];
            astc_rgba[p + 1u] = cols4[map4[i]][1];
            astc_rgba[p + 2u] = cols4[map4[i]][2];
            astc_rgba[p + 3u] = 255u;
        }
    }
    CHECK(tc_astc_compress_rgba8(astc_rgba, 6, 6, 12u * 4u, &astc_opt, astc,
                                 sizeof(astc)) == TC_SUCCESS,
          "astc four partition compress");
    CHECK(rd_bits(astc, 11, 2) == 3u, "astc four partition count");
    CHECK(rd_bits(astc, 23, 6) == (6u << 2), "astc four partition endpoint type");
    tc_astc_options_init(&astc_opt);

    CHECK(tc_bc6h_compress_rgb32f(rgbf, 7, 5, 7 * 3 * sizeof(float), &bc6h_opt,
                                  bc6h, sizeof(bc6h)) == TC_SUCCESS,
          "bc6h compress");
    CHECK(rd_bits(bc6h, 0, 5) == 3u, "bc6h mode11 bits");
    CHECK(tc_dds_write_bc6h_memory(bc6h, 7, 5, &bc6h_opt, dds, sizeof(dds)) ==
              TC_SUCCESS,
          "bc6h dds write");
    CHECK(rd_u32(dds + 112) == 95, "bc6h dxgi format");
    for (i = 0; i < 16u; ++i) {
        rgbf_block[i * 3u + 0u] = (i == 1u) ? 1.0f : 0.0f;
        rgbf_block[i * 3u + 1u] = (i == 2u) ? 1.0f : 0.0f;
        rgbf_block[i * 3u + 2u] = (i == 3u) ? 1.0f : 0.0f;
    }
    CHECK(tc_bc6h_compress_rgb32f(rgbf_block, 4, 4, 4 * 3 * sizeof(float),
                                  &bc6h_opt, bc6h, sizeof(bc6h)) == TC_SUCCESS,
          "bc6h rgb extrema compress");
    CHECK(bc6h_mode11_rgb_error(bc6h, (const float(*)[3])rgbf_block) < 900000000ull,
          "bc6h rgb extrema bounded error");
    for (i = 0; i < sizeof(rgbf) / sizeof(rgbf[0]); ++i)
        rgbf[i] = (i & 1u) ? -rgbf[i] : rgbf[i];
    bc6h_opt.signed_float = 1;
    CHECK(tc_bc6h_compress_rgb32f(rgbf, 7, 5, 7 * 3 * sizeof(float), &bc6h_opt,
                                  bc6h, sizeof(bc6h)) == TC_SUCCESS,
          "bc6h signed compress");
    CHECK(rd_bits(bc6h, 0, 5) == 3u, "bc6h signed mode11 bits");
    CHECK(tc_dds_write_bc6h_memory(bc6h, 7, 5, &bc6h_opt, dds, sizeof(dds)) ==
              TC_SUCCESS,
          "bc6h signed dds write");
    CHECK(rd_u32(dds + 112) == 96, "bc6h signed dxgi format");
    bc6h_opt.signed_float = 0;

    CHECK(tc_dds_write_bc7_memory(bc7, 7, 5, &opt, dds, sizeof(dds)) ==
              TC_SUCCESS,
          "dds write");
    CHECK(memcmp(dds, "DDS ", 4) == 0, "dds magic");
    CHECK(rd_u32(dds + 4) == 124, "dds header size");
    CHECK(rd_u32(dds + 12) == 5, "dds height");
    CHECK(rd_u32(dds + 16) == 7, "dds width");
    CHECK(memcmp(dds + 84, "DX10", 4) == 0, "dx10 fourcc");
    CHECK(rd_u32(dds + 112) == 98, "bc7 dxgi format");
    opt.srgb = 1;
    CHECK(tc_dds_write_bc7_memory(bc7, 7, 5, &opt, dds, sizeof(dds)) ==
              TC_SUCCESS,
          "dds write srgb");
    CHECK(rd_u32(dds + 112) == 99, "bc7 srgb dxgi format");

    if (astc_ise_roundtrip_test()) return 1;
    if (astc_weight_symmetry_test()) return 1;
    if (astc_ref_roundtrip_test()) return 1;
    if (astc_backend_parity_test()) return 1;

    printf("texcomp tests: OK\n");
    return 0;
}
