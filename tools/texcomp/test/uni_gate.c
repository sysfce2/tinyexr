/*
 * TinyEXR texcomp - universal ("uni") transcode gate.
 *
 * Verifies: encode/decode fidelity, and that the cheap transcoders emit valid
 * blocks -- uni -> BC7 decoded by the SHIPPED BC7 decoder must match the uni
 * reconstruction (proves the hand-written mode-6 packing is decoder-correct),
 * and uni -> BC1 round-trips at BC1-class quality.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "texcomp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); g_fail = 1; } } while (0)

static double psnr(const uint8_t *a, const uint8_t *b, size_t n, int chans, int stride_ch) {
    double mse = 0; size_t i; int c;
    for (i = 0; i < n; ++i)
        for (c = 0; c < chans; ++c) { double d = (double)a[i*stride_ch+c] - b[i*stride_ch+c]; mse += d*d; }
    mse /= (double)(n * chans);
    if (mse <= 0) return 1e9;
    return 10.0 * log10(255.0*255.0/mse);
}

static uint8_t exp5(int v) { return (uint8_t)((v << 3) | (v >> 2)); }
static uint8_t exp6(int v) { return (uint8_t)((v << 2) | (v >> 4)); }

/* Minimal BC1 decoder (4-colour mode). */
static void bc1_decode(const uint8_t *bc1, uint32_t W, uint32_t H, uint8_t *out) {
    uint32_t bx, by, i;
    size_t off = 0;
    for (by = 0; by < H; by += 4)
        for (bx = 0; bx < W; bx += 4, off += 8) {
            const uint8_t *b = bc1 + off;
            uint16_t c0 = (uint16_t)(b[0] | (b[1] << 8)), c1 = (uint16_t)(b[2] | (b[3] << 8));
            uint8_t pal[4][4];
            uint32_t idx = (uint32_t)(b[4] | (b[5]<<8) | (b[6]<<16) | ((uint32_t)b[7]<<24));
            pal[0][0]=exp5((c0>>11)&31); pal[0][1]=exp6((c0>>5)&63); pal[0][2]=exp5(c0&31); pal[0][3]=255;
            pal[1][0]=exp5((c1>>11)&31); pal[1][1]=exp6((c1>>5)&63); pal[1][2]=exp5(c1&31); pal[1][3]=255;
            for (i = 0; i < 3; ++i) {
                pal[2][i] = (uint8_t)((2*pal[0][i]+pal[1][i])/3);
                pal[3][i] = (uint8_t)((pal[0][i]+2*pal[1][i])/3);
            }
            pal[2][3]=pal[3][3]=255;
            for (i = 0; i < 16; ++i) {
                uint32_t x = bx + (i&3), y = by + (i>>2), sel = (idx >> (i*2)) & 3;
                if (x < W && y < H) memcpy(out + ((size_t)y*W+x)*4, pal[sel], 4);
            }
        }
}

int main(void) {
    const uint32_t W = 128, H = 128;
    uint8_t *img = malloc((size_t)W*H*4), *unid = malloc((size_t)W*H*4);
    uint8_t *bc7d = malloc((size_t)W*H*4), *bc1d = malloc((size_t)W*H*4);
    uint8_t *uni = malloc(tc_uni_compressed_size(W,H));
    uint8_t *bc7 = malloc(tc_bc7_compressed_size(W,H));
    uint8_t *bc1 = malloc(tc_bc1_compressed_size(W,H));
    uint32_t x, y;
    double p_uni, p_tb7, p_bc1;

    for (y = 0; y < H; ++y)
        for (x = 0; x < W; ++x) {
            uint8_t *p = img + ((size_t)y*W+x)*4;
            p[0]=(uint8_t)(x*255/(W-1)); p[1]=(uint8_t)(y*255/(H-1));
            p[2]=(uint8_t)((x+y)*255/(W+H-2)); p[3]=(uint8_t)(255 - x*128/(W-1));
        }

    CHECK(tc_uni_compress_rgba8(img, W, H, W*4, uni, tc_uni_compressed_size(W,H)) == TC_SUCCESS, "uni encode");
    CHECK(tc_uni_decompress_rgba8(uni, W, H, W*4, unid, (size_t)W*H*4) == TC_SUCCESS, "uni decode");
    p_uni = psnr(img, unid, (size_t)W*H, 4, 4);

    /* transcode uni -> BC7, decode with the SHIPPED decoder, compare to the uni
     * reconstruction: the direct mode-6 pack must be near-lossless + valid. */
    CHECK(tc_uni_transcode_bc7(uni, W, H, bc7, tc_bc7_compressed_size(W,H)) == TC_SUCCESS, "transcode bc7");
    CHECK(tc_bc7_decompress_rgba8(bc7, W, H, W*4, bc7d, (size_t)W*H*4) == TC_SUCCESS, "bc7 decode");
    p_tb7 = psnr(unid, bc7d, (size_t)W*H, 4, 4);

    /* transcode uni -> BC1, decode, compare to source (BC1-class quality). */
    CHECK(tc_uni_transcode_bc1(uni, W, H, bc1, tc_bc1_compressed_size(W,H)) == TC_SUCCESS, "transcode bc1");
    bc1_decode(bc1, W, H, bc1d);
    p_bc1 = psnr(img, bc1d, (size_t)W*H, 3, 4); /* BC1 is RGB */

    printf("uni encode/decode PSNR      = %.1f dB\n", p_uni);
    printf("uni->BC7 vs uni recon PSNR  = %.1f dB (transcode fidelity)\n", p_tb7);
    printf("uni->BC1 vs source PSNR     = %.1f dB\n", p_bc1);
    CHECK(p_uni >= 30.0, "uni intermediate reasonable quality");
    CHECK(p_tb7 >= 44.0, "uni->BC7 transcode is near-lossless + decoder-valid");
    CHECK(p_bc1 >= 28.0, "uni->BC1 transcode is BC1-class");

    free(img); free(unid); free(bc7d); free(bc1d); free(uni); free(bc7); free(bc1);
    if (g_fail) { printf("UNI GATE: FAIL\n"); return 1; }
    printf("UNI GATE: OK\n");
    return 0;
}
