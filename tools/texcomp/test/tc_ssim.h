/*
 * TinyEXR texcomp — float RGB SSIM (11×11 Gaussian, σ=1.5)
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TC_SSIM_H_
#define TC_SSIM_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 11×11 Gaussian kernel (σ=1.5), row-major, sum ≈ 1.0. */
static const float tc_ssim_gauss11[121] = {
#include "tc_ssim_gauss11.inc"
};

/* Compute per-channel SSIM between two float RGB images.
 * img_a, img_b: row-major float RGB, w*h*3 elements, values in any range.
 * w, h: image dimensions (must be >= 11 for the Gaussian window to fit).
 * channel_ssim[3]: output per-channel SSIM (R, G, B).
 * Returns mean SSIM across all three channels (or 0 on degenerate input).
 *
 * Standard SSIM parameters: K1=0.01, K2=0.03, L = per-image dynamic range.
 * Boundary: reflect (mirror) padding for windows near edges.
 * Visual difference: SSIM < 0.95 is usually noticeable, < 0.90 is poor. */
static double tc_ssim(const float *img_a, const float *img_b,
                       uint32_t w, uint32_t h,
                       double channel_ssim[3]) {
    static const double C1 = (0.01 * 0.01); /* K1^2 */
    static const double C2 = (0.03 * 0.03); /* K2^2 */
    double total[3] = {0, 0, 0};
    uint32_t c, i, j, count = 0;

    if (!img_a || !img_b || w < 11u || h < 11u) {
        if (channel_ssim) { channel_ssim[0] = channel_ssim[1] = channel_ssim[2] = 0; }
        return 0;
    }

    for (c = 0; c < 3u; ++c) {
        double sum_ssim = 0;
        uint32_t n = 0;

        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                double mu_x = 0, mu_y = 0, sigma_x2 = 0, sigma_y2 = 0, sigma_xy = 0;
                double ssim_val;
                int ky, kx;

                for (ky = -5; ky <= 5; ++ky) {
                    for (kx = -5; kx <= 5; ++kx) {
                        int rx = (int)i + kx;
                        int ry = (int)j + ky;
                        double wgt = tc_ssim_gauss11[(ky + 5) * 11 + (kx + 5)];
                        double va, vb;

                        /* Reflect boundary. */
                        if (rx < 0) rx = -rx - 1;
                        if (rx >= (int)w) rx = 2 * (int)w - rx - 1;
                        if (ry < 0) ry = -ry - 1;
                        if (ry >= (int)h) ry = 2 * (int)h - ry - 1;

                        va = img_a[((size_t)ry * w + (size_t)rx) * 3u + c];
                        vb = img_b[((size_t)ry * w + (size_t)rx) * 3u + c];
                        mu_x += wgt * va;
                        mu_y += wgt * vb;
                    }
                }

                for (ky = -5; ky <= 5; ++ky) {
                    for (kx = -5; kx <= 5; ++kx) {
                        int rx = (int)i + kx;
                        int ry = (int)j + ky;
                        double wgt = tc_ssim_gauss11[(ky + 5) * 11 + (kx + 5)];
                        double va, vb;

                        if (rx < 0) rx = -rx - 1;
                        if (rx >= (int)w) rx = 2 * (int)w - rx - 1;
                        if (ry < 0) ry = -ry - 1;
                        if (ry >= (int)h) ry = 2 * (int)h - ry - 1;

                        va = img_a[((size_t)ry * w + (size_t)rx) * 3u + c];
                        vb = img_b[((size_t)ry * w + (size_t)rx) * 3u + c];
                        sigma_x2 += wgt * (va - mu_x) * (va - mu_x);
                        sigma_y2 += wgt * (vb - mu_y) * (vb - mu_y);
                        sigma_xy += wgt * (va - mu_x) * (vb - mu_y);
                    }
                }

                ssim_val = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
                         / ((mu_x * mu_x + mu_y * mu_y + C1) *
                            (sigma_x2 + sigma_y2 + C2));
                sum_ssim += ssim_val;
                ++n;
            }
        }

        total[c] = (n > 0) ? sum_ssim / (double)n : 0.0;
        if (channel_ssim) channel_ssim[c] = total[c];
        count = n;
    }

    if (count == 0) return 0;
    return (total[0] + total[1] + total[2]) / 3.0;
}

#ifdef __cplusplus
}
#endif

#endif /* TC_SSIM_H_ */
