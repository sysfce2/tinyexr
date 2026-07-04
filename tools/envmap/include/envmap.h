/*
 * TinyEXR envmap - environment-map projections, spherical harmonics and
 * spherical gaussians for image-based lighting.
 *
 * Pure-C11. Projections: equirectangular (Y-up), cube (KTX/D3D/GL face order
 * +X,-X,+Y,-Y,+Z,-Z, matching tools/texpipe/src/texpipe_cube.c), and octahedral
 * (Y-up pole). Reuses tir's allocator; the CLI does HDR EXR I/O.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TINYEXR_ENVMAP_H_
#define TINYEXR_ENVMAP_H_

#include <stddef.h>
#include <stdint.h>

#include "tir.h" /* tir_allocator */

#ifdef __cplusplus
extern "C" {
#endif

#define EM_VERSION_MAJOR 0
#define EM_VERSION_MINOR 1
#define EM_VERSION_PATCH 0

typedef enum em_result {
    EM_SUCCESS = 0,
    EM_ERROR_INVALID_ARGUMENT = -1,
    EM_ERROR_OUT_OF_MEMORY = -2,
    EM_ERROR_UNSUPPORTED = -3,
    EM_ERROR_IO = -4
} em_result;

#define EM_OK(r) ((int)(r) == 0)

const char *em_result_string(em_result r);

typedef enum em_proj {
    EM_PROJ_EQUIRECT = 0, /* width x height, 2:1; Y-up */
    EM_PROJ_CUBE = 1,     /* 6 faces, each face_dim x face_dim */
    EM_PROJ_OCTA = 2      /* size x size octahedral, Y-up pole */
} em_proj;

/* Interleaved float RGB environment image. For CUBE, `faces` == 6 and the data
 * is face-major: face f occupies rows [f*height, (f+1)*height); width==height
 * is the face dimension. For EQUIRECT/OCTA, `faces` == 1. */
typedef struct em_image {
    em_proj proj;
    int width;
    int height;
    int channels; /* 3 */
    int faces;    /* 1, or 6 for cube */
    float *data;  /* faces * width * height * channels floats */
} em_image;

em_result em_image_alloc(const tir_allocator *a, em_image *img, em_proj proj,
                         int width, int height, int channels);
void em_image_free(const tir_allocator *a, em_image *img);

/* Pointer to texel (x,y) of face `face` (face 0 for non-cube). */
float *em_image_texel(const em_image *img, int face, int x, int y);

/* ===========================================================================
 * Projection <-> direction
 * ========================================================================= */

/* Direction (need not be normalized) -> (face, u, v) with u,v in [0,1].
 * face is 0 for equirect/octa, 0..5 for cube. */
void em_dir_to_uv(em_proj proj, const float dir[3], int *face, float *u,
                  float *v);

/* (face, u, v in [0,1]) -> normalized direction. */
void em_uv_to_dir(em_proj proj, int face, float u, float v, float dir[3]);

/* Solid angle (steradians) of texel (x,y) [face `face`] of a proj image of the
 * given dimensions. Exact for equirect and cube; octa uses a local Jacobian
 * estimate. */
float em_texel_solid_angle(em_proj proj, int width, int height, int face, int x,
                           int y);

/* Bilinearly sample `img` along `dir` (normalized or not) into out_rgb.
 * Wrapping: equirect wraps in u / clamps in v; cube clamps within a face; octa
 * uses the octahedral border fold. */
void em_sample(const em_image *img, const float dir[3], float out_rgb[3]);

/* Resample `src` into projection `dst_proj` at `dst_size` (face_dim for cube,
 * width for octa; equirect uses dst_size x dst_size/2). */
em_result em_convert(const tir_allocator *a, const em_image *src,
                     em_proj dst_proj, int dst_size, em_image *out);

/* ===========================================================================
 * Spherical sampling (Phase B helpers)
 * ========================================================================= */

/* Low-discrepancy Hammersley point i of N in [0,1)^2 (van der Corput radical). */
void em_hammersley(uint32_t i, uint32_t n, float out_xy[2]);

/* Cosine-weighted and uniform hemisphere/sphere directions from xi in [0,1)^2
 * about +Z (caller rotates to the shading frame). */
void em_sample_cosine(const float xi[2], float out_dir[3]);
void em_sample_uniform_sphere(const float xi[2], float out_dir[3]);

/* Iterate a source env's texels, invoking cb(dir, radiance[3], solid_angle,
 * user) for each. Sum of solid angles is ~4*pi. */
typedef void (*em_texel_fn)(const float dir[3], const float rgb[3],
                            float solid_angle, void *user);
void em_foreach_texel(const em_image *src, em_texel_fn cb, void *user);

/* ===========================================================================
 * Spherical harmonics (Phase B) — real SH, polar axis = +Y (up), self-consistent
 * project/eval convention.
 * ========================================================================= */

#define EM_SH_MAX_ORDER 4

int em_sh_num_coeffs(int order); /* (order+1)^2 */

/* Evaluate the real-SH basis for `dir` into basis[(order+1)^2]. */
void em_sh_eval_basis(int order, const float dir[3], float *basis);

/* Project a source env onto order-N SH. `coeffs` holds num_coeffs*3 floats,
 * coefficient-major (coeff i channel c at coeffs[i*3+c]). */
em_result em_sh_project(const em_image *src, int order, float *coeffs);

/* Reconstruct radiance from SH coeffs at `dir`. */
void em_sh_eval(int order, const float *coeffs, const float dir[3],
                float rgb[3]);

/* Apply a Hanning window over SH bands (reduces ringing); window_width in
 * bands, <=0 disables. Modifies coeffs in place. */
void em_sh_window(int order, float *coeffs, float window_width);

/* ===========================================================================
 * Spherical gaussians (Phase B)
 * ========================================================================= */

#define EM_SG_MAX_LOBES 64

typedef struct em_sg_lobe {
    float axis[3];      /* unit lobe axis */
    float sharpness;    /* lambda (concentration) */
    float amplitude[3]; /* per-channel amplitude */
} em_sg_lobe;

/* Fit `num_lobes` spherical gaussians (fixed Fibonacci axes + shared sharpness,
 * non-negative least-squares amplitudes) to a source env. `asg` is reserved
 * (isotropic SG today). Writes num_lobes lobes. */
em_result em_sg_fit(const tir_allocator *a, const em_image *src, int num_lobes,
                    int asg, em_sg_lobe *out_lobes);

/* Reconstruct radiance from a lobe set at `dir`. */
void em_sg_eval(const em_sg_lobe *lobes, int num_lobes, const float dir[3],
                float rgb[3]);

#ifdef __cplusplus
}
#endif

#endif /* TINYEXR_ENVMAP_H_ */
