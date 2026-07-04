/*
 * TinyEXR envmap CLI - environment-map projection conversion (and, in Phase B,
 * SH / spherical-gaussian fitting). The only translation unit that does file
 * I/O; the library core stays free of <stdio.h>.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "envmap.h"
#include "exr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ EXR I/O */

static int channel_index(const exr_part *part, const char *name) {
    int32_t i;
    for (i = 0; i < part->header.num_channels; ++i)
        if (strcmp(part->header.channels[i].name, name) == 0) return (int)i;
    return -1;
}

static float read_sample(const exr_part *part, int ch, size_t idx) {
    exr_pixel_type t = part->header.channels[ch].pixel_type;
    if (t == EXR_PIXEL_UINT) return (float)((const uint32_t *)part->images[ch])[idx];
    if (t == EXR_PIXEL_HALF) {
        float f;
        exr_half_to_float(&((const uint16_t *)part->images[ch])[idx], &f, 1);
        return f;
    }
    return ((const float *)part->images[ch])[idx];
}

/* Load an EXR into an interleaved-RGB em_image of projection `proj`. */
static int load_exr(const char *path, em_proj proj, em_image *out) {
    exr_image img;
    exr_part *part;
    int r, g, b;
    size_t i, n;
    memset(&img, 0, sizeof(img));
    if (!EXR_OK(exr_load_from_file(path, NULL, &img))) return 0;
    part = &img.parts[0];
    if (img.num_parts < 1 || part->is_deep || !part->images ||
        part->width <= 0 || part->height <= 0) {
        exr_image_free(&img);
        return 0;
    }
    r = channel_index(part, "R");
    g = channel_index(part, "G");
    b = channel_index(part, "B");
    if (r < 0 || g < 0 || b < 0) { exr_image_free(&img); return 0; }
    if (!EM_OK(em_image_alloc(NULL, out, proj, part->width, part->height, 3))) {
        exr_image_free(&img);
        return 0;
    }
    n = (size_t)part->width * (size_t)part->height;
    for (i = 0; i < n; ++i) {
        out->data[i * 3 + 0] = read_sample(part, r, i);
        out->data[i * 3 + 1] = read_sample(part, g, i);
        out->data[i * 3 + 2] = read_sample(part, b, i);
    }
    exr_image_free(&img);
    return 1;
}

/* Save one face of an em_image (or the whole image for non-cube) as RGB EXR. */
static int save_exr_face(const char *path, const em_image *img, int face) {
    exr_image out_img;
    exr_part out_part;
    exr_channel ch[3];
    void *planes[3];
    float *R, *G, *B;
    size_t i, n = (size_t)img->width * (size_t)img->height;
    int ok;
    R = (float *)malloc(n * sizeof(float));
    G = (float *)malloc(n * sizeof(float));
    B = (float *)malloc(n * sizeof(float));
    if (!R || !G || !B) { free(R); free(G); free(B); return 0; }
    for (i = 0; i < n; ++i) {
        const float *t = img->data + ((size_t)face * n + i) * img->channels;
        R[i] = t[0]; G[i] = t[1]; B[i] = t[2];
    }
    memset(&out_img, 0, sizeof(out_img));
    memset(&out_part, 0, sizeof(out_part));
    memset(ch, 0, sizeof(ch));
    /* EXR channels are stored alphabetically; B,G,R is conventional. */
    strcpy(ch[0].name, "B"); strcpy(ch[1].name, "G"); strcpy(ch[2].name, "R");
    planes[0] = B; planes[1] = G; planes[2] = R;
    for (i = 0; i < 3; ++i) {
        ch[i].pixel_type = EXR_PIXEL_FLOAT;
        ch[i].x_sampling = 1;
        ch[i].y_sampling = 1;
    }
    out_img.num_parts = 1;
    out_img.parts = &out_part;
    out_part.header.num_channels = 3;
    out_part.header.channels = ch;
    out_part.images = planes;
    out_part.header.data_window.min_x = 0;
    out_part.header.data_window.min_y = 0;
    out_part.header.data_window.max_x = img->width - 1;
    out_part.header.data_window.max_y = img->height - 1;
    out_part.header.display_window = out_part.header.data_window;
    out_part.width = img->width;
    out_part.height = img->height;
    out_part.header.compression = EXR_COMPRESSION_ZIP;
    ok = EXR_OK(exr_save_to_file(path, &out_img, EXR_COMPRESSION_ZIP));
    free(R); free(G); free(B);
    return ok;
}

static const char *cube_face_suffix[6] = {"_px", "_nx", "_py", "_ny", "_pz", "_nz"};

/* Save an em_image: one file (equirect/octa) or 6 face files (cube). */
static int save_image(const char *out, const em_image *img) {
    if (img->proj != EM_PROJ_CUBE) return save_exr_face(out, img, 0);
    {
        size_t base = strlen(out);
        const char *dot = strrchr(out, '.');
        size_t stem = dot ? (size_t)(dot - out) : base;
        int f, ok = 1;
        for (f = 0; f < 6; ++f) {
            char path[1024];
            if (stem + 8 >= sizeof(path)) return 0;
            memcpy(path, out, stem);
            memcpy(path + stem, cube_face_suffix[f], 3);
            strcpy(path + stem + 3, ".exr");
            if (!save_exr_face(path, img, f)) ok = 0;
            else printf("  wrote %s\n", path);
        }
        return ok;
    }
}

static int parse_proj(const char *s, em_proj *p) {
    if (!strcmp(s, "equirect") || !strcmp(s, "latlong")) *p = EM_PROJ_EQUIRECT;
    else if (!strcmp(s, "cube")) *p = EM_PROJ_CUBE;
    else if (!strcmp(s, "octa") || !strcmp(s, "octahedral")) *p = EM_PROJ_OCTA;
    else return 0;
    return 1;
}

static void usage(void) {
    printf(
        "envmap - environment-map projection / SH / spherical-gaussian tool\n"
        "usage:\n"
        "  envmap convert -i in.exr [--from equirect|cube|octa] --to equirect|cube|octa\n"
        "                 [--size N] -o out.exr\n"
        "    cube output writes 6 face EXRs out_{px,nx,py,ny,pz,nz}.exr\n"
        "  envmap sh -i env.exr [--from P] [--order 0..4] [--window W] [--recon-size N] -o env.sh\n"
        "  envmap sg -i env.exr [--from P] [--lobes K] [--asg] [--recon-size N] -o env.sg\n"
        "    sh/sg also write <out>_recon.exr (equirect reconstruction)\n");
}

static int cmd_convert(int argc, char **argv) {
    const char *in = NULL, *out = NULL;
    em_proj from = EM_PROJ_EQUIRECT, to = EM_PROJ_EQUIRECT;
    int size = 0, i, have_to = 0;
    em_image src, dst;
    for (i = 0; i < argc; ++i) {
        const char *a = argv[i];
#define NEXT() (++i < argc ? argv[i] : NULL)
        if (!strcmp(a, "-i")) in = NEXT();
        else if (!strcmp(a, "-o")) out = NEXT();
        else if (!strcmp(a, "--from")) { const char *v = NEXT(); if (!v || !parse_proj(v, &from)) { usage(); return 2; } }
        else if (!strcmp(a, "--to")) { const char *v = NEXT(); if (!v || !parse_proj(v, &to)) { usage(); return 2; } have_to = 1; }
        else if (!strcmp(a, "--size")) { const char *v = NEXT(); if (!v) { usage(); return 2; } size = atoi(v); }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(); return 2; }
#undef NEXT
    }
    if (!in || !out || !have_to) { usage(); return 2; }
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    if (!load_exr(in, from, &src)) { fprintf(stderr, "failed to load %s\n", in); return 1; }
    if (size <= 0) {
        /* sensible default: keep detail relative to the source */
        size = (from == EM_PROJ_EQUIRECT) ? src.width / 4 : src.width;
        if (size < 1) size = 1;
    }
    if (!EM_OK(em_convert(NULL, &src, to, size, &dst))) {
        fprintf(stderr, "conversion failed\n");
        em_image_free(NULL, &src);
        return 1;
    }
    if (!save_image(out, &dst)) { fprintf(stderr, "failed to write %s\n", out); }
    else printf("wrote %s (%s %dx%d%s)\n", out,
                to == EM_PROJ_CUBE ? "cube" : (to == EM_PROJ_OCTA ? "octa" : "equirect"),
                dst.width, dst.height, to == EM_PROJ_CUBE ? " x6" : "");
    em_image_free(NULL, &src);
    em_image_free(NULL, &dst);
    return 0;
}

/* Build "<stem>_recon.exr" from an output path. */
static void recon_path(const char *out, char *buf, size_t buf_n) {
    const char *dot = strrchr(out, '.');
    size_t stem = dot ? (size_t)(dot - out) : strlen(out);
    if (stem + 11 >= buf_n) stem = buf_n - 12;
    memcpy(buf, out, stem);
    strcpy(buf + stem, "_recon.exr");
}

/* Reconstruct an equirect env from an eval callback and save it. */
typedef void (*recon_fn)(const float dir[3], float rgb[3], void *user);
static int save_recon(const char *path, int size, recon_fn fn, void *user) {
    em_image img;
    int x, y, ok;
    if (!EM_OK(em_image_alloc(NULL, &img, EM_PROJ_EQUIRECT, size, size / 2, 3)))
        return 0;
    for (y = 0; y < img.height; ++y)
        for (x = 0; x < img.width; ++x) {
            float u = (x + 0.5f) / img.width, v = (y + 0.5f) / img.height, d[3];
            float *t = em_image_texel(&img, 0, x, y);
            em_uv_to_dir(EM_PROJ_EQUIRECT, 0, u, v, d);
            fn(d, t, user);
        }
    ok = save_exr_face(path, &img, 0);
    em_image_free(NULL, &img);
    return ok;
}

typedef struct { int order; const float *coeffs; } sh_recon_ctx;
static void sh_recon_cb(const float d[3], float rgb[3], void *u) {
    sh_recon_ctx *c = (sh_recon_ctx *)u;
    em_sh_eval(c->order, c->coeffs, d, rgb);
}
typedef struct { const em_sg_lobe *lobes; int n; } sg_recon_ctx;
static void sg_recon_cb(const float d[3], float rgb[3], void *u) {
    sg_recon_ctx *c = (sg_recon_ctx *)u;
    em_sg_eval(c->lobes, c->n, d, rgb);
}

static int cmd_sh(int argc, char **argv) {
    const char *in = NULL, *out = NULL;
    em_proj from = EM_PROJ_EQUIRECT;
    int order = 2, recon = 256, i, ncoeff;
    float window = 0.0f;
    em_image src;
    float *coeffs;
    FILE *f;
    for (i = 0; i < argc; ++i) {
        const char *a = argv[i];
#define NEXT() (++i < argc ? argv[i] : NULL)
        if (!strcmp(a, "-i")) in = NEXT();
        else if (!strcmp(a, "-o")) out = NEXT();
        else if (!strcmp(a, "--from")) { const char *v = NEXT(); if (!v || !parse_proj(v, &from)) { usage(); return 2; } }
        else if (!strcmp(a, "--order")) { const char *v = NEXT(); if (!v) { usage(); return 2; } order = atoi(v); }
        else if (!strcmp(a, "--window")) { const char *v = NEXT(); if (!v) { usage(); return 2; } window = (float)atof(v); }
        else if (!strcmp(a, "--recon-size")) { const char *v = NEXT(); if (!v) { usage(); return 2; } recon = atoi(v); }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(); return 2; }
#undef NEXT
    }
    if (!in || !out || order < 0 || order > EM_SH_MAX_ORDER) { usage(); return 2; }
    memset(&src, 0, sizeof(src));
    if (!load_exr(in, from, &src)) { fprintf(stderr, "failed to load %s\n", in); return 1; }
    ncoeff = em_sh_num_coeffs(order);
    coeffs = (float *)malloc((size_t)ncoeff * 3 * sizeof(float));
    if (!coeffs) { em_image_free(NULL, &src); return 1; }
    em_sh_project(&src, order, coeffs);
    if (window > 0.0f) em_sh_window(order, coeffs, window);
    f = fopen(out, "w");
    if (f) {
        fprintf(f, "# tinyexr envmap SH  order=%d  coeffs=%d  (R G B)\n", order, ncoeff);
        fprintf(f, "order %d\n", order);
        for (i = 0; i < ncoeff; ++i)
            fprintf(f, "%.8g %.8g %.8g\n", coeffs[i*3+0], coeffs[i*3+1], coeffs[i*3+2]);
        fclose(f);
        printf("wrote %s (SH order %d, %d coeffs, DC=%.4f)\n", out, order, ncoeff, coeffs[0]);
    } else fprintf(stderr, "failed to write %s\n", out);
    {
        char rp[1024];
        sh_recon_ctx ctx; ctx.order = order; ctx.coeffs = coeffs;
        recon_path(out, rp, sizeof(rp));
        if (save_recon(rp, recon, sh_recon_cb, &ctx)) printf("wrote %s (reconstruction)\n", rp);
    }
    free(coeffs);
    em_image_free(NULL, &src);
    return 0;
}

static int cmd_sg(int argc, char **argv) {
    const char *in = NULL, *out = NULL;
    em_proj from = EM_PROJ_EQUIRECT;
    int lobes = 16, asg = 0, recon = 256, i;
    em_image src;
    em_sg_lobe *L;
    FILE *f;
    for (i = 0; i < argc; ++i) {
        const char *a = argv[i];
#define NEXT() (++i < argc ? argv[i] : NULL)
        if (!strcmp(a, "-i")) in = NEXT();
        else if (!strcmp(a, "-o")) out = NEXT();
        else if (!strcmp(a, "--from")) { const char *v = NEXT(); if (!v || !parse_proj(v, &from)) { usage(); return 2; } }
        else if (!strcmp(a, "--lobes")) { const char *v = NEXT(); if (!v) { usage(); return 2; } lobes = atoi(v); }
        else if (!strcmp(a, "--asg")) asg = 1;
        else if (!strcmp(a, "--recon-size")) { const char *v = NEXT(); if (!v) { usage(); return 2; } recon = atoi(v); }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(); return 2; }
#undef NEXT
    }
    if (!in || !out || lobes < 1 || lobes > EM_SG_MAX_LOBES) { usage(); return 2; }
    memset(&src, 0, sizeof(src));
    if (!load_exr(in, from, &src)) { fprintf(stderr, "failed to load %s\n", in); return 1; }
    L = (em_sg_lobe *)malloc((size_t)lobes * sizeof(em_sg_lobe));
    if (!L) { em_image_free(NULL, &src); return 1; }
    if (!EM_OK(em_sg_fit(NULL, &src, lobes, asg, L))) {
        fprintf(stderr, "sg fit failed\n"); free(L); em_image_free(NULL, &src); return 1;
    }
    f = fopen(out, "w");
    if (f) {
        fprintf(f, "# tinyexr envmap SG  lobes=%d  (axis.xyz sharpness amp.rgb)\n", lobes);
        fprintf(f, "lobes %d\n", lobes);
        for (i = 0; i < lobes; ++i)
            fprintf(f, "%.6g %.6g %.6g  %.6g  %.6g %.6g %.6g\n",
                    L[i].axis[0], L[i].axis[1], L[i].axis[2], L[i].sharpness,
                    L[i].amplitude[0], L[i].amplitude[1], L[i].amplitude[2]);
        fclose(f);
        printf("wrote %s (SG %d lobes, sharpness=%.2f)\n", out, lobes, L[0].sharpness);
    } else fprintf(stderr, "failed to write %s\n", out);
    {
        char rp[1024];
        sg_recon_ctx ctx; ctx.lobes = L; ctx.n = lobes;
        recon_path(out, rp, sizeof(rp));
        if (save_recon(rp, recon, sg_recon_cb, &ctx)) printf("wrote %s (reconstruction)\n", rp);
    }
    free(L);
    em_image_free(NULL, &src);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    if (!strcmp(argv[1], "convert")) return cmd_convert(argc - 2, argv + 2);
    if (!strcmp(argv[1], "sh")) return cmd_sh(argc - 2, argv + 2);
    if (!strcmp(argv[1], "sg")) return cmd_sg(argc - 2, argv + 2);
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(); return 0; }
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    usage();
    return 2;
}
