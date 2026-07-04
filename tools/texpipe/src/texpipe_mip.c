/*
 * TinyEXR texpipe - content-aware mip-chain generation via tir.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "texpipe_internal.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- pixel decoding */

static float tp_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    union { uint32_t u; float f; } v;
    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            /* subnormal */
            exp = 127u - 15u + 1u;
            while ((man & 0x400u) == 0u) {
                man <<= 1;
                --exp;
            }
            man &= 0x3ffu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }
    v.u = bits;
    return v.f;
}

static size_t tp_type_bytes(tir_pixel_type t) {
    switch (t) {
    case TIR_F32: return 4;
    case TIR_F16: return 2;
    case TIR_U8: return 1;
    case TIR_U16: return 2;
    }
    return 0;
}

static float tp_read_sample(const tir_image_view *v, int x, int y, int c) {
    /* row_stride_bytes == 0 means tightly packed (tir's convention). */
    size_t stride = v->row_stride_bytes
                        ? v->row_stride_bytes
                        : (size_t)v->width * (size_t)v->channels *
                              tp_type_bytes(v->type);
    const uint8_t *row = (const uint8_t *)v->data + (size_t)y * stride;
    size_t idx = (size_t)x * (size_t)v->channels + (size_t)c;
    switch (v->type) {
    case TIR_F32: return ((const float *)row)[idx];
    case TIR_F16: return tp_half_to_float(((const uint16_t *)row)[idx]);
    case TIR_U8: return (float)((const uint8_t *)row)[idx] / 255.0f;
    case TIR_U16: return (float)((const uint16_t *)row)[idx] / 65535.0f;
    }
    return 0.0f;
}

/* ------------------------------------------------------- surface alloc */

static tp_result tp_surface_alloc(const tir_allocator *a, tp_surface *s, int w,
                                  int h, int channels) {
    size_t stride = (size_t)w * (size_t)channels * sizeof(float);
    s->width = w;
    s->height = h;
    s->channels = channels;
    s->stride = stride;
    s->data = (float *)tp_alloc(a, stride * (size_t)h);
    if (!s->data) return TP_ERROR_OUT_OF_MEMORY;
    return TP_SUCCESS;
}

/* Level 0 is the authored image: a straight type-convert, no resample. */
static void tp_copy_base(const tir_image_view *src, tp_surface *dst) {
    int x, y, c;
    for (y = 0; y < dst->height; ++y) {
        float *drow = (float *)((uint8_t *)dst->data + (size_t)y * dst->stride);
        for (x = 0; x < dst->width; ++x)
            for (c = 0; c < dst->channels; ++c)
                drow[x * dst->channels + c] = tp_read_sample(src, x, y, c);
    }
}

static tir_mode tp_tir_mode(tp_content content) {
    switch (content) {
    case TP_CONTENT_NORMAL: return TIR_MODE_NORMAL_MAP;
    case TP_CONTENT_HEIGHT: return TIR_MODE_HEIGHTMAP;
    default: return TIR_MODE_GENERAL;
    }
}

static void tp_view_of_surface(const tp_surface *s, tir_image_view *v) {
    v->data = s->data;
    v->width = s->width;
    v->height = s->height;
    v->channels = s->channels;
    v->type = TIR_F32;
    v->row_stride_bytes = s->stride;
}

/* ------------------------------------------------------- build */

tp_result tp_build_mips(const tir_allocator *a, const tir_image_view *faces,
                        int num_faces, const tp_options *opt,
                        tp_mip_chain *out) {
    tp_codec_desc d;
    int face, level, num_levels, channels, normal_channels = 0;
    tp_result pr;

    if (!faces || !opt || !out || num_faces < 1) return TP_ERROR_INVALID_ARGUMENT;
    /* num_faces is 1 (2D) or 6 (cubemap, order +X,-X,+Y,-Y,+Z,-Z). */
    if (num_faces != 1 && num_faces != 6) return TP_ERROR_UNSUPPORTED;
    pr = tp_codec_describe(opt->codec, opt, &d);
    if (!TP_OK(pr)) return pr;

    if (opt->content == TP_CONTENT_NORMAL) {
        /* tir's normal-map mode filters 2 (RG) or 3 (xyz) components. The chain
         * surfaces hold exactly those; compression pads them back to RGBA. */
        normal_channels = (opt->normal_encoding == TIR_NORMAL_RG) ? 2 : 3;
        channels = normal_channels;
        if (faces[0].channels < normal_channels) return TP_ERROR_INVALID_ARGUMENT;
    } else {
        channels = faces[0].channels;
        if (channels != d.channels_in) return TP_ERROR_INVALID_ARGUMENT;
    }
    if (faces[0].width < 1 || faces[0].height < 1) return TP_ERROR_INVALID_ARGUMENT;
    if (num_faces == 6) {
        int fi;
        /* Cube faces must be square and identically sized. */
        if (faces[0].width != faces[0].height) return TP_ERROR_INVALID_ARGUMENT;
        for (fi = 1; fi < 6; ++fi)
            if (faces[fi].width != faces[0].width ||
                faces[fi].height != faces[0].height ||
                faces[fi].channels != faces[0].channels)
                return TP_ERROR_INVALID_ARGUMENT;
    }

    num_levels = tp_level_count(faces[0].width, faces[0].height, opt->max_levels);

    memset(out, 0, sizeof(*out));
    out->num_faces = num_faces;
    out->num_levels = num_levels;
    out->channels = channels;
    out->level = (tp_surface *)tp_alloc(
        a, (size_t)num_faces * (size_t)num_levels * sizeof(tp_surface));
    if (!out->level) return TP_ERROR_OUT_OF_MEMORY;
    memset(out->level, 0,
           (size_t)num_faces * (size_t)num_levels * sizeof(tp_surface));

    /* Toksvig roughness baking (normal maps only): allocate the per-surface
     * roughness pointer array; each entry filled below. */
    if (opt->content == TP_CONTENT_NORMAL && opt->bake_toksvig_roughness) {
        size_t np = (size_t)num_faces * (size_t)num_levels;
        out->roughness = (float **)tp_alloc(a, np * sizeof(float *));
        if (!out->roughness) {
            tp_mip_chain_free(a, out);
            return TP_ERROR_OUT_OF_MEMORY;
        }
        memset(out->roughness, 0, np * sizeof(float *));
    }

    for (face = 0; face < num_faces; ++face) {
        const tir_image_view *base = &faces[face];
        tir_image_view src_base = *base;
        float *base3 = NULL;
        float base_coverage = 0.0f;
        int do_coverage = (opt->content == TP_CONTENT_ALPHA_TESTED &&
                           opt->preserve_alpha_coverage != 0 && channels == 4);

        if (opt->content == TP_CONTENT_NORMAL) {
            /* Repack the base's first `channels` components into a tight F32
             * buffer so tir gets a 2/3-channel normal image (its requirement). */
            int bx, by, bc;
            base3 = (float *)tp_alloc(a, (size_t)base->width * (size_t)base->height *
                                             (size_t)channels * sizeof(float));
            if (!base3) { tp_mip_chain_free(a, out); return TP_ERROR_OUT_OF_MEMORY; }
            for (by = 0; by < base->height; ++by)
                for (bx = 0; bx < base->width; ++bx)
                    for (bc = 0; bc < channels; ++bc)
                        base3[((size_t)by * base->width + bx) * channels + bc] =
                            tp_read_sample(base, bx, by, bc);
            src_base.data = base3;
            src_base.channels = channels;
            src_base.type = TIR_F32;
            src_base.row_stride_bytes = 0;
        }

        for (level = 0; level < num_levels; ++level) {
            int idx = face * num_levels + level;
            tp_surface *s = &out->level[idx];
            int w = tp_level_dim(base->width, level);
            int h = tp_level_dim(base->height, level);
            float *rough = NULL;
            pr = tp_surface_alloc(a, s, w, h, channels);
            if (!TP_OK(pr)) {
                tp_dealloc(a, base3);
                tp_mip_chain_free(a, out);
                return pr;
            }
            if (out->roughness) {
                rough = (float *)tp_alloc(a, (size_t)w * (size_t)h * sizeof(float));
                if (!rough) {
                    tp_dealloc(a, base3);
                    tp_mip_chain_free(a, out);
                    return TP_ERROR_OUT_OF_MEMORY;
                }
                out->roughness[idx] = rough;
            }

            if (level == 0) {
                tp_copy_base(&src_base, s);
                if (do_coverage)
                    base_coverage = tp_alpha_coverage(s, opt->alpha_test_threshold);
                if (rough) {
                    /* Base normals are unit-length -> roughness == base. */
                    int k, np = w * h;
                    for (k = 0; k < np; ++k) rough[k] = opt->base_roughness;
                }
            } else {
                tir_options topt;
                tir_image_view srcv, dstv;
                tir_result tr;
                tir_options_init(&topt);
                topt.filter_x = opt->filter;
                topt.filter_y = opt->filter;
                topt.edge_x = opt->edge_x;
                topt.edge_y = opt->edge_y;
                topt.mode = tp_tir_mode(opt->content);
                topt.alpha = opt->alpha;
                topt.normal_encoding = opt->normal_encoding;
                topt.normal_renormalize = opt->renormalize;
                topt.num_threads = opt->threads;
                /* Capture pre-renormalize |N| into the roughness buffer, then
                 * map it in place (Toksvig). */
                topt.normal_length_out = rough;

                if (opt->mip_source == TP_MIP_FROM_PREVIOUS)
                    tp_view_of_surface(&out->level[face * num_levels + level - 1],
                                       &srcv);
                else
                    srcv = src_base;
                tp_view_of_surface(s, &dstv);

                tr = tir_resize(a, &srcv, &dstv, &topt);
                if (!TIR_OK(tr)) {
                    tp_dealloc(a, base3);
                    tp_mip_chain_free(a, out);
                    return TP_ERROR_UNSUPPORTED;
                }
                if (do_coverage)
                    tp_alpha_scale_to_coverage(s, base_coverage,
                                               opt->alpha_test_threshold);
                if (rough)
                    tp_toksvig_roughness(rough, w * h, opt->base_roughness, rough);
                /* Per-channel packed-map rule: threshold MAJORITY channels so a
                 * binary metallic/mask stays crisp instead of averaging gray. */
                if ((opt->content == TP_CONTENT_COLOR ||
                     opt->content == TP_CONTENT_ALPHA_TESTED) && channels == 4) {
                    int cc, xx, yy;
                    for (cc = 0; cc < 4; ++cc) {
                        if (opt->channel_op[cc] != TP_CH_MAJORITY) continue;
                        for (yy = 0; yy < h; ++yy) {
                            float *row = (float *)((uint8_t *)s->data +
                                                   (size_t)yy * s->stride);
                            for (xx = 0; xx < w; ++xx) {
                                float *px = row + xx * channels;
                                px[cc] = px[cc] >= 0.5f ? 1.0f : 0.0f;
                            }
                        }
                    }
                }
            }
        }
        tp_dealloc(a, base3);
    }

    /* Cube seam fixup: make face borders bit-identical per mip level, after all
     * 6 faces of a level exist. cube_fixup_max_level (-1 = all) bounds it. */
    if (num_faces == 6 && opt->cube_seam_fixup) {
        int maxlvl = opt->cube_fixup_max_level;
        for (level = 0; level < num_levels; ++level) {
            tp_surface fs[6];
            int fi;
            if (maxlvl >= 0 && level > maxlvl) break;
            for (fi = 0; fi < 6; ++fi) fs[fi] = out->level[fi * num_levels + level];
            tp_cube_seam_fixup(fs, opt);
        }
    }

    /* Octahedral fold-seam fixup: per-mip border coherence for a 2D octa map. */
    if (num_faces == 1 && opt->projection == TP_PROJ_OCTA && opt->octa_seam_fixup) {
        for (level = 0; level < num_levels; ++level)
            tp_octa_seam_fixup(&out->level[level], opt);
    }
    return TP_SUCCESS;
}

void tp_mip_chain_free(const tir_allocator *a, tp_mip_chain *c) {
    int i, n;
    if (!c) return;
    if (c->level) {
        n = c->num_faces * c->num_levels;
        for (i = 0; i < n; ++i) tp_dealloc(a, c->level[i].data);
        tp_dealloc(a, c->level);
        c->level = NULL;
    }
    if (c->roughness) {
        n = c->num_faces * c->num_levels;
        for (i = 0; i < n; ++i) tp_dealloc(a, c->roughness[i]);
        tp_dealloc(a, c->roughness);
        c->roughness = NULL;
    }
}
