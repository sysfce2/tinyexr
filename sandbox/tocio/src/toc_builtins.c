/*
 * tocio - ACES BuiltinTransform expansion + FixedFunctionTransform lowering.
 *
 * All OCIO FixedFunction styles are implemented: ACES glow/red-mod/dark-to-dim/
 * gamut-comp + Rec.2100 surround + RGB/HSV + XYZ/xyY/uvY/LUV conversions.
 * Unrecognized styles return TOC_ERROR_UNSUPPORTED (loud, never silent).
 *
 * Constants and curves reimplemented from OpenColorIO / ACES (BSD-3-Clause).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

/* Freestanding-friendly approx_atan2f (the RedMod hue weight doesn't need
 * high precision; the B-spline kernel is approximate anyway). */
static float ff_atan2f(float y, float x) {
    float ax = x < 0.0f ? -x : x, ay = y < 0.0f ? -y : y;
    float a, r, a2;
    int q = 0;
    if (ax + ay == 0.0f) return 0.0f;
    a = (ay < ax) ? y / x : x / y;
    if (ay < ax) q = 0; else q = 2;
    if (x < 0.0f && ay < ax) q = 1;
    if (y < 0.0f && ay >= ax) q = 3;
    a2 = a * a;
    r = a * (1.0f - a2 * (1.0f / 3.0f - a2 * (1.0f / 5.0f - a2 *
           (1.0f / 7.0f - a2 * (1.0f / 9.0f)))));
    switch (q) {
        case 1: r = (r < 0.0f ? -3.141592653589793f : 3.141592653589793f) + r; break;
        case 2: r = 1.5707963267948966f - r; break;
        case 3: r = -1.5707963267948966f - r; break;
    }
    return r;
}

static inline float ff_copysignf(float x, float y) {
    uint32_t ux, uy;
    memcpy(&ux, &x, sizeof(ux));
    memcpy(&uy, &y, sizeof(uy));
    ux = (ux & 0x7fffffffu) | (uy & 0x80000000u);
    memcpy(&x, &ux, sizeof(x));
    return x;
}

static inline float ff_fabsf(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u &= 0x7fffffffu;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float ff_floorf(float x) {
    float r = (float)(int)x;
    return (r > x) ? r - 1.0f : r;
}

static inline float ff_fmaxf(float a, float b) { return a > b ? a : b; }
static inline float ff_fminf(float a, float b) { return a < b ? a : b; }
static inline float ff_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/* ---- Bump-arena helpers for the node interface --------------------------- */
static float parse_param(const toc_node *params, int idx, float def) {
    if (params && params->kind == TOC_NODE_SEQ &&
        params->n_items > (size_t)idx) {
        const char *s = toc_node_scalar(params->items[idx]);
        const char *p = s;
        float v;
        if (s && toc_parse_float(&p, s + strlen(s), &v)) return v;
    }
    return def;
}

/* ---- ACES AP matrices + builtin transform expansion (unchanged) ---------- */
static const float AP1_TO_AP0[9] = {
    0.6954522414f, 0.1406786965f, 0.1638690622f,
    0.0447945634f, 0.8596711185f, 0.0955343182f,
    -0.0055258826f, 0.0040252103f, 1.0015006723f};
static const float AP0_TO_AP1[9] = {
    1.4514393161f, -0.2365107469f, -0.2149285693f,
    -0.0765537734f, 1.1762296998f, -0.0996759264f,
    0.0083161484f, -0.0060324498f, 0.9977163014f};

static toc_op *push_mat3(toc_op_list *list, const float rm3[9]) {
    toc_op *op = toc_op_list_push(list, TOC_OP_MATRIX);
    int r, c;
    if (!op) return NULL;
    for (r = 0; r < 4; ++r)
        for (c = 0; c < 4; ++c)
            op->u.matrix.m[c * 4 + r] =
                (r < 3 && c < 3) ? rm3[r * 3 + c] : (r == c ? 1.0f : 0.0f);
    return op;
}

static toc_op *push_acescct_log(toc_op_list *list, int log_to_lin) {
    toc_op *op = toc_op_list_push(list, TOC_OP_LOG_CAMERA);
    int i;
    if (!op) return NULL;
    op->u.logcam.base = 2.0f;
    for (i = 0; i < 3; ++i) {
        op->u.logcam.log_slope[i] = 1.0f / 17.52f;
        op->u.logcam.log_offset[i] = 9.72f / 17.52f;
        op->u.logcam.lin_slope[i] = 1.0f;
        op->u.logcam.lin_offset[i] = 0.0f;
        op->u.logcam.lin_break[i] = 0.0078125f;
        op->u.logcam.linear_slope[i] = 10.5402377416545f;
        op->u.logcam.linear_offset[i] = 0.0729055341958355f;
    }
    op->u.logcam.inverse = log_to_lin ? 1 : 0;
    return op;
}

static toc_result reverse_invert(toc_op_list *list, size_t start) {
    size_t i, j;
    for (i = start, j = list->count; i < j; ++i, --j) {
        toc_op t = list->ops[i];
        list->ops[i] = list->ops[j - 1];
        list->ops[j - 1] = t;
    }
    for (i = start; i < list->count; ++i) {
        toc_result rc = toc_invert_op(&list->ops[i]);
        if (!TOC_OK(rc)) return rc;
    }
    return TOC_SUCCESS;
}

toc_result toc_builtin_expand(toc_op_list *list, const char *style, int invert) {
    size_t start = list->count;
    toc_result rc = TOC_SUCCESS;
    int matched = 1;
    if (strcmp(style, "ACEScg_to_ACES2065-1") == 0) {
        if (!push_mat3(list, AP1_TO_AP0)) rc = TOC_ERROR_OUT_OF_MEMORY;
    } else if (strcmp(style, "ACES2065-1_to_ACEScg") == 0) {
        if (!push_mat3(list, AP0_TO_AP1)) rc = TOC_ERROR_OUT_OF_MEMORY;
    } else if (strcmp(style, "ACEScct_to_ACES2065-1") == 0) {
        if (!push_acescct_log(list, 1) || !push_mat3(list, AP1_TO_AP0))
            rc = TOC_ERROR_OUT_OF_MEMORY;
    } else if (strcmp(style, "ACES2065-1_to_ACEScct") == 0) {
        if (!push_mat3(list, AP0_TO_AP1) || !push_acescct_log(list, 0))
            rc = TOC_ERROR_OUT_OF_MEMORY;
    } else {
        matched = 0;
    }
    if (!matched) return TOC_ERROR_UNSUPPORTED;
    if (!TOC_OK(rc)) return rc;
    if (invert) return reverse_invert(list, start);
    return TOC_SUCCESS;
}

/* ---- FixedFunction lowering: parse YAML style + params -> op ------------- */
toc_result toc_lower_fixedfunc(toc_op_list *list, const toc_node *node,
                               int invert) {
    const char *style = toc_node_scalar(toc_node_map_get(node, "style"));
    const toc_node *params = toc_node_map_get(node, "params");
    toc_op *op;

    if (!style) return TOC_ERROR_PARSE;

    /* REC2100_SURROUND: one param (gamma, default 0.78). */
    if (strcmp(style, "REC2100_SURROUND") == 0 ||
        strcmp(style, "Rec2100Surround") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_REC2100_SURROUND_INV : TOC_FF_REC2100_SURROUND;
        op->u.fixedfunc.params[0] = parse_param(params, 0, 0.78f);
        op->u.fixedfunc.nparams = 1;
        return TOC_SUCCESS;
    }

    /* ACES_Glow_03: baked glowGain=0.075, glowMid=0.1 */
    if (strcmp(style, "ACES_Glow_03") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_ACES_GLOW03_INV : TOC_FF_ACES_GLOW03;
        op->u.fixedfunc.params[0] = 0.075f;
        op->u.fixedfunc.params[1] = 0.1f;
        op->u.fixedfunc.nparams = 2;
        return TOC_SUCCESS;
    }

    /* ACES_Glow_10: baked glowGain=0.05, glowMid=0.08 */
    if (strcmp(style, "ACES_Glow_10") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_ACES_GLOW10_INV : TOC_FF_ACES_GLOW10;
        op->u.fixedfunc.params[0] = 0.05f;
        op->u.fixedfunc.params[1] = 0.08f;
        op->u.fixedfunc.nparams = 2;
        return TOC_SUCCESS;
    }

    /* ACES_Dark_To_Dim_10: baked gamma */
    if (strcmp(style, "ACES_Dark_To_Dim_10") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_ACES_DARKTODIM10_INV : TOC_FF_ACES_DARKTODIM10;
        op->u.fixedfunc.params[0] = invert ? 1.0192640913260627f : 0.9811f;
        op->u.fixedfunc.nparams = 1;
        return TOC_SUCCESS;
    }

    /* ACES_Gamut_Comp_13: 7 params [limC,limM,limY, thrC,thrM,thrY, power] */
    if (strcmp(style, "ACES_Gamut_Comp_13") == 0) {
        float limC = parse_param(params, 0, 0.0f);
        float limM = parse_param(params, 1, 0.0f);
        float limY = parse_param(params, 2, 0.0f);
        float thrC = parse_param(params, 3, 0.0f);
        float thrM = parse_param(params, 4, 0.0f);
        float thrY = parse_param(params, 5, 0.0f);
        float pwr  = parse_param(params, 6, 1.0f);
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_ACES_GAMUTCOMP13_INV : TOC_FF_ACES_GAMUTCOMP13;
        op->u.fixedfunc.params[0] = limC;
        op->u.fixedfunc.params[1] = limM;
        op->u.fixedfunc.params[2] = limY;
        op->u.fixedfunc.params[3] = thrC;
        op->u.fixedfunc.params[4] = thrM;
        op->u.fixedfunc.params[5] = thrY;
        op->u.fixedfunc.params[6] = pwr;
        op->u.fixedfunc.nparams = 7;
        return TOC_SUCCESS;
    }

    /* ACES_Red_Mod_03: no params (baked constants) */
    if (strcmp(style, "ACES_Red_Mod_03") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_ACES_RED_MOD_03_INV : TOC_FF_ACES_RED_MOD_03;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    /* ACES_Red_Mod_10: no params (baked constants) */
    if (strcmp(style, "ACES_Red_Mod_10") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_ACES_RED_MOD_10_INV : TOC_FF_ACES_RED_MOD_10;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    /* Colorspace conversions: no params. */
    if (strcmp(style, "RGB_TO_HSV") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_HSV_TO_RGB : TOC_FF_RGB_TO_HSV;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "HSV_TO_RGB") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_RGB_TO_HSV : TOC_FF_HSV_TO_RGB;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "XYZ_TO_xyY") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_xyY_TO_XYZ : TOC_FF_XYZ_TO_xyY;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "xyY_TO_XYZ") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_XYZ_TO_xyY : TOC_FF_xyY_TO_XYZ;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "XYZ_TO_uvY") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_uvY_TO_XYZ : TOC_FF_XYZ_TO_uvY;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "uvY_TO_XYZ") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_XYZ_TO_uvY : TOC_FF_uvY_TO_XYZ;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "XYZ_TO_LUV") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_LUV_TO_XYZ : TOC_FF_XYZ_TO_LUV;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    if (strcmp(style, "LUV_TO_XYZ") == 0) {
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style = invert ? TOC_FF_XYZ_TO_LUV : TOC_FF_LUV_TO_XYZ;
        op->u.fixedfunc.nparams = 0;
        return TOC_SUCCESS;
    }

    return TOC_ERROR_UNSUPPORTED;
}

/* ---- Per-style private helpers ------------------------------------------- */

/* ---- ACES Red Mod 03/10: hue-weighted saturation modulation --------------- */

/* Calculate hue weight using quadratic B-spline (from OCIO). */
static float redmod_hue_weight(float red, float grn, float blu,
                                float inv_width) {
    float a = 2.0f * red - (grn + blu);
    float b = 1.7320508075688772f * (grn - blu);
    float hue = ff_atan2f(b, a);
    float knot = hue * inv_width + 2.0f;
    int j = (int)knot;
    if (j < 0 || j >= 4) return 0.0f;
    {
        static const float M[4][4] = {
            {0.25f, 0.00f, 0.00f, 0.00f},
            {-0.75f, 0.75f, 0.75f, 0.25f},
            {0.75f, -1.50f, 0.00f, 1.00f},
            {-0.25f, 0.75f, -0.75f, 0.25f}};
        float t = knot - (float)j;
        const float *coefs = M[j];
        return coefs[3] + t * (coefs[2] + t * (coefs[1] + t * coefs[0]));
    }
}

static float redmod_sat_weight(float red, float grn, float blu) {
    float mn = ff_fminf(red, ff_fminf(grn, blu));
    float mx = ff_fmaxf(red, ff_fmaxf(grn, blu));
    float s_num = ff_fmaxf(1e-10f, mx) - ff_fmaxf(1e-10f, mn);
    float s_den = ff_fmaxf(1e-2f, mx);
    return s_num / s_den;
}

static void apply_redmod_fwd(float *px, float scale, float pivot,
                              float inv_width) {
    float red = px[0], grn = px[1], blu = px[2];
    float f_H = redmod_hue_weight(red, grn, blu, inv_width);
    if (f_H > 0.0f) {
        float f_S = redmod_sat_weight(red, grn, blu);
        float one_minus_scale = 1.0f - scale;
        float newRed = red + f_H * f_S * (pivot - red) * one_minus_scale;
        if (grn >= blu) {
            float hue_fac = (grn - blu) / ff_fmaxf(1e-10f, red - blu);
            grn = hue_fac * (newRed - blu) + blu;
        } else {
            float hue_fac = (blu - grn) / ff_fmaxf(1e-10f, red - grn);
            blu = hue_fac * (newRed - grn) + grn;
        }
        red = newRed;
    }
    px[0] = red; px[1] = grn; px[2] = blu;
}

static void apply_redmod_inv(float *px, float scale, float pivot,
                              float inv_width) {
    float red = px[0], grn = px[1], blu = px[2];
    float f_H = redmod_hue_weight(red, grn, blu, inv_width);
    if (f_H > 0.0f) {
        float one_minus_scale = 1.0f - scale;
        float minChan = ff_fminf(grn, blu);
        float a = f_H * one_minus_scale - 1.0f;
        float b = red - f_H * (pivot + minChan) * one_minus_scale;
        float c_val = f_H * pivot * minChan * one_minus_scale;
        float disc = b * b - 4.0f * a * c_val;
        float newRed = disc >= 0.0f ? (-b - toc_sqrtf(disc)) / (2.0f * a) : red;
        if (grn >= blu) {
            float hue_fac = (grn - blu) / ff_fmaxf(1e-10f, red - blu);
            grn = hue_fac * (newRed - blu) + blu;
        } else {
            float hue_fac = (blu - grn) / ff_fmaxf(1e-10f, red - grn);
            blu = hue_fac * (newRed - grn) + grn;
        }
        red = newRed;
    }
    px[0] = red; px[1] = grn; px[2] = blu;
}

/* ---- ACES Glow: sigmoid-weighted luminance-dependent glow ----------------- */
static float glow_yc(float red, float grn, float blu) {
    float c = blu * (blu - grn) + grn * (grn - red) + red * (red - blu);
    float chroma = c > 0.0f ? toc_sqrtf(c) : 0.0f;
    return (blu + grn + red + 1.75f * chroma) / 3.0f;
}

static float glow_sigmoid(float sat) {
    float x = (sat - 0.4f) * 5.0f;
    float sign = ff_copysignf(1.0f, x);
    float t = ff_fmaxf(0.0f, 1.0f - 0.5f * sign * x);
    return (1.0f + sign * (1.0f - t * t)) * 0.5f;
}

static void apply_glow_fwd(float *px, float glowGain, float glowMid) {
    float red = px[0], grn = px[1], blu = px[2];
    float YC = glow_yc(red, grn, blu);
    float sat = redmod_sat_weight(red, grn, blu);
    float s = glow_sigmoid(sat);
    float GG = glowGain * s;
    float gm = glowMid;
    float glowGainOut;
    if (YC >= gm * 2.0f) {
        glowGainOut = 0.0f;
    } else if (YC <= gm * 2.0f / 3.0f) {
        glowGainOut = GG;
    } else {
        glowGainOut = GG * (gm / YC - 0.5f);
    }
    float addedGlow = 1.0f + glowGainOut;
    px[0] = red * addedGlow; px[1] = grn * addedGlow; px[2] = blu * addedGlow;
}

static void apply_glow_inv(float *px, float glowGain, float glowMid) {
    float red = px[0], grn = px[1], blu = px[2];
    float YC = glow_yc(red, grn, blu);
    float sat = redmod_sat_weight(red, grn, blu);
    float s = glow_sigmoid(sat);
    float GG = glowGain * s;
    float gm = glowMid;
    float glowGainOut;
    if (YC >= gm * 2.0f) {
        glowGainOut = 0.0f;
    } else if (YC <= (1.0f + GG) * gm * 2.0f / 3.0f) {
        glowGainOut = -GG / (1.0f + GG);
    } else {
        glowGainOut = GG * (gm / YC - 0.5f) / (GG * 0.5f - 1.0f);
    }
    float reducedGlow = 1.0f + glowGainOut;
    px[0] = red * reducedGlow; px[1] = grn * reducedGlow; px[2] = blu * reducedGlow;
}

/* ---- ACES Dark-to-Dim: apply Y^(gamma-1) scaling on AP1 luminance -------- */
static void apply_darktodim(float *px, float gamma_minus_one) {
    float Y = ff_fmaxf(1e-10f,
        0.27222871678091454f * px[0] +
        0.67408176581114831f * px[1] +
        0.053689517407937051f * px[2]);
    float s = toc_powf(Y, gamma_minus_one);
    px[0] *= s; px[1] *= s; px[2] *= s;
}

/* ---- ACES Gamut Comp 13: parametric distance compression per axis --------- */
static float gc_compress(float dist, float thr, float scale, float power) {
    float nd = (dist - thr) / scale;
    float p = toc_powf(nd, power);
    float ip = 1.0f / power;
    return thr + scale * nd / toc_powf(1.0f + p, ip);
}

static float gc_uncompress(float dist, float thr, float scale, float power) {
    if (dist >= thr + scale) return dist;
    float nd = (dist - thr) / scale;
    float p = toc_powf(nd, power);
    float ip = 1.0f / power;
    return thr + scale * toc_powf(-(p / (p - 1.0f)), ip);
}

static float gc_apply(float val, float ach, float thr, float scale,
                       float power, int invert) {
    if (ach == 0.0f) return 0.0f;
    float dist = (ach - val) / ff_fabsf(ach);
    if (dist < thr) return val;
    float compr = invert ? gc_uncompress(dist, thr, scale, power)
                         : gc_compress(dist, thr, scale, power);
    return ach - compr * ff_fabsf(ach);
}

/* Precompute scale factor for gamut compression. */
static float gc_scale(float lim, float thr, float power) {
    if (lim <= thr) return 1.0f;
    float ip = 1.0f / power;
    float t = (1.0f - thr) / (lim - thr);
    float p = toc_powf(t, power);
    float d = toc_powf(p - 1.0f, ip);
    return (1.0f - thr) / d;
}

/* ---- RGB/HSV conversions ------------------------------------------------- */
static void apply_rgb_to_hsv(float *px) {
    float r = px[0], g = px[1], b = px[2];
    float mn = ff_fminf(r, ff_fminf(g, b));
    float mx = ff_fmaxf(r, ff_fmaxf(g, b));
    float val = mx, sat = 0.0f, hue = 0.0f;
    if (mn != mx) {
        float delta = mx - mn;
        if (mx != 0.0f) sat = delta / mx;
        if (r == mx)
            hue = (g - b) / delta;
        else if (g == mx)
            hue = 2.0f + (b - r) / delta;
        else
            hue = 4.0f + (r - g) / delta;
        if (hue < 0.0f) hue += 6.0f;
        hue *= 0.16666666666666666f;
    }
    if (mn < 0.0f) val += mn;
    if (-mn > mx) sat = (mx - mn) / -mn;
    px[0] = hue; px[1] = sat; px[2] = val;
}

static void apply_hsv_to_rgb(float *px) {
    float h = (px[0] - ff_floorf(px[0])) * 6.0f;
    float s = ff_clampf(px[1], 0.0f, 1.999f);
    float v = px[2];
    float r = ff_clampf(ff_fabsf(h - 3.0f) - 1.0f, 0.0f, 1.0f);
    float g = ff_clampf(2.0f - ff_fabsf(h - 2.0f), 0.0f, 1.0f);
    float b = ff_clampf(2.0f - ff_fabsf(h - 4.0f), 0.0f, 1.0f);
    float rgb_max = v, rgb_min = v * (1.0f - s);
    if (s > 1.0f) {
        rgb_min = v * (1.0f - s) / (2.0f - s);
        rgb_max = v - rgb_min;
    }
    if (v < 0.0f) {
        rgb_min = v / (2.0f - s);
        rgb_max = v - rgb_min;
    }
    float delta = rgb_max - rgb_min;
    px[0] = r * delta + rgb_min;
    px[1] = g * delta + rgb_min;
    px[2] = b * delta + rgb_min;
}

/* ---- XYZ / xyY conversions ------------------------------------------------ */
static void apply_xyz_to_xyy(float *px) {
    float X = px[0], Y = px[1], Z = px[2];
    float d = (X + Y + Z);
    d = (d == 0.0f) ? 0.0f : 1.0f / d;
    px[0] = X * d;
    px[1] = Y * d;
    px[2] = Y;
}

static void apply_xyy_to_xyz(float *px) {
    float x = px[0], y = px[1], Y = px[2];
    float d = (y == 0.0f) ? 0.0f : 1.0f / y;
    px[0] = Y * x * d;
    px[1] = Y;
    px[2] = Y * (1.0f - x - y) * d;
}

/* ---- XYZ / uvY conversions ------------------------------------------------ */
static void apply_xyz_to_uvy(float *px) {
    float X = px[0], Y = px[1], Z = px[2];
    float d = X + 15.0f * Y + 3.0f * Z;
    d = (d == 0.0f) ? 0.0f : 1.0f / d;
    px[0] = 4.0f * X * d;
    px[1] = 9.0f * Y * d;
    px[2] = Y;
}

static void apply_uvy_to_xyz(float *px) {
    float u = px[0], v = px[1], Y = px[2];
    float d = (v == 0.0f) ? 0.0f : 1.0f / v;
    float X = (9.0f / 4.0f) * Y * u * d;
    float Z = (3.0f / 4.0f) * Y * (4.0f - u - 6.666666666666667f * v) * d;
    px[0] = X;
    px[1] = Y;
    px[2] = Z;
}

/* ---- XYZ / LUV (CIELUV, D65 white) ---------------------------------------- */
static void apply_xyz_to_luv(float *px) {
    float X = px[0], Y = px[1], Z = px[2];
    float d = X + 15.0f * Y + 3.0f * Z;
    d = (d == 0.0f) ? 0.0f : 1.0f / d;
    float u = 4.0f * X * d;
    float v = 9.0f * Y * d;
    float Lstar = (Y <= 0.008856451679f)
                      ? 9.0329629629629608f * Y
                      : 1.16f * toc_powf(Y, 0.333333333f) - 0.16f;
    float ustar = 13.0f * Lstar * (u - 0.19783001f);
    float vstar = 13.0f * Lstar * (v - 0.46831999f);
    px[0] = Lstar; px[1] = ustar; px[2] = vstar;
}

static void apply_luv_to_xyz(float *px) {
    float Lstar = px[0], ustar = px[1], vstar = px[2];
    float d = (Lstar == 0.0f) ? 0.0f : 0.076923076923076927f / Lstar;
    float u = ustar * d + 0.19783001f;
    float v = vstar * d + 0.46831999f;
    float tmp = (Lstar + 0.16f) * 0.86206896551724144f;
    float Y = (Lstar <= 0.08f) ? 0.11070564598794539f * Lstar
                               : tmp * tmp * tmp;
    float dd = (v == 0.0f) ? 0.0f : 0.25f / v;
    float X = 9.0f * Y * u * dd;
    float Z = Y * (12.0f - 3.0f * u - 20.0f * v) * dd;
    px[0] = X; px[1] = Y; px[2] = Z;
}

/* ---- FixedFunction apply (the single dispatch point) ----------------------- */
void toc_fixedfunc_apply_pixel(const toc_op *op, float *px, int ch) {
    (void)ch;
    switch (op->u.fixedfunc.style) {
        case TOC_FF_REC2100_SURROUND:
        case TOC_FF_REC2100_SURROUND_INV: {
            float g = op->u.fixedfunc.params[0];
            float Y = 0.2627f * px[0] + 0.6780f * px[1] + 0.0593f * px[2];
            float e = (op->u.fixedfunc.style == TOC_FF_REC2100_SURROUND)
                          ? (g - 1.0f)
                          : (1.0f / g - 1.0f);
            if (Y <= 0.0f) return;
            Y = toc_powf(Y, e);
            px[0] *= Y; px[1] *= Y; px[2] *= Y;
            return;
        }

        /* ACES Glow 03/10 forward */
        case TOC_FF_ACES_GLOW03:
        case TOC_FF_ACES_GLOW10:
            apply_glow_fwd(px, op->u.fixedfunc.params[0],
                           op->u.fixedfunc.params[1]);
            return;

        /* ACES Glow 03/10 inverse */
        case TOC_FF_ACES_GLOW03_INV:
        case TOC_FF_ACES_GLOW10_INV:
            apply_glow_inv(px, op->u.fixedfunc.params[0],
                           op->u.fixedfunc.params[1]);
            return;

        /* ACES Dark-to-Dim 10 */
        case TOC_FF_ACES_DARKTODIM10:
            apply_darktodim(px, op->u.fixedfunc.params[0] - 1.0f);
            return;
        case TOC_FF_ACES_DARKTODIM10_INV: {
            float g = op->u.fixedfunc.params[0];
            apply_darktodim(px, (1.0f / g) - 1.0f);
            return;
        }

        /* ACES Gamut Comp 13 */
        case TOC_FF_ACES_GAMUTCOMP13:
        case TOC_FF_ACES_GAMUTCOMP13_INV: {
            int inv = (op->u.fixedfunc.style == TOC_FF_ACES_GAMUTCOMP13_INV);
            const float *p = op->u.fixedfunc.params;
            float limC = p[0], limM = p[1], limY = p[2];
            float thrC = p[3], thrM = p[4], thrY = p[5];
            float power = p[6];
            float sc = gc_scale(limC, thrC, power);
            float sm = gc_scale(limM, thrM, power);
            float sy = gc_scale(limY, thrY, power);
            float ach = ff_fmaxf(px[0], ff_fmaxf(px[1], px[2]));
            px[0] = gc_apply(px[0], ach, thrC, sc, power, inv);
            px[1] = gc_apply(px[1], ach, thrM, sm, power, inv);
            px[2] = gc_apply(px[2], ach, thrY, sy, power, inv);
            return;
        }

        /* ACES Red Mod 03 */
        case TOC_FF_ACES_RED_MOD_03:
            apply_redmod_fwd(px, 0.85f, 0.03f, 1.9098593171027443f);
            return;
        case TOC_FF_ACES_RED_MOD_03_INV:
            apply_redmod_inv(px, 0.85f, 0.03f, 1.9098593171027443f);
            return;

        /* ACES Red Mod 10 */
        case TOC_FF_ACES_RED_MOD_10:
            apply_redmod_fwd(px, 0.82f, 0.03f, 1.6976527263135504f);
            return;
        case TOC_FF_ACES_RED_MOD_10_INV:
            apply_redmod_inv(px, 0.82f, 0.03f, 1.6976527263135504f);
            return;

        /* Colorspace conversions */
        case TOC_FF_RGB_TO_HSV: apply_rgb_to_hsv(px); return;
        case TOC_FF_HSV_TO_RGB: apply_hsv_to_rgb(px); return;
        case TOC_FF_XYZ_TO_xyY: apply_xyz_to_xyy(px); return;
        case TOC_FF_xyY_TO_XYZ: apply_xyy_to_xyz(px); return;
        case TOC_FF_XYZ_TO_uvY: apply_xyz_to_uvy(px); return;
        case TOC_FF_uvY_TO_XYZ: apply_uvy_to_xyz(px); return;
        case TOC_FF_XYZ_TO_LUV: apply_xyz_to_luv(px); return;
        case TOC_FF_LUV_TO_XYZ: apply_luv_to_xyz(px); return;

        default:
            break;
    }
}
