/*
 * tocio - ACES BuiltinTransform expansion + FixedFunctionTransform lowering.
 *
 * A pragmatic subset: ACEScg/ACEScct <-> ACES2065-1 builtins (AP0/AP1 matrices +
 * the ACEScct camera-log curve), and the Rec.2100 surround fixed function.
 * Unknown styles return TOC_ERROR_UNSUPPORTED (loud, never silent).
 *
 * Constants and curves reimplemented from OpenColorIO / ACES (BSD-3-Clause).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

/* ACES AP1(ACEScg) -> AP0(ACES2065-1) and inverse (row-major 3x3). */
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

/* ACEScct camera log (linear AP1 <-> ACEScct). Forward is lin->ACEScct. */
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

/* Reverse + invert the ops appended since `start` (for builtin inversion). */
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
        /* ACEScct(AP1) -> linear AP1 -> AP0 */
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

/* ---- FixedFunction ------------------------------------------------------- */
toc_result toc_lower_fixedfunc(toc_op_list *list, const toc_node *node,
                               int invert) {
    const char *style = toc_node_scalar(toc_node_map_get(node, "style"));
    toc_op *op;
    if (!style) return TOC_ERROR_PARSE;
    if (strcmp(style, "REC2100_SURROUND") == 0 ||
        strcmp(style, "Rec2100Surround") == 0) {
        const toc_node *params = toc_node_map_get(node, "params");
        float gamma = 0.78f;
        if (params && params->kind == TOC_NODE_SEQ && params->n_items >= 1) {
            const char *s = toc_node_scalar(params->items[0]);
            const char *p = s;
            if (s) toc_parse_float(&p, s + strlen(s), &gamma);
        }
        op = toc_op_list_push(list, TOC_OP_FIXEDFUNC);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.fixedfunc.style =
            invert ? TOC_FF_REC2100_SURROUND_INV : TOC_FF_REC2100_SURROUND;
        op->u.fixedfunc.params[0] = gamma;
        op->u.fixedfunc.nparams = 1;
        return TOC_SUCCESS;
    }
    return TOC_ERROR_UNSUPPORTED;
}

/* ---- FixedFunction apply (overrides the identity stub in toc_lut.c is NOT
 * possible; this is THE definition — the stub was removed). ----------------- */
void toc_fixedfunc_apply_pixel(const toc_op *op, float *px, int ch) {
    switch (op->u.fixedfunc.style) {
        case TOC_FF_REC2100_SURROUND:
        case TOC_FF_REC2100_SURROUND_INV: {
            /* Y in Rec.2020; scale rgb by Y^(gamma-1). Inverse uses 1/gamma. */
            float g = op->u.fixedfunc.params[0];
            float Y = 0.2627f * px[0] + 0.6780f * px[1] + 0.0593f * px[2];
            float e = (op->u.fixedfunc.style == TOC_FF_REC2100_SURROUND)
                          ? (g - 1.0f)
                          : (1.0f / g - 1.0f);
            float s;
            if (Y <= 0.0f) return;
            s = toc_powf(Y, e);
            px[0] *= s; px[1] *= s; px[2] *= s;
            break;
        }
        default:
            break; /* unimplemented styles: identity */
    }
    (void)ch;
}
