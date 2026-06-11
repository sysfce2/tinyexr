/*
 * tocio - processor: config + endpoints -> flat op list. Composes a colorspace
 * conversion through the reference space, flattens GroupTransform, resolves
 * FileTransform via the file-reader hook, expands BuiltinTransform, and inverts
 * ops as required by direction.
 *
 * Reimplemented from OpenColorIO (BSD-3-Clause).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

/* ---- scalar / vector parameter reads ------------------------------------- */
static int scalar_to_float(const char *s, float *out) {
    const char *p = s, *end;
    if (!s) return 0;
    end = s + strlen(s);
    return toc_parse_float(&p, end, out);
}

/* Read up to maxn floats from a SEQ value, or broadcast a single SCALAR.
 * Returns the number written (0 if absent). */
static int read_vec(const toc_node *parent, const char *key, float *out,
                    int maxn) {
    const toc_node *v = toc_node_map_get(parent, key);
    int i;
    if (!v) return 0;
    if (v->kind == TOC_NODE_SEQ) {
        int n = (int)v->n_items < maxn ? (int)v->n_items : maxn;
        for (i = 0; i < n; ++i)
            if (!scalar_to_float(toc_node_scalar(v->items[i]), &out[i]))
                out[i] = 0.0f;
        return n;
    }
    if (v->kind == TOC_NODE_SCALAR) {
        float f = 0.0f;
        scalar_to_float(toc_node_scalar(v), &f);
        for (i = 0; i < maxn; ++i) out[i] = f;
        return 1;
    }
    return 0;
}

static int read_scalar(const toc_node *parent, const char *key, float *out) {
    const toc_node *v = toc_node_map_get(parent, key);
    if (!v || v->kind != TOC_NODE_SCALAR) return 0;
    return scalar_to_float(toc_node_scalar(v), out);
}

static void matvec4(const float *m /*row-major*/, const float *v, float *out) {
    int r;
    for (r = 0; r < 4; ++r)
        out[r] = m[r * 4 + 0] * v[0] + m[r * 4 + 1] * v[1] +
                 m[r * 4 + 2] * v[2] + m[r * 4 + 3] * v[3];
}

/* ---- MonCurve (ExponentWithLinear) forward params from (gamma,offset) ----- */
static void moncurve_params(float gamma, float offset, float *scale, float *off,
                            float *g, float *brk, float *slope) {
    double G = gamma < 1.000001 ? 1.000001 : gamma;
    double O = offset < 1e-6 ? 1e-6 : offset;
    double a = (G - 1.0) / O;
    double b = O * G / ((G - 1.0) * (1.0 + O));
    *g = (float)G;
    *scale = (float)(1.0 / (1.0 + O));
    *off = (float)(O / (1.0 + O));
    *brk = (float)(O / (G - 1.0));
    *slope = (float)(a * toc_powf((float)b, (float)G));
}

/* ---- lower a primitive transform into an op ------------------------------ */
toc_result toc_lower_transform(const toc_config *cfg, toc_op_list *list,
                               const toc_node *node, int invert) {
    const char *tag = node->tag;
    toc_op *op;
    (void)cfg;
    if (!tag) return TOC_SUCCESS;

    if (strcmp(tag, "MatrixTransform") == 0) {
        float rm[16], off[4] = {0, 0, 0, 0};
        int i;
        float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        if (read_vec(node, "matrix", rm, 16) != 16) memcpy(rm, ident, sizeof(rm));
        read_vec(node, "offset", off, 4);
        if (invert) {
            float inv[16], noff[4], tmp[4];
            if (!toc_inv4x4(rm, inv)) return TOC_ERROR_NONINVERTIBLE;
            matvec4(inv, off, tmp);
            for (i = 0; i < 4; ++i) noff[i] = -tmp[i];
            memcpy(rm, inv, sizeof(rm));
            memcpy(off, noff, sizeof(off));
        }
        op = toc_op_list_push(list, TOC_OP_MATRIX);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        /* config matrix is row-major; interp uses col-major m[c*4+r] */
        for (i = 0; i < 4; ++i) {
            int j;
            for (j = 0; j < 4; ++j) op->u.matrix.m[j * 4 + i] = rm[i * 4 + j];
            op->u.matrix.off[i] = off[i];
        }
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "ExponentTransform") == 0) {
        float e[4] = {1, 1, 1, 1};
        int i;
        read_vec(node, "value", e, 4);
        op = toc_op_list_push(list, TOC_OP_EXPONENT);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        for (i = 0; i < 4; ++i)
            op->u.exponent.e[i] = invert ? (e[i] != 0.0f ? 1.0f / e[i] : 1.0f)
                                         : e[i];
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "ExponentWithLinearTransform") == 0) {
        float gamma[4] = {1, 1, 1, 1}, offset[4] = {0, 0, 0, 0};
        int i;
        read_vec(node, "gamma", gamma, 4);
        read_vec(node, "offset", offset, 4);
        op = toc_op_list_push(list, TOC_OP_EXP_LINEAR);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        for (i = 0; i < 4; ++i)
            moncurve_params(gamma[i], offset[i], &op->u.exp_linear.scale[i],
                            &op->u.exp_linear.offset[i],
                            &op->u.exp_linear.gamma[i],
                            &op->u.exp_linear.breakpoint[i],
                            &op->u.exp_linear.slope[i]);
        op->u.exp_linear.inverse = invert ? 1 : 0;
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "LogTransform") == 0 ||
        strcmp(tag, "LogAffineTransform") == 0) {
        float base = 2.0f;
        float ls[3] = {1, 1, 1}, lo[3] = {0, 0, 0};
        float ns[3] = {1, 1, 1}, no[3] = {0, 0, 0};
        int i;
        read_scalar(node, "base", &base);
        read_vec(node, "logSideSlope", ls, 3);
        read_vec(node, "logSideOffset", lo, 3);
        read_vec(node, "linSideSlope", ns, 3);
        read_vec(node, "linSideOffset", no, 3);
        op = toc_op_list_push(list, TOC_OP_LOG);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.log.base = base;
        for (i = 0; i < 3; ++i) {
            op->u.log.log_slope[i] = ls[i];
            op->u.log.log_offset[i] = lo[i];
            op->u.log.lin_slope[i] = ns[i];
            op->u.log.lin_offset[i] = no[i];
        }
        /* LogTransform forward is lin->log; "direction: inverse" => log->lin */
        op->u.log.inverse = invert ? 1 : 0;
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "LogCameraTransform") == 0) {
        float base = 2.0f;
        float ls[3] = {1, 1, 1}, lo[3] = {0, 0, 0};
        float ns[3] = {1, 1, 1}, no[3] = {0, 0, 0};
        float brk[3] = {0, 0, 0}, lslope[3];
        int has_lslope, i;
        read_scalar(node, "base", &base);
        read_vec(node, "logSideSlope", ls, 3);
        read_vec(node, "logSideOffset", lo, 3);
        read_vec(node, "linSideSlope", ns, 3);
        read_vec(node, "linSideOffset", no, 3);
        if (read_vec(node, "linSideBreak", brk, 3) == 0)
            return TOC_ERROR_PARSE; /* required */
        has_lslope = read_vec(node, "linearSlope", lslope, 3) != 0;
        op = toc_op_list_push(list, TOC_OP_LOG_CAMERA);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        op->u.logcam.base = base;
        for (i = 0; i < 3; ++i) {
            float lnb = toc_log2f(base) * 0.6931471805599453f; /* ln(base) */
            float xb = ns[i] * brk[i] + no[i];
            float yb = ls[i] * (toc_log2f(xb > 0 ? xb : 1e-30f) / toc_log2f(base)) + lo[i];
            float lsl = has_lslope ? lslope[i]
                                   : (ls[i] * ns[i] / ((xb != 0 ? xb : 1e-30f) * lnb));
            op->u.logcam.log_slope[i] = ls[i];
            op->u.logcam.log_offset[i] = lo[i];
            op->u.logcam.lin_slope[i] = ns[i];
            op->u.logcam.lin_offset[i] = no[i];
            op->u.logcam.lin_break[i] = brk[i];
            op->u.logcam.linear_slope[i] = lsl;
            op->u.logcam.linear_offset[i] = yb - lsl * brk[i]; /* C0 continuity */
        }
        op->u.logcam.inverse = invert ? 1 : 0;
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "CDLTransform") == 0) {
        float sl[3] = {1, 1, 1}, of[3] = {0, 0, 0}, pw[3] = {1, 1, 1};
        float sat = 1.0f;
        const toc_node *style;
        int i, clamp_val = 1;
        read_vec(node, "slope", sl, 3);
        read_vec(node, "offset", of, 3);
        read_vec(node, "power", pw, 3);
        read_scalar(node, "sat", &sat);
        style = toc_node_map_get(node, "style");
        if (style) {
            const char *s = toc_node_scalar(style);
            if (s && strcmp(s, "noclamp") == 0) clamp_val = 0;
        }
        if (invert) {
            /* Decompose CDL inverse into basic ops for backend-agnostic support.
             * Forward: slope*in+offset -> power -> saturation.
             * Inverse: inv_saturation -> inv_power -> inv_slope+offset. */
            float luma[3] = {0.2126f, 0.7152f, 0.0593f};
            float ny = 1.0f / sat;
            /* 1. Inverse saturation as a matrix:
             *    c' = c/s - ((1-s)/s)*broadcast(luma·c)
             *  => M[i][j] = delta(i,j)/s + (1-1/s)*luma[j]   (col-major). */
            {
                toc_op *m = toc_op_list_push(list, TOC_OP_MATRIX);
                if (!m) return TOC_ERROR_OUT_OF_MEMORY;
                memset(m->u.matrix.m, 0, sizeof(m->u.matrix.m));
                for (i = 0; i < 3; ++i) {
                    int j;
                    for (j = 0; j < 3; ++j)
                        m->u.matrix.m[j * 4 + i] =
                            (i == j ? ny : 0.0f) + (1.0f - ny) * luma[j];
                }
                m->u.matrix.m[15] = 1.0f;
                memset(m->u.matrix.off, 0, sizeof(m->u.matrix.off));
            }
            /* 2. Inverse power: pow(c, 1/power) */
            {
                toc_op *e = toc_op_list_push(list, TOC_OP_EXPONENT);
                if (!e) return TOC_ERROR_OUT_OF_MEMORY;
                for (i = 0; i < 3; ++i)
                    e->u.exponent.e[i] = (pw[i] != 0.0f) ? 1.0f / pw[i] : 1.0f;
                e->u.exponent.e[3] = 1.0f;
            }
            /* 3. Inverse (slope,offset): (c - offset) / slope = (1/slope)*c - offset/slope */
            {
                toc_op *r = toc_op_list_push(list, TOC_OP_RANGE);
                if (!r) return TOC_ERROR_OUT_OF_MEMORY;
                for (i = 0; i < 4; ++i) {
                    float s = (i < 3 && sl[i] != 0.0f) ? 1.0f / sl[i] : 1.0f;
                    r->u.range.scale[i] = s;
                    r->u.range.offset[i] = (i < 3) ? -of[i] * s : 0.0f;
                    r->u.range.min[i] = 0.0f;
                    r->u.range.max[i] = 1.0f;
                }
                r->u.range.clamp_lo = clamp_val;
                r->u.range.clamp_hi = clamp_val;
            }
            return TOC_SUCCESS;
        }
        op = toc_op_list_push(list, TOC_OP_CDL);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        for (i = 0; i < 3; ++i) {
            op->u.cdl.slope[i] = sl[i];
            op->u.cdl.offset[i] = of[i];
            op->u.cdl.power[i] = pw[i];
        }
        op->u.cdl.saturation = sat;
        op->u.cdl.clamp = clamp_val;
        return TOC_SUCCESS;
    }

    if (strcmp(tag, "RangeTransform") == 0) {
        float minIn, maxIn, minOut, maxOut, scale = 1.0f, offset = 0.0f;
        int hasIn = read_scalar(node, "minInValue", &minIn);
        int hasInH = read_scalar(node, "maxInValue", &maxIn);
        int hasOut = read_scalar(node, "minOutValue", &minOut);
        int hasOutH = read_scalar(node, "maxOutValue", &maxOut);
        int i;
        if (hasIn && hasInH && hasOut && hasOutH && (maxIn - minIn) != 0.0f) {
            scale = (maxOut - minOut) / (maxIn - minIn);
            offset = minOut - minIn * scale;
        }
        op = toc_op_list_push(list, TOC_OP_RANGE);
        if (!op) return TOC_ERROR_OUT_OF_MEMORY;
        for (i = 0; i < 4; ++i) {
            op->u.range.scale[i] = scale;
            op->u.range.offset[i] = offset;
            op->u.range.min[i] = hasOut ? minOut : 0.0f;
            op->u.range.max[i] = hasOutH ? maxOut : 1.0f;
        }
        op->u.range.clamp_lo = hasOut;
        op->u.range.clamp_hi = hasOutH;
        if (invert) return toc_invert_op(list, op);
        return TOC_SUCCESS;
    }

    return TOC_ERROR_UNSUPPORTED;
}

/* ---- invert an already-lowered op in place ------------------------------- */
/* Build the inverse of a monotonic (non-decreasing) LUT1D into newly-allocated
 * owned storage. Per channel f maps the input domain to [omin_c, omax_c]; the
 * inverse is sampled on a common output domain [min omin_c, max omax_c] and, for
 * each target, the input is found by binary search + linear interpolation on the
 * forward curve. Non-monotonic or degenerate curves are not invertible. */
static toc_result lut1d_invert(toc_op_list *list, toc_lut1d *lut) {
    int N = lut->length, ch = lut->channels, c, j;
    const float *fwd = lut->data;
    float dmin = lut->domain_min, dmax = lut->domain_max;
    float omin = 1e30f, omax = -1e30f, *inv;
    if (!list || N < 2 || (ch != 1 && ch != 3) || !fwd)
        return TOC_ERROR_NONINVERTIBLE;
    for (c = 0; c < ch; ++c) {
        int i;
        for (i = 0; i + 1 < N; ++i)
            if (fwd[(i + 1) * ch + c] < fwd[i * ch + c])
                return TOC_ERROR_NONINVERTIBLE; /* not non-decreasing */
        if (fwd[c] < omin) omin = fwd[c];
        if (fwd[(N - 1) * ch + c] > omax) omax = fwd[(N - 1) * ch + c];
    }
    if (!(omax > omin)) return TOC_ERROR_NONINVERTIBLE; /* flat / degenerate */
    inv = (float *)toc_malloc(&list->alloc, (size_t)N * ch * sizeof(float));
    if (!inv) return TOC_ERROR_OUT_OF_MEMORY;
    for (c = 0; c < ch; ++c) {
        for (j = 0; j < N; ++j) {
            float t = omin + (omax - omin) * (float)j / (float)(N - 1);
            float x;
            if (t <= fwd[c]) {
                x = dmin;
            } else if (t >= fwd[(N - 1) * ch + c]) {
                x = dmax;
            } else {
                int lo = 0, hi = N - 1;
                float y0, y1, x0, x1, frac;
                while (hi - lo > 1) {
                    int mid = (lo + hi) / 2;
                    if (fwd[mid * ch + c] <= t) lo = mid; else hi = mid;
                }
                y0 = fwd[lo * ch + c]; y1 = fwd[hi * ch + c];
                x0 = dmin + (dmax - dmin) * (float)lo / (float)(N - 1);
                x1 = dmin + (dmax - dmin) * (float)hi / (float)(N - 1);
                frac = (y1 != y0) ? (t - y0) / (y1 - y0) : 0.0f;
                x = x0 + frac * (x1 - x0);
            }
            inv[j * ch + c] = x;
        }
    }
    if (!toc_op_list_own(list, inv)) {
        toc_free(&list->alloc, inv);
        return TOC_ERROR_OUT_OF_MEMORY;
    }
    lut->data = inv;
    lut->domain_min = omin;
    lut->domain_max = omax;
    lut->interp = TOC_INTERP_LINEAR;
    return TOC_SUCCESS;
}

toc_result toc_invert_op(toc_op_list *list, toc_op *op) {
    int i;
    switch (op->kind) {
        case TOC_OP_RANGE: {
            float ns[4], no[4];
            for (i = 0; i < 4; ++i) {
                float s = op->u.range.scale[i];
                ns[i] = (s != 0.0f) ? 1.0f / s : 1.0f;
                no[i] = -op->u.range.offset[i] * ns[i];
            }
            for (i = 0; i < 4; ++i) {
                float lo = op->u.range.min[i], hi = op->u.range.max[i];
                /* output bounds become input bounds; recompute via new map */
                op->u.range.scale[i] = ns[i];
                op->u.range.offset[i] = no[i];
                op->u.range.min[i] = lo; /* clamp range is symmetric for [0,1] */
                op->u.range.max[i] = hi;
            }
            { int t = op->u.range.clamp_lo; op->u.range.clamp_lo =
                  op->u.range.clamp_hi; op->u.range.clamp_hi = t; }
            return TOC_SUCCESS;
        }
        case TOC_OP_EXPONENT:
            for (i = 0; i < 4; ++i)
                op->u.exponent.e[i] =
                    op->u.exponent.e[i] != 0.0f ? 1.0f / op->u.exponent.e[i]
                                                : 1.0f;
            return TOC_SUCCESS;
        case TOC_OP_EXP_LINEAR:
            op->u.exp_linear.inverse = !op->u.exp_linear.inverse;
            return TOC_SUCCESS;
        case TOC_OP_LOG:
            op->u.log.inverse = !op->u.log.inverse;
            return TOC_SUCCESS;
        case TOC_OP_LOG_CAMERA:
            op->u.logcam.inverse = !op->u.logcam.inverse;
            return TOC_SUCCESS;
        case TOC_OP_MATRIX: {
            /* m is stored col-major; invert via row-major round-trip */
            float rm[16], inv[16], off[4], tmp[4];
            int r, c;
            for (r = 0; r < 4; ++r)
                for (c = 0; c < 4; ++c) rm[r * 4 + c] = op->u.matrix.m[c * 4 + r];
            if (!toc_inv4x4(rm, inv)) return TOC_ERROR_NONINVERTIBLE;
            for (i = 0; i < 4; ++i) off[i] = op->u.matrix.off[i];
            matvec4(inv, off, tmp);
            for (r = 0; r < 4; ++r) {
                for (c = 0; c < 4; ++c)
                    op->u.matrix.m[c * 4 + r] = inv[r * 4 + c];
                op->u.matrix.off[r] = -tmp[r];
            }
            return TOC_SUCCESS;
        }
        case TOC_OP_FIXEDFUNC:
            /* every style is an even/odd forward/inverse pair, so toggling the
             * low bit selects the inverse (PQ<->lin, HSV<->RGB, glow fwd<->inv). */
            op->u.fixedfunc.style ^= 1;
            return TOC_SUCCESS;
        case TOC_OP_CDL:
            /* ASC CDL is analytically invertible (saturation preserves luma). */
            op->u.cdl.inverse = !op->u.cdl.inverse;
            return TOC_SUCCESS;
        case TOC_OP_LUT1D:
            return lut1d_invert(list, &op->u.lut1d);
        case TOC_OP_LUT3D: /* tetrahedral inverse: hard, not implemented */
        default:
            return TOC_ERROR_NONINVERTIBLE;
    }
}

/* ---- recursive walk ------------------------------------------------------ */
static toc_result emit_to_ref(const toc_config *, toc_op_list *, const char *,
                              int);
static toc_result emit_from_ref(const toc_config *, toc_op_list *, const char *,
                                int);

static toc_result walk(const toc_config *cfg, toc_op_list *list,
                       const toc_node *node, int invert) {
    const char *tag;
    const toc_node *dir;
    int dir_inv = invert;
    toc_result rc;
    if (!node) return TOC_SUCCESS;
    tag = node->tag;
    dir = toc_node_map_get(node, "direction");
    if (dir) {
        const char *ds = toc_node_scalar(dir);
        if (ds && strcmp(ds, "inverse") == 0) dir_inv = !dir_inv;
    }
    if (!tag) return TOC_SUCCESS;

    if (strcmp(tag, "GroupTransform") == 0) {
        const toc_node *ch = toc_node_map_get(node, "children");
        size_t i;
        if (!ch || ch->kind != TOC_NODE_SEQ) return TOC_SUCCESS;
        if (!dir_inv) {
            for (i = 0; i < ch->n_items; ++i) {
                rc = walk(cfg, list, ch->items[i], 0);
                if (!TOC_OK(rc)) return rc;
            }
        } else {
            for (i = ch->n_items; i-- > 0;) {
                rc = walk(cfg, list, ch->items[i], 1);
                if (!TOC_OK(rc)) return rc;
            }
        }
        return TOC_SUCCESS;
    }
    if (strcmp(tag, "ColorSpaceTransform") == 0) {
        const char *src = toc_node_scalar(toc_node_map_get(node, "src"));
        const char *dst = toc_node_scalar(toc_node_map_get(node, "dst"));
        if (!src || !dst) return TOC_ERROR_PARSE;
        if (dir_inv) { const char *t = src; src = dst; dst = t; }
        rc = emit_to_ref(cfg, list, src, 0);
        if (!TOC_OK(rc)) return rc;
        return emit_from_ref(cfg, list, dst, 0);
    }
    if (strcmp(tag, "FileTransform") == 0) {
        const char *src = toc_node_scalar(toc_node_map_get(node, "src"));
        char *data = NULL;
        size_t len = 0;
        if (!src) return TOC_ERROR_PARSE;
        if (!cfg->reader) return TOC_ERROR_IO;
        rc = cfg->reader(cfg->reader_user, src, &cfg->alloc, &data, &len);
        if (!TOC_OK(rc)) return rc;
        rc = toc_load_lutfile(list, src, data, len, dir_inv);
        toc_free(&cfg->alloc, data);
        return rc;
    }
    if (strcmp(tag, "BuiltinTransform") == 0) {
        const char *style = toc_node_scalar(toc_node_map_get(node, "style"));
        if (!style) return TOC_ERROR_PARSE;
        return toc_builtin_expand(list, style, dir_inv);
    }
    if (strcmp(tag, "FixedFunctionTransform") == 0) {
        return toc_lower_fixedfunc(list, node, dir_inv);
    }
    return toc_lower_transform(cfg, list, node, dir_inv);
}

static toc_result emit_to_ref(const toc_config *cfg, toc_op_list *list,
                              const char *cs_name, int invert) {
    const char *name = toc_cfg_resolve_role(cfg, cs_name);
    const toc_node *cs = toc_cfg_find_colorspace(cfg, name);
    const toc_node *t;
    int needinv = 0;
    if (!cs) return TOC_ERROR_NOT_FOUND;
    if (toc_cfg_is_data(cs)) return TOC_SUCCESS;
    t = toc_cfg_cs_transform(cs, 1, &needinv);
    if (!t) return TOC_SUCCESS;
    return walk(cfg, list, t, invert ^ needinv);
}

static toc_result emit_from_ref(const toc_config *cfg, toc_op_list *list,
                                const char *cs_name, int invert) {
    const char *name = toc_cfg_resolve_role(cfg, cs_name);
    const toc_node *cs = toc_cfg_find_colorspace(cfg, name);
    const toc_node *t;
    int needinv = 0;
    if (!cs) return TOC_ERROR_NOT_FOUND;
    if (toc_cfg_is_data(cs)) return TOC_SUCCESS;
    t = toc_cfg_cs_transform(cs, 0, &needinv);
    if (!t) return TOC_SUCCESS;
    return walk(cfg, list, t, invert ^ needinv);
}

/* ---- public builders ----------------------------------------------------- */
static toc_op_list *new_list(const toc_allocator *a) {
    toc_op_list *l = (toc_op_list *)toc_malloc(a, sizeof(*l));
    if (!l) return NULL;
    memset(l, 0, sizeof(*l));
    l->alloc = *a;
    return l;
}

toc_result toc_processor_from_colorspaces(const toc_config *cfg, const char *src,
                                          const char *dst,
                                          const toc_allocator *a,
                                          toc_op_list **out) {
    toc_op_list *list;
    toc_result rc;
    if (!cfg || !src || !dst || !out) return TOC_ERROR_INVALID_ARGUMENT;
    if (!a) a = toc_default_allocator();
    *out = NULL;
    list = new_list(a);
    if (!list) return TOC_ERROR_OUT_OF_MEMORY;
    rc = emit_to_ref(cfg, list, src, 0);
    if (TOC_OK(rc)) rc = emit_from_ref(cfg, list, dst, 0);
    if (!TOC_OK(rc)) {
        toc_op_list_free(list);
        return rc;
    }
    *out = list;
    return TOC_SUCCESS;
}

toc_result toc_processor_from_display_view(const toc_config *cfg,
                                           const char *src_cs,
                                           const char *display, const char *view,
                                           const toc_allocator *a,
                                           toc_op_list **out) {
    const toc_node *d, *views, *vnode;
    const char *view_cs, *view_vt, *view_dc;
    size_t i;
    toc_op_list *list;
    toc_result rc;
    if (!cfg || !src_cs || !display || !view || !out)
        return TOC_ERROR_INVALID_ARGUMENT;
    *out = NULL;
    d = toc_node_map_get(cfg->root, "displays");
    views = d ? toc_node_map_get(d, display) : NULL;
    if (!views || views->kind != TOC_NODE_SEQ) return TOC_ERROR_NOT_FOUND;
    vnode = NULL;
    for (i = 0; i < views->n_items; ++i) {
        const char *vn = toc_node_scalar(toc_node_map_get(views->items[i],
                                                          "name"));
        if (vn && strcmp(vn, view) == 0) { vnode = views->items[i]; break; }
    }
    if (!vnode) return TOC_ERROR_NOT_FOUND;
    /* Determine view type, build pipeline.
     * Simple view:  src -> reference -> [looks] -> display_colorspace
     * VT view:      src -> reference -> [looks] -> view_transform -> [display_colorspace] */
    view_cs = toc_node_scalar(toc_node_map_get(vnode, "colorspace"));
    view_vt = toc_node_scalar(toc_node_map_get(vnode, "view_transform"));
    if (!view_cs && !view_vt) return TOC_ERROR_UNSUPPORTED;
    if (!a) a = toc_default_allocator();
    list = new_list(a);
    if (!list) return TOC_ERROR_OUT_OF_MEMORY;
    /* 1. src -> reference */
    rc = emit_to_ref(cfg, list, src_cs, 0);
    if (!TOC_OK(rc)) goto fail;
    /* 2. apply looks (comma-separated) in order */
    {
        const char *lk_names[8];
        int nlk = toc_cfg_view_looks(vnode, lk_names, 8);
        int li;
        for (li = 0; li < nlk; ++li) {
            const toc_node *lk = toc_cfg_find_look(cfg, lk_names[li]);
            const toc_node *tf, *ps;
            if (!lk) { rc = TOC_ERROR_NOT_FOUND; goto fail; }
            ps = toc_node_map_get(lk, "process_space");
            if (ps) {
                const char *ps_name = toc_node_scalar(ps);
                if (ps_name) {
                    rc = emit_from_ref(cfg, list, ps_name, 0);
                    if (!TOC_OK(rc)) goto fail;
                    tf = toc_node_map_get(lk, "transform");
                    if (tf) { rc = walk(cfg, list, tf, 0); if (!TOC_OK(rc)) goto fail; }
                    rc = emit_to_ref(cfg, list, ps_name, 0);
                    if (!TOC_OK(rc)) goto fail;
                    continue;
                }
            }
            tf = toc_node_map_get(lk, "transform");
            if (tf) { rc = walk(cfg, list, tf, 0); if (!TOC_OK(rc)) goto fail; }
        }
    }
    /* 3. apply view_transform (if present) */
    if (view_vt) {
        const toc_node *vt = toc_cfg_find_view_transform(cfg, view_vt);
        const toc_node *tf;
        if (!vt) { rc = TOC_ERROR_NOT_FOUND; goto fail; }
        tf = toc_node_map_get(vt, "from_reference");
        if (!tf) { rc = TOC_ERROR_UNSUPPORTED; goto fail; }
        rc = walk(cfg, list, tf, 0);
        if (!TOC_OK(rc)) goto fail;
    }
    /* 4. display colorspace (view_cs for simple, display_colorspace for VT views) */
    view_dc = view_cs ? view_cs
                      : toc_node_scalar(toc_node_map_get(vnode, "display_colorspace"));
    if (view_dc) {
        rc = emit_from_ref(cfg, list, view_dc, 0);
        if (!TOC_OK(rc)) goto fail;
    }
    *out = list;
    return TOC_SUCCESS;
fail:
    toc_op_list_free(list);
    return rc;
}
