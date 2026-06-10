/*
 * tocio - host test harness (ASan/UBSan). Grows with each phase.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            g_pass++;                                                           \
        } else {                                                                \
            g_fail++;                                                           \
            printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);            \
        }                                                                       \
    } while (0)

static int approx(float a, float b, float eps) {
    float d = a - b;
    if (d < 0) d = -d;
    return d <= eps;
}

/* ---- math vs libm -------------------------------------------------------- */
static void test_math(void) {
    int i;
    int ok_log = 1, ok_exp = 1, ok_pow = 1, ok_sqrt = 1;
    for (i = 1; i < 400; ++i) {
        float x = (float)i * 0.05f;
        if (!approx(toc_log2f(x), log2f(x), 3e-5f * (1.0f + x))) ok_log = 0;
        if (!approx(toc_exp2f((float)i * 0.03f - 5.0f),
                    exp2f((float)i * 0.03f - 5.0f),
                    1e-5f * exp2f((float)i * 0.03f - 5.0f)))
            ok_exp = 0;
        if (!approx(toc_powf(x, 1.0f / 2.4f), powf(x, 1.0f / 2.4f),
                    2e-5f * (1.0f + x)))
            ok_pow = 0;
        if (!approx(toc_sqrtf(x), sqrtf(x), 2e-5f * (1.0f + x))) ok_sqrt = 0;
    }
    CHECK(ok_log, "toc_log2f vs libm");
    CHECK(ok_exp, "toc_exp2f vs libm");
    CHECK(ok_pow, "toc_powf vs libm");
    CHECK(ok_sqrt, "toc_sqrtf vs libm");
    CHECK(approx(toc_raisef(10.0f, 3.0f), 1000.0f, 0.5f), "toc_raisef(10,3)");
}

/* ---- 4x4 inverse --------------------------------------------------------- */
static void test_inv4x4(void) {
    float m[16] = {2, 0, 0, 1, 0, 3, 0, 2, 0, 0, 4, 3, 0, 0, 0, 1};
    float inv[16];
    int r, c, k, ok = 1;
    CHECK(toc_inv4x4(m, inv), "inv4x4 nonsingular");
    for (r = 0; r < 4; ++r)
        for (c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (k = 0; k < 4; ++k) s += m[r * 4 + k] * inv[k * 4 + c];
            if (!approx(s, r == c ? 1.0f : 0.0f, 1e-5f)) ok = 0;
        }
    CHECK(ok, "M * inv(M) == I");
}

/* ---- strbuf: hex-float round-trips bit-exact; decfloat parses close ------ */
static void test_strbuf(void) {
    float vals[] = {0.0f,    1.0f,   0.5f,    2.4f,     -3.25f,
                    0.0031308f, 65504.0f, 1.0e-6f, 1234.5678f, -0.0f};
    int i, ok_hex = 1, ok_dec = 1;
    for (i = 0; i < (int)(sizeof(vals) / sizeof(vals[0])); ++i) {
        toc_sb sb;
        char *s;
        float back;
        toc_sb_init(&sb, NULL);
        toc_sb_hexfloat(&sb, vals[i]);
        s = sb.buf;
        back = strtof(s, NULL);
        /* hex float must be bit-exact */
        if (memcmp(&back, &vals[i], sizeof(float)) != 0 &&
            !(vals[i] == 0.0f && back == 0.0f))
            ok_hex = 0;
        toc_sb_free(&sb);

        toc_sb_init(&sb, NULL);
        toc_sb_decfloat(&sb, vals[i]);
        s = sb.buf;
        back = strtof(s, NULL);
        if (!approx(back, vals[i], 1e-4f * (1.0f + fabsf(vals[i])))) ok_dec = 0;
        toc_sb_free(&sb);
    }
    CHECK(ok_hex, "hexfloat round-trips bit-exact");
    CHECK(ok_dec, "decfloat round-trips within 1e-4 rel");
    {
        toc_sb sb;
        toc_sb_init(&sb, NULL);
        toc_sb_int(&sb, -123456);
        CHECK(strcmp(sb.buf, "-123456") == 0, "sb_int");
        toc_sb_free(&sb);
    }
}

/* ---- float parser -------------------------------------------------------- */
static void test_parse(void) {
    const char *s = "  -1.5e2 , 0.25  3";
    const char *p = s, *end = s + strlen(s);
    float a, b, c;
    long n;
    CHECK(toc_parse_float(&p, end, &a) && approx(a, -150.0f, 1e-3f), "parse -1.5e2");
    CHECK(toc_parse_float(&p, end, &b) && approx(b, 0.25f, 1e-6f), "parse 0.25");
    CHECK(toc_parse_float(&p, end, &c) && approx(c, 3.0f, 1e-6f), "parse 3");
    p = "42 x";
    end = p + 4;
    CHECK(toc_parse_int(&p, end, &n) && n == 42, "parse_int 42");
}

/* ---- interpreter --------------------------------------------------------- */
static toc_op_list *newlist(void) {
    const toc_allocator *a = toc_default_allocator();
    toc_op_list *l = (toc_op_list *)malloc(sizeof(*l));
    memset(l, 0, sizeof(*l));
    l->alloc = *a;
    return l;
}

static void test_interp(void) {
    /* matrix: 2x scale + offset 0.1 on RGB */
    {
        toc_op_list *l = newlist();
        toc_op *m = toc_op_list_push(l, TOC_OP_MATRIX);
        float px[4] = {0.2f, 0.3f, 0.4f, 1.0f};
        m->u.matrix.m[0] = m->u.matrix.m[5] = m->u.matrix.m[10] = 2.0f;
        m->u.matrix.m[15] = 1.0f;
        m->u.matrix.off[0] = m->u.matrix.off[1] = m->u.matrix.off[2] = 0.1f;
        toc_apply(l, px, 1, 4);
        CHECK(approx(px[0], 0.5f, 1e-6f) && approx(px[1], 0.7f, 1e-6f) &&
                  approx(px[2], 0.9f, 1e-6f) && approx(px[3], 1.0f, 1e-6f),
              "matrix scale+offset");
        toc_op_list_free(l);
    }
    /* range: scale 2, clamp [0,1] */
    {
        toc_op_list *l = newlist();
        toc_op *r = toc_op_list_push(l, TOC_OP_RANGE);
        float px[3] = {0.3f, 0.7f, -0.1f};
        int c;
        for (c = 0; c < 4; ++c) {
            r->u.range.scale[c] = 2.0f;
            r->u.range.min[c] = 0.0f;
            r->u.range.max[c] = 1.0f;
        }
        r->u.range.clamp_lo = r->u.range.clamp_hi = 1;
        toc_apply(l, px, 1, 3);
        CHECK(approx(px[0], 0.6f, 1e-6f) && approx(px[1], 1.0f, 1e-6f) &&
                  approx(px[2], 0.0f, 1e-6f),
              "range scale+clamp");
        toc_op_list_free(l);
    }
    /* exponent 2.2 */
    {
        toc_op_list *l = newlist();
        toc_op *e = toc_op_list_push(l, TOC_OP_EXPONENT);
        float px[3] = {0.5f, 0.5f, 0.5f};
        int c;
        for (c = 0; c < 4; ++c) e->u.exponent.e[c] = 2.2f;
        toc_apply(l, px, 1, 3);
        CHECK(approx(px[0], powf(0.5f, 2.2f), 1e-4f), "exponent 2.2");
        toc_op_list_free(l);
    }
    /* log2 forward then inverse round-trips */
    {
        toc_op_list *l = newlist();
        toc_op *lg = toc_op_list_push(l, TOC_OP_LOG);
        toc_op *ag = toc_op_list_push(l, TOC_OP_LOG);
        float px[3] = {0.18f, 0.5f, 1.0f}, orig[3];
        int c;
        memcpy(orig, px, sizeof(px));
        for (c = 0; c < 3; ++c) {
            lg->u.log.log_slope[c] = ag->u.log.log_slope[c] = 1.0f;
            lg->u.log.lin_slope[c] = ag->u.log.lin_slope[c] = 1.0f;
        }
        lg->u.log.base = ag->u.log.base = 2.0f;
        lg->u.log.inverse = 0;
        ag->u.log.inverse = 1;
        toc_apply(l, px, 1, 3);
        CHECK(approx(px[0], orig[0], 1e-4f) && approx(px[1], orig[1], 1e-4f),
              "log forward+inverse round-trip");
        toc_op_list_free(l);
    }
    /* lut1d identity (shared) */
    {
        toc_op_list *l = newlist();
        toc_op *op = toc_op_list_push(l, TOC_OP_LUT1D);
        static const float tbl[4] = {0.0f, 1.0f / 3, 2.0f / 3, 1.0f};
        float px[3] = {0.5f, 0.25f, 0.9f};
        op->u.lut1d.length = 4;
        op->u.lut1d.channels = 1;
        op->u.lut1d.domain_min = 0.0f;
        op->u.lut1d.domain_max = 1.0f;
        op->u.lut1d.data = tbl;
        op->u.lut1d.interp = TOC_INTERP_LINEAR;
        toc_apply(l, px, 1, 3);
        CHECK(approx(px[0], 0.5f, 1e-5f) && approx(px[1], 0.25f, 1e-5f) &&
                  approx(px[2], 0.9f, 1e-5f),
              "lut1d identity");
        toc_op_list_free(l);
    }
    /* lut3d identity (trilinear + tetrahedral) */
    {
        int N = 2, ir, ig, ib, t;
        static float data[2 * 2 * 2 * 3];
        for (ib = 0; ib < N; ++ib)
            for (ig = 0; ig < N; ++ig)
                for (ir = 0; ir < N; ++ir) {
                    size_t idx = (((size_t)ib * N + ig) * N + ir) * 3;
                    data[idx + 0] = (float)ir;
                    data[idx + 1] = (float)ig;
                    data[idx + 2] = (float)ib;
                }
        for (t = 0; t < 2; ++t) {
            toc_op_list *l = newlist();
            toc_op *op = toc_op_list_push(l, TOC_OP_LUT3D);
            float px[3] = {0.2f, 0.6f, 0.9f};
            op->u.lut3d.size = N;
            op->u.lut3d.data = data;
            op->u.lut3d.domain_min[0] = op->u.lut3d.domain_min[1] =
                op->u.lut3d.domain_min[2] = 0.0f;
            op->u.lut3d.domain_max[0] = op->u.lut3d.domain_max[1] =
                op->u.lut3d.domain_max[2] = 1.0f;
            op->u.lut3d.interp = t ? TOC_INTERP_TETRAHEDRAL : TOC_INTERP_TRILINEAR;
            toc_apply(l, px, 1, 3);
            CHECK(approx(px[0], 0.2f, 1e-5f) && approx(px[1], 0.6f, 1e-5f) &&
                      approx(px[2], 0.9f, 1e-5f),
                  t ? "lut3d identity tetrahedral" : "lut3d identity trilinear");
            toc_op_list_free(l);
        }
    }
}

/* ---- YAML parser --------------------------------------------------------- */
static const char *g_cfg =
    "ocio_profile_version: 2\n"
    "roles:\n"
    "  scene_linear: lin\n"
    "  default: raw\n"
    "colorspaces:\n"
    "  - !<ColorSpace>\n"
    "    name: lin\n"
    "    isdata: false\n"
    "  - !<ColorSpace>\n"
    "    name: raw\n"
    "    isdata: true\n"
    "    to_reference: !<MatrixTransform> {matrix: [2,0,0,0, 0,2,0,0, 0,0,2,0, "
    "0,0,0,1]}\n"
    "displays:\n"
    "  sRGB:\n"
    "    - !<View> {name: Raw, colorspace: raw}\n";

static void test_yaml(void) {
    toc_arena ar;
    toc_node *root = NULL;
    int err = 0;
    toc_result rc;
    toc_arena_init(&ar, toc_default_allocator());
    rc = toc_yaml_parse(g_cfg, strlen(g_cfg), &ar, &root, &err);
    CHECK(TOC_OK(rc), "yaml parse ok");
    if (TOC_OK(rc)) {
        const toc_node *roles = toc_node_map_get(root, "roles");
        const toc_node *cs = toc_node_map_get(root, "colorspaces");
        const toc_node *disp = toc_node_map_get(root, "displays");
        CHECK(roles && roles->kind == TOC_NODE_MAP, "roles is a map");
        CHECK(roles && strcmp(toc_node_scalar(toc_node_map_get(roles,
                                                               "scene_linear")),
                              "lin") == 0,
              "role scene_linear=lin");
        CHECK(cs && cs->kind == TOC_NODE_SEQ && cs->n_items == 2,
              "2 colorspaces");
        if (cs && cs->n_items == 2) {
            const toc_node *c0 = cs->items[0];
            const toc_node *c1 = cs->items[1];
            const toc_node *t1, *mat;
            CHECK(c0->tag && strcmp(c0->tag, "ColorSpace") == 0, "cs0 tag");
            CHECK(strcmp(toc_node_scalar(toc_node_map_get(c0, "name")), "lin") ==
                      0,
                  "cs0 name=lin");
            t1 = toc_node_map_get(c1, "to_reference");
            CHECK(t1 && t1->tag && strcmp(t1->tag, "MatrixTransform") == 0,
                  "cs1 to_reference tag MatrixTransform");
            mat = t1 ? toc_node_map_get(t1, "matrix") : NULL;
            CHECK(mat && mat->kind == TOC_NODE_SEQ && mat->n_items == 16,
                  "matrix has 16 items");
            CHECK(mat && strcmp(toc_node_scalar(mat->items[0]), "2") == 0,
                  "matrix[0]=2");
        }
        {
            const toc_node *srgb = disp ? toc_node_map_get(disp, "sRGB") : NULL;
            const toc_node *v0;
            CHECK(srgb && srgb->kind == TOC_NODE_SEQ && srgb->n_items == 1,
                  "display sRGB has 1 view");
            v0 = srgb && srgb->n_items ? srgb->items[0] : NULL;
            CHECK(v0 && v0->tag && strcmp(v0->tag, "View") == 0, "view tag");
            CHECK(v0 && strcmp(toc_node_scalar(toc_node_map_get(v0, "name")),
                               "Raw") == 0,
                  "view name=Raw");
        }
    }
    toc_arena_free(&ar);
}

/* ---- processor (end to end) ---------------------------------------------- */
static const char *g_proc_cfg =
    "ocio_profile_version: 2\n"
    "roles:\n"
    "  scene_linear: lin\n"
    "colorspaces:\n"
    "  - !<ColorSpace>\n"
    "    name: lin\n"
    "    isdata: false\n"
    "  - !<ColorSpace>\n"
    "    name: srgb\n"
    "    isdata: false\n"
    "    from_reference: !<GroupTransform>\n"
    "      children:\n"
    "        - !<MatrixTransform> {matrix: [2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,1]}\n"
    "        - !<ExponentTransform> {value: [2, 2, 2, 1]}\n";

static void test_processor(void) {
    toc_config *cfg = NULL;
    toc_op_list *fwd = NULL, *rev = NULL;
    toc_result rc;
    rc = toc_config_parse(g_proc_cfg, strlen(g_proc_cfg), NULL, &cfg);
    CHECK(TOC_OK(rc), "config parse");
    if (!TOC_OK(rc)) return;
    CHECK(toc_config_num_colorspaces(cfg) == 2, "2 colorspaces");
    CHECK(toc_config_role(cfg, "scene_linear") &&
              strcmp(toc_config_role(cfg, "scene_linear"), "lin") == 0,
          "role lookup");

    /* lin -> srgb : scale by 2 then ^2.  0.3 -> 0.6 -> 0.36 */
    rc = toc_processor_from_colorspaces(cfg, "lin", "srgb", NULL, &fwd);
    CHECK(TOC_OK(rc) && fwd && fwd->count == 2, "build lin->srgb (2 ops)");
    if (TOC_OK(rc)) {
        float px[3] = {0.3f, 0.3f, 0.3f};
        toc_apply(fwd, px, 1, 3);
        CHECK(approx(px[0], 0.36f, 1e-4f), "lin->srgb forward value");
    }
    /* srgb -> lin : inverse (sqrt then /2). 0.36 -> 0.6 -> 0.3 */
    rc = toc_processor_from_colorspaces(cfg, "srgb", "lin", NULL, &rev);
    CHECK(TOC_OK(rc) && rev, "build srgb->lin (inverse)");
    if (TOC_OK(rc)) {
        float px[3] = {0.36f, 0.36f, 0.36f};
        toc_apply(rev, px, 1, 3);
        CHECK(approx(px[0], 0.3f, 1e-3f), "srgb->lin inverse round-trips");
    }
    /* role used as endpoint */
    {
        toc_op_list *r2 = NULL;
        rc = toc_processor_from_colorspaces(cfg, "scene_linear", "srgb", NULL,
                                            &r2);
        CHECK(TOC_OK(rc) && r2 && r2->count == 2, "role endpoint resolves");
        if (r2) toc_op_list_free(r2);
    }
    if (fwd) toc_op_list_free(fwd);
    if (rev) toc_op_list_free(rev);
    toc_config_free(cfg);
}

/* ---- file LUTs ----------------------------------------------------------- */
static void load_and_check(const char *name, const char *txt, float in,
                           float expect, const char *msg) {
    toc_op_list *l = newlist();
    toc_result rc = toc_load_lutfile(l, name, txt, strlen(txt), 0);
    if (TOC_OK(rc)) {
        float px[3] = {in, in, in};
        toc_apply(l, px, 1, 3);
        CHECK(approx(px[0], expect, 1e-4f), msg);
    } else {
        CHECK(0, msg);
    }
    toc_op_list_free(l);
}

static void test_lutfile(void) {
    load_and_check("id.cube",
                   "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n", 0.5f, 0.5f,
                   ".cube 1D identity");
    load_and_check(
        "id3.cube",
        "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        0.3f, 0.3f, ".cube 3D identity");
    load_and_check("id.spi1d",
                   "Version 1\nFrom 0.0 1.0\nLength 2\nComponents 1\n{\n0.0\n"
                   "1.0\n}\n",
                   0.5f, 0.5f, ".spi1d identity");
    /* .spi3d identity N=2 (8 entries) */
    load_and_check(
        "id.spi3d",
        "SPILUT 1.0\n3 3\n2 2 2\n"
        "0 0 0  0 0 0\n0 0 1  0 0 1\n0 1 0  0 1 0\n0 1 1  0 1 1\n"
        "1 0 0  1 0 0\n1 0 1  1 0 1\n1 1 0  1 1 0\n1 1 1  1 1 1\n",
        0.4f, 0.4f, ".spi3d identity");
    /* a non-identity .spi1d: doubles input (clamped at domain top) */
    load_and_check("dbl.spi1d",
                   "Version 1\nFrom 0.0 1.0\nLength 3\nComponents 1\n{\n0.0\n"
                   "0.5\n1.0\n}\n",
                   0.25f, 0.25f, ".spi1d ramp midpoint");
}

/* file reader returning a baked identity .spi1d for any name */
static toc_result test_reader(void *user, const char *name,
                              const toc_allocator *a, char **data, size_t *len) {
    static const char *spi =
        "Version 1\nFrom 0.0 1.0\nLength 2\nComponents 3\n{\n"
        "0 0 0\n1 1 1\n}\n";
    size_t n = strlen(spi);
    char *buf = (char *)toc_malloc(a, n + 1);
    (void)user;
    (void)name;
    if (!buf) return TOC_ERROR_OUT_OF_MEMORY;
    memcpy(buf, spi, n + 1);
    *data = buf;
    *len = n;
    return TOC_SUCCESS;
}

static void test_filetransform(void) {
    static const char *cfg_txt =
        "ocio_profile_version: 2\n"
        "colorspaces:\n"
        "  - !<ColorSpace>\n"
        "    name: lin\n"
        "    isdata: false\n"
        "  - !<ColorSpace>\n"
        "    name: lut\n"
        "    isdata: false\n"
        "    from_reference: !<FileTransform> {src: id.spi1d}\n";
    toc_config *cfg = NULL;
    toc_op_list *ops = NULL;
    toc_result rc = toc_config_parse(cfg_txt, strlen(cfg_txt), NULL, &cfg);
    CHECK(TOC_OK(rc), "filetransform config parse");
    if (!TOC_OK(rc)) return;
    toc_config_set_file_reader(cfg, test_reader, NULL);
    rc = toc_processor_from_colorspaces(cfg, "lin", "lut", NULL, &ops);
    CHECK(TOC_OK(rc) && ops && ops->count == 1, "FileTransform -> 1 LUT op");
    if (TOC_OK(rc)) {
        float px[3] = {0.42f, 0.1f, 0.9f};
        toc_apply(ops, px, 1, 3);
        CHECK(approx(px[0], 0.42f, 1e-4f), "FileTransform identity applies");
        toc_op_list_free(ops);
    }
    toc_config_free(cfg);
}

/* ---- AOT-C codegen: interpreter == compiled emitted-C --------------------- */
typedef void (*apply_fn)(float *, size_t);

static void build_pipeline(toc_op_list *l) {
    toc_op *m = toc_op_list_push(l, TOC_OP_MATRIX);
    toc_op *e = toc_op_list_push(l, TOC_OP_EXPONENT);
    toc_op *r = toc_op_list_push(l, TOC_OP_RANGE);
    int c;
    /* a non-trivial mixing matrix */
    static const float rm[16] = {0.6f, 0.3f, 0.1f, 0.0f, 0.2f, 0.7f, 0.1f, 0.0f,
                                 0.1f, 0.2f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    int i, j;
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) m->u.matrix.m[j * 4 + i] = rm[i * 4 + j];
        m->u.matrix.off[i] = 0.0f;
    }
    for (c = 0; c < 4; ++c) e->u.exponent.e[c] = (c < 3) ? 2.2f : 1.0f;
    for (c = 0; c < 4; ++c) {
        r->u.range.scale[c] = 1.0f;
        r->u.range.min[c] = 0.0f;
        r->u.range.max[c] = 1.0f;
    }
    r->u.range.clamp_lo = r->u.range.clamp_hi = 1;
}

static void test_codegen_c(void) {
    toc_op_list *l = newlist();
    char *src = NULL;
    size_t len = 0;
    toc_result rc;
    FILE *f;
    void *h;
    apply_fn fn;
    int ok = 1, i;
    build_pipeline(l);
    rc = toc_emit_c(l, NULL, NULL, &src, &len);
    CHECK(TOC_OK(rc) && src, "emit C source");
    if (!TOC_OK(rc)) { toc_op_list_free(l); return; }
    f = fopen("build/toc_gen.c", "w");
    if (f) { fwrite(src, 1, len, f); fclose(f); }
    if (system("cc -O2 -fPIC -shared build/toc_gen.c -o build/toc_gen.so") != 0) {
        CHECK(0, "compile emitted C");
        free(src);
        toc_op_list_free(l);
        return;
    }
    h = dlopen("./build/toc_gen.so", RTLD_NOW);
    CHECK(h != NULL, "dlopen emitted .so");
    if (h) {
        fn = (apply_fn)dlsym(h, "tocio_apply");
        CHECK(fn != NULL, "dlsym tocio_apply");
        if (fn) {
            for (i = 0; i < 64; ++i) {
                float a[4], b[4];
                int c;
                for (c = 0; c < 4; ++c) {
                    float v = (float)((i * 37 + c * 11) % 100) / 99.0f;
                    a[c] = b[c] = v;
                }
                toc_apply(l, a, 1, 4);
                fn(b, 1);
                for (c = 0; c < 4; ++c)
                    if (!approx(a[c], b[c], 1e-5f)) ok = 0;
            }
            CHECK(ok, "interpreter == compiled emitted-C (64 pixels)");
        }
        dlclose(h);
    }
    free(src);
    toc_op_list_free(l);
}

/* ---- GLSL codegen -------------------------------------------------------- */
static int glsl_validate(const char *body, const char *version, int es) {
    /* wrap OCIOMain into a complete fragment shader and run glslangValidator */
    FILE *f = fopen("build/toc_gen.frag", "w");
    int rc;
    if (!f) return -1;
    fprintf(f, "%s", body);
    /* append a trivial main that uses OCIOMain so it's a valid stage */
    fprintf(f, "out vec4 fragColor;\nvoid main(){ fragColor = OCIOMain(vec4(0.5)); }\n");
    fclose(f);
    (void)version;
    (void)es;
    rc = system("glslangValidator -S frag build/toc_gen.frag >/dev/null 2>&1");
    return rc;
}

static void test_codegen_glsl(void) {
    toc_glsl_target tgts[3] = {TOC_GLSL_ES30, TOC_GLSL_330, TOC_GLSL_VULKAN450};
    const char *names[3] = {"ES3.0", "GLSL330", "Vulkan450"};
    int t;
    /* pipeline with a matrix + a 3D LUT to exercise samplers */
    static float cube[2 * 2 * 2 * 3];
    int ir, ig, ib;
    for (ib = 0; ib < 2; ++ib)
        for (ig = 0; ig < 2; ++ig)
            for (ir = 0; ir < 2; ++ir) {
                size_t idx = (((size_t)ib * 2 + ig) * 2 + ir) * 3;
                cube[idx + 0] = (float)ir;
                cube[idx + 1] = (float)ig;
                cube[idx + 2] = (float)ib;
            }
    for (t = 0; t < 3; ++t) {
        toc_op_list *l = newlist();
        toc_op *m, *lut;
        toc_shader sh;
        toc_result rc;
        int i, j;
        static const float rm[16] = {0.6f, 0.3f, 0.1f, 0, 0.2f, 0.7f, 0.1f, 0,
                                     0.1f, 0.2f, 0.7f, 0, 0, 0, 0, 1};
        m = toc_op_list_push(l, TOC_OP_MATRIX);
        for (i = 0; i < 4; ++i) {
            for (j = 0; j < 4; ++j) m->u.matrix.m[j * 4 + i] = rm[i * 4 + j];
            m->u.matrix.off[i] = 0.0f;
        }
        lut = toc_op_list_push(l, TOC_OP_LUT3D);
        lut->u.lut3d.size = 2;
        lut->u.lut3d.data = cube;
        lut->u.lut3d.domain_max[0] = lut->u.lut3d.domain_max[1] =
            lut->u.lut3d.domain_max[2] = 1.0f;
        lut->u.lut3d.interp = TOC_INTERP_TRILINEAR;

        rc = toc_emit_glsl(l, tgts[t], NULL, &sh);
        CHECK(TOC_OK(rc) && sh.source, names[t]);
        if (TOC_OK(rc)) {
            CHECK(sh.num_textures == 1 && sh.textures[0].dim == TOC_TEX_3D &&
                      sh.textures[0].width == 2,
                  "glsl 3D texture descriptor");
            CHECK(strstr(sh.source, "OCIOMain") != NULL, "glsl has OCIOMain");
            CHECK(strstr(sh.source, "sampler3D") != NULL, "glsl has sampler3D");
            {
                int vrc = glsl_validate(sh.source, names[t], t == 0);
                if (vrc == 0)
                    CHECK(1, "glslangValidator accepts shader");
                else
                    printf("  note: glslangValidator skipped/failed for %s\n",
                           names[t]);
            }
            toc_shader_free(&sh);
        }
        toc_op_list_free(l);
    }
}

/* ---- SIMD parity (scalar vs SSE2 vs AVX2) -------------------------------- */
static void test_simd_parity(void) {
    enum { NP = 257 };
    static float base[NP * 4], s0[NP * 4], s1[NP * 4], s2[NP * 4];
    toc_op_list *l = newlist();
    int i, c;
    build_pipeline(l); /* matrix + exponent + range */
    for (i = 0; i < NP * 4; ++i)
        base[i] = (float)((i * 41) % 200 - 50) / 100.0f; /* incl negatives */
    memcpy(s0, base, sizeof(base));
    memcpy(s1, base, sizeof(base));
    memcpy(s2, base, sizeof(base));
    toc_simd_force(0);
    toc_apply(l, s0, NP, 4);
    toc_simd_force(1);
    toc_apply(l, s1, NP, 4);
    toc_simd_force(2);
    toc_apply(l, s2, NP, 4);
    {
        int ok1 = 1, ok2 = 1;
        for (i = 0; i < NP * 4; ++i) {
            if (s0[i] != s1[i]) ok1 = 0;       /* SSE2 must be bit-exact */
            if (!approx(s0[i], s2[i], 1e-6f)) ok2 = 0;
        }
        CHECK(ok1, "SSE2 batch == scalar (bit-exact)");
        CHECK(ok2, "AVX2 batch == scalar");
    }
    (void)c;
    toc_simd_force(2);
    toc_op_list_free(l);
}

/* ---- builtins + fixedfunc + logcamera ------------------------------------ */
static const char *g_builtin_cfg =
    "ocio_profile_version: 2\n"
    "colorspaces:\n"
    "  - !<ColorSpace>\n"
    "    name: ap0\n"
    "    isdata: false\n"
    "  - !<ColorSpace>\n"
    "    name: acescg\n"
    "    isdata: false\n"
    "    to_reference: !<BuiltinTransform> {style: ACEScg_to_ACES2065-1}\n"
    "  - !<ColorSpace>\n"
    "    name: acescct\n"
    "    isdata: false\n"
    "    to_reference: !<BuiltinTransform> {style: ACEScct_to_ACES2065-1}\n"
    "  - !<ColorSpace>\n"
    "    name: surround\n"
    "    isdata: false\n"
    "    from_reference: !<FixedFunctionTransform> {style: Rec2100Surround, "
    "params: [0.78]}\n";

static void roundtrip(toc_config *cfg, const char *a, const char *b,
                      const char *msg) {
    toc_op_list *f = NULL, *r = NULL;
    float px[3] = {0.18f, 0.35f, 0.09f}, orig[3];
    memcpy(orig, px, sizeof(px));
    if (TOC_OK(toc_processor_from_colorspaces(cfg, a, b, NULL, &f)) &&
        TOC_OK(toc_processor_from_colorspaces(cfg, b, a, NULL, &r))) {
        toc_apply(f, px, 1, 3);
        toc_apply(r, px, 1, 3);
        CHECK(approx(px[0], orig[0], 2e-3f) && approx(px[1], orig[1], 2e-3f) &&
                  approx(px[2], orig[2], 2e-3f),
              msg);
    } else {
        CHECK(0, msg);
    }
    if (f) toc_op_list_free(f);
    if (r) toc_op_list_free(r);
}

static void test_builtins(void) {
    toc_config *cfg = NULL;
    toc_result rc =
        toc_config_parse(g_builtin_cfg, strlen(g_builtin_cfg), NULL, &cfg);
    CHECK(TOC_OK(rc), "builtin config parse");
    if (!TOC_OK(rc)) return;
    roundtrip(cfg, "acescg", "ap0", "ACEScg<->ACES2065-1 round-trip");
    roundtrip(cfg, "acescct", "ap0", "ACEScct<->ACES2065-1 round-trip (logcam)");
    /* FixedFunction builds + applies (surround darkens mid-grey for g<1) */
    {
        toc_op_list *ops = NULL;
        rc = toc_processor_from_colorspaces(cfg, "ap0", "surround", NULL, &ops);
        CHECK(TOC_OK(rc) && ops && ops->count == 1, "Rec2100Surround -> 1 op");
        if (TOC_OK(rc)) {
            float px[3] = {0.18f, 0.18f, 0.18f};
            toc_apply(ops, px, 1, 3);
            CHECK(px[0] > 0.0f && fabsf(px[0] - 0.18f) > 1e-3f,
                  "surround scales by Y^(g-1)");
            toc_op_list_free(ops);
        }
    }
    toc_config_free(cfg);
}

/* ---- x64 JIT: emitted machine code == interpreter ------------------------ */
static void test_jit(void) {
    toc_op_list *l = newlist();
    toc_jit *j = NULL;
    toc_result rc;
    int i, ok = 1;
    build_pipeline(l); /* matrix + exponent + range */
    rc = toc_jit_compile(l, 4, NULL, &j);
    if (rc == TOC_ERROR_UNSUPPORTED) {
        printf("  note: JIT unsupported on this target; skipping\n");
        toc_op_list_free(l);
        return;
    }
    CHECK(TOC_OK(rc) && j, "jit compile");
    if (TOC_OK(rc)) {
        toc_jit_fn fn = toc_jit_func(j);
        CHECK(fn != NULL, "jit func ptr");
        if (fn) {
            for (i = 0; i < 64; ++i) {
                float a[4], b[4];
                int c;
                for (c = 0; c < 4; ++c) {
                    float v = (float)((i * 53 + c * 17) % 200 - 50) / 100.0f;
                    a[c] = b[c] = v;
                }
                toc_apply(l, a, 1, 4); /* interpreter */
                fn(b, 1);              /* native code */
                for (c = 0; c < 4; ++c)
                    if (!approx(a[c], b[c], 1e-6f)) ok = 0;
            }
            CHECK(ok, "JIT == interpreter (64 pixels, matrix+exp+range)");
            /* multi-pixel run exercises the loop */
            {
                static float buf[16 * 4], ref[16 * 4];
                int n;
                for (n = 0; n < 16 * 4; ++n)
                    buf[n] = ref[n] = (float)(n % 7) * 0.1f;
                toc_apply(l, ref, 16, 4);
                fn(buf, 16);
                ok = 1;
                for (n = 0; n < 16 * 4; ++n)
                    if (!approx(buf[n], ref[n], 1e-6f)) ok = 0;
                CHECK(ok, "JIT loop over 16 pixels == interpreter");
            }
        }
        toc_jit_destroy(j);
    }
    /* inline SSE2 pow: exponent-only pipelines across several gammas */
    {
        float gammas[4] = {2.2f, 1.0f / 2.4f, 2.4f, 0.5f};
        int gi;
        for (gi = 0; gi < 4; ++gi) {
            toc_op_list *le = newlist();
            toc_op *e = toc_op_list_push(le, TOC_OP_EXPONENT);
            toc_jit *je = NULL;
            int c;
            for (c = 0; c < 4; ++c) e->u.exponent.e[c] = (c < 3) ? gammas[gi] : 1.0f;
            if (TOC_OK(toc_jit_compile(le, 4, NULL, &je)) && je) {
                int p;
                ok = 1;
                for (p = 0; p < 32; ++p) {
                    float a[4], b[4];
                    for (c = 0; c < 4; ++c) {
                        float v = (float)((p * 31 + c * 7) % 100) / 99.0f;
                        a[c] = b[c] = v;
                    }
                    toc_apply(le, a, 1, 4);
                    toc_jit_func(je)(b, 1);
                    for (c = 0; c < 4; ++c)
                        if (!approx(a[c], b[c], 1e-5f)) ok = 0;
                }
                CHECK(ok, "JIT inline pow == interpreter (gamma sweep)");
                toc_jit_destroy(je);
            }
            toc_op_list_free(le);
        }
    }
    /* AVX 2px path: pure matrix+range pipeline on an ODD pixel count so both
     * the 2-pixel main loop and the 1-pixel SSE tail execute. */
    {
        toc_op_list *la = newlist();
        toc_op *m1 = toc_op_list_push(la, TOC_OP_MATRIX);
        toc_op *m2 = toc_op_list_push(la, TOC_OP_MATRIX);
        toc_op *r = toc_op_list_push(la, TOC_OP_RANGE);
        static const float A[16] = {0.6f, 0.3f, 0.1f, 0, 0.2f, 0.7f, 0.1f, 0,
                                    0.1f, 0.2f, 0.7f, 0, 0, 0, 0, 1};
        static const float B[16] = {1.1f, -0.1f, 0, 0, -0.05f, 1.05f, 0, 0,
                                    0, 0, 1.0f, 0, 0, 0, 0, 1};
        int i, jx, c;
        toc_jit *ja = NULL;
        for (i = 0; i < 4; ++i)
            for (jx = 0; jx < 4; ++jx) {
                m1->u.matrix.m[jx * 4 + i] = A[i * 4 + jx];
                m2->u.matrix.m[jx * 4 + i] = B[i * 4 + jx];
            }
        for (c = 0; c < 4; ++c) {
            r->u.range.scale[c] = 1.0f; r->u.range.min[c] = 0.0f;
            r->u.range.max[c] = 1.0f;
        }
        r->u.range.clamp_lo = r->u.range.clamp_hi = 1;
        rc = toc_jit_compile(la, 4, NULL, &ja);
        if (TOC_OK(rc) && ja) {
            enum { NP = 17 };
            static float buf[NP * 4], ref[NP * 4];
            int n;
            for (n = 0; n < NP * 4; ++n)
                buf[n] = ref[n] = (float)((n * 29) % 150 - 30) / 100.0f;
            toc_apply(la, ref, NP, 4);
            toc_jit_func(ja)(buf, NP);
            ok = 1;
            for (n = 0; n < NP * 4; ++n)
                if (!approx(buf[n], ref[n], 1e-6f)) ok = 0;
            CHECK(ok, "JIT AVX 2px (matrix+matrix+range, 17 px) == interpreter");
            toc_jit_destroy(ja);
        }
        toc_op_list_free(la);
    }
    /* a LUT op (helper-call path) also runs through the JIT */
    {
        toc_op_list *l2 = newlist();
        toc_op *m = toc_op_list_push(l2, TOC_OP_MATRIX);
        toc_op *lut = toc_op_list_push(l2, TOC_OP_LUT3D);
        static float cube[2 * 2 * 2 * 3];
        int ir, ig, ib, n;
        toc_jit *j2 = NULL;
        for (ib = 0; ib < 2; ++ib)
            for (ig = 0; ig < 2; ++ig)
                for (ir = 0; ir < 2; ++ir) {
                    size_t idx = (((size_t)ib * 2 + ig) * 2 + ir) * 3;
                    cube[idx + 0] = (float)ir; cube[idx + 1] = (float)ig;
                    cube[idx + 2] = (float)ib;
                }
        for (n = 0; n < 4; ++n) m->u.matrix.m[n * 4 + n] = 1.0f; /* identity */
        lut->u.lut3d.size = 2; lut->u.lut3d.data = cube;
        lut->u.lut3d.domain_max[0] = lut->u.lut3d.domain_max[1] =
            lut->u.lut3d.domain_max[2] = 1.0f;
        lut->u.lut3d.interp = TOC_INTERP_TETRAHEDRAL;
        if (TOC_OK(toc_jit_compile(l2, 4, NULL, &j2)) && j2) {
            float a[4] = {0.2f, 0.6f, 0.9f, 1.0f}, b[4];
            int c;
            memcpy(b, a, sizeof(a));
            toc_apply(l2, a, 1, 4);
            toc_jit_func(j2)(b, 1);
            ok = 1;
            for (c = 0; c < 4; ++c) if (!approx(a[c], b[c], 1e-6f)) ok = 0;
            CHECK(ok, "JIT with LUT3D (helper call) == interpreter");
            toc_jit_destroy(j2);
        }
        toc_op_list_free(l2);
    }
    toc_op_list_free(l);
}

int main(void) {
    printf("== tocio math ==\n");
    test_math();
    test_inv4x4();
    printf("== tocio strbuf ==\n");
    test_strbuf();
    printf("== tocio parse ==\n");
    test_parse();
    printf("== tocio interpreter ==\n");
    test_interp();
    printf("== tocio yaml ==\n");
    test_yaml();
    printf("== tocio processor ==\n");
    test_processor();
    printf("== tocio file LUTs ==\n");
    test_lutfile();
    test_filetransform();
    printf("== tocio AOT-C codegen ==\n");
    test_codegen_c();
    printf("== tocio GLSL codegen ==\n");
    test_codegen_glsl();
    printf("== tocio SIMD parity ==\n");
    test_simd_parity();
    printf("== tocio builtins + fixedfunc ==\n");
    test_builtins();
    printf("== tocio x64 JIT ==\n");
    test_jit();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
