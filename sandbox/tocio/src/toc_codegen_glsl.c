/*
 * tocio - GLSL/WebGL/Vulkan shader codegen. Emits a `vec4 OCIOMain(vec4)`
 * function plus a list of LUT texture descriptors the host uploads, following
 * OpenColorIO's GPU section model. 1D LUTs use an Nx1 sampler2D (portable
 * across ES3.0/330/450); 3D LUTs use sampler3D with hardware trilinear.
 *
 * Reimplemented from OpenColorIO's GPU path (BSD-3-Clause).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

static void emit_f(toc_sb *sb, float v) { toc_sb_decfloat(sb, v); }

static void emit_vec3(toc_sb *sb, float a, float b, float c) {
    toc_sb_puts(sb, "vec3(");
    emit_f(sb, a); toc_sb_putc(sb, ',');
    emit_f(sb, b); toc_sb_putc(sb, ',');
    emit_f(sb, c); toc_sb_putc(sb, ')');
}

static int sh_add_texture(toc_shader *sh, const toc_op *op, int idx) {
    toc_texture_desc *nt;
    toc_texture_desc *d;
    size_t cap = sh->num_textures + 1;
    nt = (toc_texture_desc *)toc_malloc(&sh->alloc, cap * sizeof(*nt));
    if (!nt) return 0;
    if (sh->textures) {
        memcpy(nt, sh->textures, sh->num_textures * sizeof(*nt));
        toc_free(&sh->alloc, sh->textures);
    }
    sh->textures = nt;
    d = &sh->textures[sh->num_textures++];
    memset(d, 0, sizeof(*d));
    /* sampler_name = ociolut<idx> */
    {
        char *p = d->sampler_name;
        const char *pre = "ociolut";
        int i = 0, v = idx;
        char tmp[12];
        int n = 0;
        while (pre[i]) p[i] = pre[i], ++i;
        if (v == 0) tmp[n++] = '0';
        while (v) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
        while (n--) p[i++] = tmp[n];
        p[i] = 0;
    }
    if (op->kind == TOC_OP_LUT1D) {
        d->dim = TOC_TEX_2D;
        d->width = op->u.lut1d.length;
        d->height = 1;
        d->depth = 1;
        d->channels = op->u.lut1d.channels == 1 ? 1 : 3;
        d->data = op->u.lut1d.data;
        d->interp_linear = op->u.lut1d.interp != TOC_INTERP_NEAREST;
    } else {
        d->dim = TOC_TEX_3D;
        d->width = d->height = d->depth = op->u.lut3d.size;
        d->channels = 3;
        d->data = op->u.lut3d.data;
        d->interp_linear = 1;
    }
    return 1;
}

/* ACES Red Modifier + Gamut Compression as GLSL helper functions (emitted once
 * before OCIOMain when the pipeline uses them). Reproduces the interpreter math
 * with GLSL built-ins. */
static const char *GLSL_ACES_SRC =
    "float tocRmHue(float r,float g,float b,float iw){\n"
    "  float a=2.0*r-(g+b),bb=1.7320508075688772*(g-b);\n"
    "  float knot=atan(bb,a)*iw+2.0;int j=int(knot);\n"
    "  if(j<0||j>=4)return 0.0;\n"
    "  vec4 c=j==0?vec4(0.25,0.0,0.0,0.0):j==1?vec4(-0.75,0.75,0.75,0.25):\n"
    "          j==2?vec4(0.75,-1.5,0.0,1.0):vec4(-0.25,0.75,-0.75,0.25);\n"
    "  float t=knot-float(j);return c.w+t*(c.z+t*(c.y+t*c.x));}\n"
    "float tocRmSat(vec3 c){float mn=min(c.r,min(c.g,c.b)),mx=max(c.r,max(c.g,c.b));\n"
    "  return (max(1e-10,mx)-max(1e-10,mn))/max(1e-2,mx);}\n"
    "vec3 tocRedmod(vec3 c,float scale,float pivot,float iw,bool inv){\n"
    "  float fH=tocRmHue(c.r,c.g,c.b,iw);if(fH<=0.0)return c;\n"
    "  float r=c.r,g=c.g,b=c.b,oms=1.0-scale,nr;\n"
    "  if(inv){float mc=min(g,b),aa=fH*oms-1.0,bb=r-fH*(pivot+mc)*oms,cc=fH*pivot*mc*oms;\n"
    "    float disc=bb*bb-4.0*aa*cc;nr=disc>=0.0?(-bb-sqrt(disc))/(2.0*aa):r;}\n"
    "  else{float fS=tocRmSat(c);nr=r+fH*fS*(pivot-r)*oms;}\n"
    "  if(g>=b)g=(g-b)/max(1e-10,r-b)*(nr-b)+b;else b=(b-g)/max(1e-10,r-g)*(nr-g)+g;\n"
    "  return vec3(nr,g,b);}\n"
    "float tocGcOne(float dist,float thr,float scale,float power,bool inv){\n"
    "  float ip=1.0/power,nd,p;\n"
    "  if(inv){if(dist>=thr+scale)return dist;nd=(dist-thr)/scale;p=pow(nd,power);\n"
    "    return thr+scale*pow(-(p/(p-1.0)),ip);}\n"
    "  nd=(dist-thr)/scale;p=pow(nd,power);return thr+scale*nd/pow(1.0+p,ip);}\n"
    "float tocGcApply(float v,float ach,float thr,float scale,float power,bool inv){\n"
    "  if(ach==0.0)return 0.0;float af=abs(ach),dist=(ach-v)/af;\n"
    "  if(dist<thr)return v;return ach-tocGcOne(dist,thr,scale,power,inv)*af;}\n"
    "vec3 tocGamutComp(vec3 c,vec3 thr,vec3 sc,float power,bool inv){\n"
    "  float ach=max(c.r,max(c.g,c.b));\n"
    "  return vec3(tocGcApply(c.r,ach,thr.x,sc.x,power,inv),\n"
    "              tocGcApply(c.g,ach,thr.y,sc.y,power,inv),\n"
    "              tocGcApply(c.b,ach,thr.z,sc.z,power,inv));}\n";

/* gamut-compression scale (codegen-time constant; matches interpreter gc_scale). */
static float glsl_gc_scale(float lim, float thr, float power) {
    float ip, t, p, d;
    if (lim <= thr) return 1.0f;
    ip = 1.0f / power;
    t = (1.0f - thr) / (lim - thr);
    p = toc_powf(t, power);
    d = toc_powf(p - 1.0f, ip);
    return (1.0f - thr) / d;
}

toc_result toc_emit_glsl(const toc_op_list *ops, toc_glsl_target target,
                         const toc_allocator *a, toc_shader *out) {
    toc_sb sb;
    size_t k;
    int tex_idx = 0;
    int vulkan = (target == TOC_GLSL_VULKAN450);
    if (!ops || !out) return TOC_ERROR_INVALID_ARGUMENT;
    if (!a) a = toc_default_allocator();
    memset(out, 0, sizeof(*out));
    out->alloc = *a;
    toc_sb_init(&sb, a);

    toc_sb_puts(&sb, "#version ");
    toc_sb_puts(&sb, target == TOC_GLSL_ES30 ? "300 es"
                     : target == TOC_GLSL_330 ? "330" : "450");
    toc_sb_putc(&sb, '\n');
    if (target == TOC_GLSL_ES30)
        toc_sb_puts(&sb, "precision highp float;\nprecision highp sampler3D;\n");
    toc_sb_puts(&sb,
                "// Generated by tocio. SPDX-License-Identifier: BSD-3-Clause\n"
                "// Color pipeline reimplemented from OpenColorIO "
                "(BSD-3-Clause).\n");

    /* sampler declarations (one per LUT op) */
    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        const char *styp;
        if (op->kind != TOC_OP_LUT1D && op->kind != TOC_OP_LUT3D) continue;
        styp = (op->kind == TOC_OP_LUT3D) ? "sampler3D" : "sampler2D";
        if (vulkan) {
            toc_sb_puts(&sb, "layout(binding=");
            toc_sb_int(&sb, tex_idx);
            toc_sb_puts(&sb, ") ");
        }
        toc_sb_puts(&sb, "uniform ");
        toc_sb_puts(&sb, styp);
        toc_sb_puts(&sb, " ociolut");
        toc_sb_int(&sb, tex_idx);
        toc_sb_puts(&sb, ";\n");
        if (!sh_add_texture(out, op, tex_idx)) { toc_sb_free(&sb); return TOC_ERROR_OUT_OF_MEMORY; }
        ++tex_idx;
    }

    /* ACES Red-Mod / Gamut-Comp helper functions, if the pipeline uses them. */
    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        int s = op->u.fixedfunc.style;
        if (op->kind == TOC_OP_FIXEDFUNC &&
            ((s >= TOC_FF_ACES_RED_MOD_03 && s <= TOC_FF_ACES_RED_MOD_10_INV) ||
             s == TOC_FF_ACES_GAMUTCOMP13 || s == TOC_FF_ACES_GAMUTCOMP13_INV)) {
            toc_sb_puts(&sb, GLSL_ACES_SRC);
            break;
        }
    }

    toc_sb_puts(&sb, "vec4 OCIOMain(vec4 inPixel){\n  vec4 v = inPixel;\n");
    tex_idx = 0;
    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        switch (op->kind) {
            case TOC_OP_MATRIX: {
                const float *m = op->u.matrix.m, *o = op->u.matrix.off;
                int i;
                toc_sb_puts(&sb, "  v = mat4(");
                for (i = 0; i < 16; ++i) {
                    if (i) toc_sb_putc(&sb, ',');
                    emit_f(&sb, m[i]); /* GLSL mat4 is column-major == our m */
                }
                toc_sb_puts(&sb, ") * v + vec4(");
                emit_f(&sb, o[0]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[1]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[2]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[3]); toc_sb_puts(&sb, ");\n");
                break;
            }
            case TOC_OP_RANGE: {
                const float *s = op->u.range.scale, *o = op->u.range.offset;
                toc_sb_puts(&sb, "  v = v * vec4(");
                emit_f(&sb, s[0]); toc_sb_putc(&sb, ',');
                emit_f(&sb, s[1]); toc_sb_putc(&sb, ',');
                emit_f(&sb, s[2]); toc_sb_putc(&sb, ',');
                emit_f(&sb, s[3]); toc_sb_puts(&sb, ") + vec4(");
                emit_f(&sb, o[0]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[1]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[2]); toc_sb_putc(&sb, ',');
                emit_f(&sb, o[3]); toc_sb_puts(&sb, ");\n");
                if (op->u.range.clamp_lo || op->u.range.clamp_hi) {
                    toc_sb_puts(&sb, "  v = clamp(v, vec4(");
                    emit_f(&sb, op->u.range.clamp_lo ? op->u.range.min[0] : -1e30f);
                    toc_sb_puts(&sb, "), vec4(");
                    emit_f(&sb, op->u.range.clamp_hi ? op->u.range.max[0] : 1e30f);
                    toc_sb_puts(&sb, "));\n");
                }
                break;
            }
            case TOC_OP_EXPONENT:
                toc_sb_puts(&sb, "  v.rgb = pow(max(v.rgb, vec3(0.0)), ");
                emit_vec3(&sb, op->u.exponent.e[0], op->u.exponent.e[1],
                          op->u.exponent.e[2]);
                toc_sb_puts(&sb, ");\n");
                break;
            case TOC_OP_LOG:
                if (!op->u.log.inverse) {
                    toc_sb_puts(&sb, "  v.rgb = ");
                    emit_vec3(&sb, op->u.log.log_slope[0], op->u.log.log_slope[1],
                              op->u.log.log_slope[2]);
                    toc_sb_puts(&sb, " * log2(");
                    emit_vec3(&sb, op->u.log.lin_slope[0], op->u.log.lin_slope[1],
                              op->u.log.lin_slope[2]);
                    toc_sb_puts(&sb, " * v.rgb + ");
                    emit_vec3(&sb, op->u.log.lin_offset[0],
                              op->u.log.lin_offset[1], op->u.log.lin_offset[2]);
                    toc_sb_puts(&sb, ") / log2(");
                    emit_f(&sb, op->u.log.base);
                    toc_sb_puts(&sb, ") + ");
                    emit_vec3(&sb, op->u.log.log_offset[0],
                              op->u.log.log_offset[1], op->u.log.log_offset[2]);
                    toc_sb_puts(&sb, ";\n");
                } else {
                    toc_sb_puts(&sb, "  v.rgb = (pow(vec3(");
                    emit_f(&sb, op->u.log.base);
                    toc_sb_puts(&sb, "), (v.rgb - ");
                    emit_vec3(&sb, op->u.log.log_offset[0],
                              op->u.log.log_offset[1], op->u.log.log_offset[2]);
                    toc_sb_puts(&sb, ") / ");
                    emit_vec3(&sb, op->u.log.log_slope[0], op->u.log.log_slope[1],
                              op->u.log.log_slope[2]);
                    toc_sb_puts(&sb, ") - ");
                    emit_vec3(&sb, op->u.log.lin_offset[0],
                              op->u.log.lin_offset[1], op->u.log.lin_offset[2]);
                    toc_sb_puts(&sb, ") / ");
                    emit_vec3(&sb, op->u.log.lin_slope[0], op->u.log.lin_slope[1],
                              op->u.log.lin_slope[2]);
                    toc_sb_puts(&sb, ";\n");
                }
                break;
            case TOC_OP_EXP_LINEAR: {
                /* MonCurve on RGB (alpha preserved): power above the breakpoint,
                 * linear below, selected per channel. */
                const float *sc = op->u.exp_linear.scale, *of = op->u.exp_linear.offset;
                const float *gm = op->u.exp_linear.gamma, *bk = op->u.exp_linear.breakpoint;
                const float *sl = op->u.exp_linear.slope;
                if (!op->u.exp_linear.inverse) {
                    toc_sb_puts(&sb, "  v.rgb = mix(v.rgb * ");
                    emit_vec3(&sb, sl[0], sl[1], sl[2]);
                    toc_sb_puts(&sb, ", pow(v.rgb * ");
                    emit_vec3(&sb, sc[0], sc[1], sc[2]);
                    toc_sb_puts(&sb, " + ");
                    emit_vec3(&sb, of[0], of[1], of[2]);
                    toc_sb_puts(&sb, ", ");
                    emit_vec3(&sb, gm[0], gm[1], gm[2]);
                    toc_sb_puts(&sb, "), greaterThan(v.rgb, ");
                    emit_vec3(&sb, bk[0], bk[1], bk[2]);
                    toc_sb_puts(&sb, "));\n");
                } else {
                    toc_sb_puts(&sb, "  v.rgb = mix(v.rgb / ");
                    emit_vec3(&sb, sl[0], sl[1], sl[2]);
                    toc_sb_puts(&sb, ", (pow(v.rgb, ");
                    emit_vec3(&sb, 1.0f / gm[0], 1.0f / gm[1], 1.0f / gm[2]);
                    toc_sb_puts(&sb, ") - ");
                    emit_vec3(&sb, of[0], of[1], of[2]);
                    toc_sb_puts(&sb, ") / ");
                    emit_vec3(&sb, sc[0], sc[1], sc[2]);
                    toc_sb_puts(&sb, ", greaterThan(v.rgb, ");
                    emit_vec3(&sb, bk[0] * sl[0], bk[1] * sl[1], bk[2] * sl[2]);
                    toc_sb_puts(&sb, "));\n");
                }
                break;
            }
            case TOC_OP_CDL: {
                const float *L = op->u.cdl.luma;
                float lr = L[0], lg = L[1], lb = L[2];
                if (lr == 0 && lg == 0 && lb == 0) {
                    lr = 0.2126f; lg = 0.7152f; lb = 0.0722f;
                }
                if (op->u.cdl.inverse) {
                    float sat = op->u.cdl.saturation;
                    float p0 = op->u.cdl.power[0], p1 = op->u.cdl.power[1],
                          p2 = op->u.cdl.power[2];
                    float s0 = op->u.cdl.slope[0], s1 = op->u.cdl.slope[1],
                          s2 = op->u.cdl.slope[2];
                    toc_sb_puts(&sb, "  { vec3 c = v.rgb;\n  float L = dot(c, ");
                    emit_vec3(&sb, lr, lg, lb);
                    toc_sb_puts(&sb, ");\n  c = L + (c - L) * ");
                    emit_f(&sb, sat != 0.0f ? 1.0f / sat : 0.0f);
                    toc_sb_puts(&sb, ";\n");
                    if (op->u.cdl.clamp)
                        toc_sb_puts(&sb, "  c = clamp(c, 0.0, 1.0);\n");
                    toc_sb_puts(&sb, "  c = pow(c, ");
                    emit_vec3(&sb, p0 != 0.0f ? 1.0f / p0 : 1.0f,
                              p1 != 0.0f ? 1.0f / p1 : 1.0f,
                              p2 != 0.0f ? 1.0f / p2 : 1.0f);
                    toc_sb_puts(&sb, ");\n  c = (c - ");
                    emit_vec3(&sb, op->u.cdl.offset[0], op->u.cdl.offset[1],
                              op->u.cdl.offset[2]);
                    toc_sb_puts(&sb, ") * ");
                    emit_vec3(&sb, s0 != 0.0f ? 1.0f / s0 : 1.0f,
                              s1 != 0.0f ? 1.0f / s1 : 1.0f,
                              s2 != 0.0f ? 1.0f / s2 : 1.0f);
                    toc_sb_puts(&sb, ";\n  v.rgb = c; }\n");
                    break;
                }
                toc_sb_puts(&sb, "  { vec3 c = v.rgb * ");
                emit_vec3(&sb, op->u.cdl.slope[0], op->u.cdl.slope[1],
                          op->u.cdl.slope[2]);
                toc_sb_puts(&sb, " + ");
                emit_vec3(&sb, op->u.cdl.offset[0], op->u.cdl.offset[1],
                          op->u.cdl.offset[2]);
                toc_sb_puts(&sb, ";\n");
                if (op->u.cdl.clamp)
                    toc_sb_puts(&sb, "  c = clamp(c, 0.0, 1.0);\n");
                toc_sb_puts(&sb, "  c = pow(c, ");
                emit_vec3(&sb, op->u.cdl.power[0], op->u.cdl.power[1],
                          op->u.cdl.power[2]);
                toc_sb_puts(&sb, ");\n  float L = dot(c, ");
                emit_vec3(&sb, lr, lg, lb);
                toc_sb_puts(&sb, ");\n  c = L + ");
                emit_f(&sb, op->u.cdl.saturation);
                toc_sb_puts(&sb, " * (c - L);\n");
                if (op->u.cdl.clamp)
                    toc_sb_puts(&sb, "  c = clamp(c, 0.0, 1.0);\n");
                toc_sb_puts(&sb, "  v.rgb = c; }\n");
                break;
            }
            case TOC_OP_LUT1D: {
                int N = op->u.lut1d.length, ch;
                static const char *comp[3] = {"r", "g", "b"};
                float dmin = op->u.lut1d.domain_min, dmax = op->u.lut1d.domain_max;
                for (ch = 0; ch < 3; ++ch) {
                    toc_sb_puts(&sb, "  v.");
                    toc_sb_puts(&sb, comp[ch]);
                    toc_sb_puts(&sb, " = texture(ociolut");
                    toc_sb_int(&sb, tex_idx);
                    toc_sb_puts(&sb, ", vec2((clamp((v.");
                    toc_sb_puts(&sb, comp[ch]);
                    toc_sb_puts(&sb, " - ");
                    emit_f(&sb, dmin);
                    toc_sb_puts(&sb, ") / ");
                    emit_f(&sb, dmax - dmin);
                    toc_sb_puts(&sb, ", 0.0, 1.0) * ");
                    emit_f(&sb, (float)(N - 1));
                    toc_sb_puts(&sb, " + 0.5) / ");
                    emit_f(&sb, (float)N);
                    toc_sb_puts(&sb, ", 0.5)).");
                    toc_sb_puts(&sb, op->u.lut1d.channels == 1 ? "r" : comp[ch]);
                    toc_sb_puts(&sb, ";\n");
                }
                ++tex_idx;
                break;
            }
            case TOC_OP_LUT3D: {
                int N = op->u.lut3d.size;
                const float *dn = op->u.lut3d.domain_min;
                const float *dx = op->u.lut3d.domain_max;
                toc_sb_puts(&sb, "  { vec3 u = clamp((v.rgb - ");
                emit_vec3(&sb, dn[0], dn[1], dn[2]);
                toc_sb_puts(&sb, ") / ");
                emit_vec3(&sb, dx[0] - dn[0], dx[1] - dn[1], dx[2] - dn[2]);
                toc_sb_puts(&sb, ", 0.0, 1.0);\n  v.rgb = texture(ociolut");
                toc_sb_int(&sb, tex_idx);
                toc_sb_puts(&sb, ", (u * ");
                emit_f(&sb, (float)(N - 1));
                toc_sb_puts(&sb, " + 0.5) / ");
                emit_f(&sb, (float)N);
                toc_sb_puts(&sb, ").rgb; }\n");
                ++tex_idx;
                break;
            }
            case TOC_OP_FIXEDFUNC: {
                int s = op->u.fixedfunc.style;
                if (s == TOC_FF_REC2100_SURROUND ||
                    s == TOC_FF_REC2100_SURROUND_INV) {
                    float g = op->u.fixedfunc.params[0];
                    float e = (s == TOC_FF_REC2100_SURROUND)
                                  ? (g - 1.0f) : (1.0f / g - 1.0f);
                    toc_sb_puts(&sb, "  { float Y = 0.2627*v.r+0.6780*v.g+0.0593*v.b;\n"
                                     "  float s = Y>0.0?pow(Y,");
                    emit_f(&sb, e);
                    toc_sb_puts(&sb, "):0.0;\n  v.rgb *= s; }\n");
                } else if (s == TOC_FF_ACES_GLOW03 ||
                           s == TOC_FF_ACES_GLOW10) {
                    float gg = op->u.fixedfunc.params[0];
                    float gm = op->u.fixedfunc.params[1];
                    toc_sb_puts(&sb, "  { float YC=(v.r+v.g+v.b+1.75*sqrt(max(v.b*(v.b-v.g)+v.g*(v.g-v.r)+v.r*(v.r-v.b),0.0)))/3.0;\n"
                                     "  float mn=min(v.r,min(v.g,v.b));float mx=max(v.r,max(v.g,v.b));\n"
                                     "  float sat=(max(1e-10,mx)-max(1e-10,mn))/max(0.01,mx);\n"
                                     "  float x=(sat-0.4)*5.0;float sg=sign(x);float t=max(0.0,1.0-0.5*sg*x);\n"
                                     "  float s=(1.0+sg*(1.0-t*t))*0.5;\n"
                                     "  float GG="); emit_f(&sb, gg); toc_sb_puts(&sb, "*s;\n"
                                     "  float go;\n"
                                     "  if(YC>= "); emit_f(&sb, gm * 2.0f); toc_sb_puts(&sb, ")go=0.0;\n"
                                     "  else if(YC<= "); emit_f(&sb, gm * 2.0f / 3.0f); toc_sb_puts(&sb, ")go=GG;\n"
                                     "  else go=GG*("); emit_f(&sb, gm); toc_sb_puts(&sb, "/YC-0.5);\n"
                                     "  v.rgb *= 1.0+go; }\n");
                } else if (s == TOC_FF_ACES_GLOW03_INV ||
                           s == TOC_FF_ACES_GLOW10_INV) {
                    float gg = op->u.fixedfunc.params[0];
                    float gm = op->u.fixedfunc.params[1];
                    toc_sb_puts(&sb, "  { float YC=(v.r+v.g+v.b+1.75*sqrt(max(v.b*(v.b-v.g)+v.g*(v.g-v.r)+v.r*(v.r-v.b),0.0)))/3.0;\n"
                                     "  float mn=min(v.r,min(v.g,v.b));float mx=max(v.r,max(v.g,v.b));\n"
                                     "  float sat=(max(1e-10,mx)-max(1e-10,mn))/max(0.01,mx);\n"
                                     "  float x=(sat-0.4)*5.0;float sg=sign(x);float t=max(0.0,1.0-0.5*sg*x);\n"
                                     "  float s=(1.0+sg*(1.0-t*t))*0.5;\n"
                                     "  float GG="); emit_f(&sb, gg); toc_sb_puts(&sb, "*s;\n"
                                     "  float go;\n"
                                     "  if(YC>= "); emit_f(&sb, gm * 2.0f); toc_sb_puts(&sb, ")go=0.0;\n"
                                     "  else if(YC<= "); emit_f(&sb, (1.0f + gg) * gm * 2.0f / 3.0f); toc_sb_puts(&sb, ")go=-GG/(1.0+GG);\n"
                                     "  else go=GG*("); emit_f(&sb, gm); toc_sb_puts(&sb, "/YC-0.5)/(GG*0.5-1.0);\n"
                                     "  v.rgb *= 1.0+go; }\n");
                } else if (s == TOC_FF_ACES_DARKTODIM10) {
                    float g = op->u.fixedfunc.params[0];
                    toc_sb_puts(&sb, "  { float Y=max(1e-10,0.27222872*v.r+0.67408177*v.g+0.05368952*v.b);\n"
                                     "  v.rgb *= pow(Y,");
                    emit_f(&sb, g - 1.0f);
                    toc_sb_puts(&sb, "); }\n");
                } else if (s == TOC_FF_ACES_DARKTODIM10_INV) {
                    float g = op->u.fixedfunc.params[0];
                    toc_sb_puts(&sb, "  { float Y=max(1e-10,0.27222872*v.r+0.67408177*v.g+0.05368952*v.b);\n"
                                     "  v.rgb *= pow(Y,");
                    emit_f(&sb, 1.0f / g - 1.0f);
                    toc_sb_puts(&sb, "); }\n");
                } else if (s == TOC_FF_ACES_RED_MOD_03 ||
                           s == TOC_FF_ACES_RED_MOD_03_INV ||
                           s == TOC_FF_ACES_RED_MOD_10 ||
                           s == TOC_FF_ACES_RED_MOD_10_INV) {
                    int is03 = (s == TOC_FF_ACES_RED_MOD_03 ||
                                s == TOC_FF_ACES_RED_MOD_03_INV);
                    int inv = (s == TOC_FF_ACES_RED_MOD_03_INV ||
                               s == TOC_FF_ACES_RED_MOD_10_INV);
                    toc_sb_puts(&sb, "  v.rgb = tocRedmod(v.rgb, ");
                    emit_f(&sb, is03 ? 0.85f : 0.82f);
                    toc_sb_puts(&sb, ", 0.03, ");
                    emit_f(&sb, is03 ? 1.9098593171027443f : 1.6976527263135504f);
                    toc_sb_puts(&sb, inv ? ", true);\n" : ", false);\n");
                } else if (s == TOC_FF_ACES_GAMUTCOMP13 ||
                           s == TOC_FF_ACES_GAMUTCOMP13_INV) {
                    const float *pp = op->u.fixedfunc.params;
                    float pw = pp[6];
                    int inv = (s == TOC_FF_ACES_GAMUTCOMP13_INV);
                    toc_sb_puts(&sb, "  v.rgb = tocGamutComp(v.rgb, vec3(");
                    emit_f(&sb, pp[3]); toc_sb_putc(&sb, ',');
                    emit_f(&sb, pp[4]); toc_sb_putc(&sb, ',');
                    emit_f(&sb, pp[5]); toc_sb_puts(&sb, "), vec3(");
                    emit_f(&sb, glsl_gc_scale(pp[0], pp[3], pw)); toc_sb_putc(&sb, ',');
                    emit_f(&sb, glsl_gc_scale(pp[1], pp[4], pw)); toc_sb_putc(&sb, ',');
                    emit_f(&sb, glsl_gc_scale(pp[2], pp[5], pw)); toc_sb_puts(&sb, "), ");
                    emit_f(&sb, pw);
                    toc_sb_puts(&sb, inv ? ", true);\n" : ", false);\n");
                } else if (s == TOC_FF_RGB_TO_HSV) {
                    toc_sb_puts(&sb, "  { float mn=min(v.r,min(v.g,v.b));float mx=max(v.r,max(v.g,v.b));\n"
                                     "  float h=0,s=0,vv=mx;if(mn!=mx){float d=mx-mn;\n"
                                     "  if(mx!=0.0)s=d/mx;\n"
                                     "  if(v.r==mx)h=(v.g-v.b)/d;else if(v.g==mx)h=2.0+(v.b-v.r)/d;else h=4.0+(v.r-v.g)/d;\n"
                                     "  if(h<0)h+=6.0;h*=0.16666667;}\n"
                                     "  if(mn<0)vv+=mn;if(-mn>mx)s=(mx-mn)/-mn;\n"
                                     "  v=vec4(h,s,vv,v.a);}\n");
                } else if (s == TOC_FF_HSV_TO_RGB) {
                    toc_sb_puts(&sb, "  { float h=(v.r-floor(v.r))*6.0,s=clamp(v.g,0.0,1.999),vv=v.b;\n"
                                     "  float rp=clamp(abs(h-3.0)-1.0,0.0,1.0);\n"
                                     "  float gp=1.0-clamp(abs(h-2.0),0.0,1.0);\n"
                                     "  float bp=1.0-clamp(abs(h-4.0),0.0,1.0);\n"
                                     "  float mx=vv,mn=vv*(1.0-s);\n"
                                     "  if(s>1.0){mn=vv*(1.0-s)/(2.0-s);mx=vv-mn;}\n"
                                     "  if(vv<0.0){mn=vv/(2.0-s);mx=vv-mn;}float d=mx-mn;\n"
                                     "  v=vec4(rp*d+mn,gp*d+mn,bp*d+mn,v.a);}\n");
                } else if (s == TOC_FF_XYZ_TO_xyY) {
                    toc_sb_puts(&sb, "  { float d=1.0/max(v.r+v.g+v.b,1e-10);v=vec4(v.r*d,v.g*d,v.b,v.a);}\n");
                } else if (s == TOC_FF_xyY_TO_XYZ) {
                    toc_sb_puts(&sb, "  { float d=1.0/max(v.g,1e-10);\n"
                                     "  v=vec4(v.b*v.r*d,v.b,v.b*(1.0-v.r-v.g)*d,v.a);}\n");
                } else if (s == TOC_FF_XYZ_TO_uvY) {
                    toc_sb_puts(&sb, "  { float d=1.0/max(v.r+15.0*v.g+3.0*v.b,1e-10);\n"
                                     "  v=vec4(4.0*v.r*d,9.0*v.g*d,v.b,v.a);}\n");
                } else if (s == TOC_FF_uvY_TO_XYZ) {
                    toc_sb_puts(&sb, "  { float d=1.0/max(v.g,1e-10);\n"
                                     "  v=vec4(2.25*v.b*v.r*d,v.b,0.75*v.b*(4.0-v.r-6.66667*v.g)*d,v.a);}\n");
                } else if (s == TOC_FF_XYZ_TO_LUV) {
                    toc_sb_puts(&sb, "  { float d=1.0/max(v.r+15.0*v.g+3.0*v.b,1e-10);\n"
                                     "  float u=4.0*v.r*d,vv=9.0*v.g*d;\n"
                                     "  float ls=v.b<=0.008856?9.03296*v.b:1.16*pow(v.b,0.33333)-0.16;\n"
                                     "  v=vec4(ls,13.0*ls*(u-0.19783),13.0*ls*(vv-0.46832),v.a);}\n");
                } else if (s == TOC_FF_LUV_TO_XYZ) {
                    toc_sb_puts(&sb, "  { float d=0.076923/max(v.r,1e-10);\n"
                                     "  float u=v.g*d+0.19783,vv=v.b*d+0.46832;\n"
                                     "  float t=(v.r+0.16)*0.862069;\n"
                                     "  float Y=v.r<=0.08?0.110706*v.r:t*t*t;\n"
                                     "  float dd=0.25/max(vv,1e-10);\n"
                                     "  v=vec4(9.0*Y*u*dd,Y,Y*(12.0-3.0*u-20.0*vv)*dd,v.a);}\n");
                } else if (s == TOC_FF_LIN_TO_PQ) {
                    toc_sb_puts(&sb,
                        "  { vec3 Lm=pow(max(v.rgb,0.0),vec3(0.1593017578125));\n"
                        "  v.rgb=pow((0.8359375+18.8515625*Lm)/(1.0+18.6875*Lm),vec3(78.84375)); }\n");
                } else if (s == TOC_FF_PQ_TO_LIN) {
                    toc_sb_puts(&sb,
                        "  { vec3 Np=pow(max(v.rgb,0.0),vec3(0.012683313));\n"
                        "  vec3 nu=max(Np-0.8359375,0.0),de=max(18.8515625-18.6875*Np,1e-10);\n"
                        "  v.rgb=pow(nu/de,vec3(6.2773438)); }\n");
                } else if (s == TOC_FF_LIN_TO_HLG) {
                    toc_sb_puts(&sb,
                        "  { vec3 e=max(v.rgb,0.0);\n"
                        "  v.rgb=mix(0.17883277*log(max(12.0*e-0.28466892,1e-10))+0.55991073,\n"
                        "            sqrt(3.0*e), lessThanEqual(e,vec3(0.083333333))); }\n");
                } else if (s == TOC_FF_HLG_TO_LIN) {
                    toc_sb_puts(&sb,
                        "  { v.rgb=mix((exp((v.rgb-0.55991073)/0.17883277)+0.28466892)/12.0,\n"
                        "            v.rgb*v.rgb/3.0, lessThanEqual(v.rgb,vec3(0.5))); }\n");
                }
                break;
            }
            default: break;
        }
    }
    toc_sb_puts(&sb, "  return v;\n}\n");
    if (sb.oom) { toc_sb_free(&sb); toc_shader_free(out); return TOC_ERROR_OUT_OF_MEMORY; }
    out->source = toc_sb_take(&sb, NULL);
    return out->source ? TOC_SUCCESS : TOC_ERROR_OUT_OF_MEMORY;
}

void toc_shader_free(toc_shader *sh) {
    if (!sh) return;
    toc_free(&sh->alloc, sh->source);
    toc_free(&sh->alloc, sh->textures);
    sh->source = NULL;
    sh->textures = NULL;
    sh->num_textures = 0;
}
