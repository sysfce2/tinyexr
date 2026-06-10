/*
 * tocio - AOT C-source codegen. Emits a standalone .c implementing the op chain
 * as a straight-line per-pixel loop over interleaved RGBA float. LUT data is
 * embedded as static const arrays; constants are bit-exact hex floats. The
 * emitted file is dependency-free (its own libm-free pow/log2/exp2).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

static void emit_hf(toc_sb *sb, float v) { toc_sb_hexfloat(sb, v); }

/* Emit a 4x4 matrix*rgba+offset into fresh temps then write back. */
static void op_matrix(toc_sb *sb, const toc_op *op) {
    const float *m = op->u.matrix.m, *o = op->u.matrix.off;
    static const char *up[4] = {"R", "G", "B", "A"};
    int r;
    toc_sb_puts(sb, "  { float R,G,B,A;\n");
    for (r = 0; r < 4; ++r) {
        toc_sb_puts(sb, "  ");
        toc_sb_puts(sb, up[r]);
        toc_sb_puts(sb, " = ");
        emit_hf(sb, m[0 + r]); toc_sb_puts(sb, "*r + ");
        emit_hf(sb, m[4 + r]); toc_sb_puts(sb, "*g + ");
        emit_hf(sb, m[8 + r]); toc_sb_puts(sb, "*b + ");
        emit_hf(sb, m[12 + r]); toc_sb_puts(sb, "*a + ");
        emit_hf(sb, o[r]); toc_sb_puts(sb, ";\n");
    }
    toc_sb_puts(sb, "  r=R; g=G; b=B; a=A; }\n");
}

static void op_range(toc_sb *sb, const toc_op *op) {
    static const char *cn[4] = {"r", "g", "b", "a"};
    int c;
    for (c = 0; c < 4; ++c) {
        toc_sb_puts(sb, "  ");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, " = ");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, "*");
        emit_hf(sb, op->u.range.scale[c]);
        toc_sb_puts(sb, " + ");
        emit_hf(sb, op->u.range.offset[c]);
        toc_sb_puts(sb, ";");
        if (op->u.range.clamp_lo) {
            toc_sb_puts(sb, " if(");
            toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, "<");
            emit_hf(sb, op->u.range.min[c]);
            toc_sb_puts(sb, ") ");
            toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, "=");
            emit_hf(sb, op->u.range.min[c]);
            toc_sb_puts(sb, ";");
        }
        if (op->u.range.clamp_hi) {
            toc_sb_puts(sb, " if(");
            toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, ">");
            emit_hf(sb, op->u.range.max[c]);
            toc_sb_puts(sb, ") ");
            toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, "=");
            emit_hf(sb, op->u.range.max[c]);
            toc_sb_puts(sb, ";");
        }
        toc_sb_puts(sb, "\n");
    }
}

static void op_exponent(toc_sb *sb, const toc_op *op) {
    static const char *cn[4] = {"r", "g", "b", "a"};
    int c;
    for (c = 0; c < 4; ++c) {
        toc_sb_puts(sb, "  ");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, " = tc_powf(");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, ">0.0f?");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, ":0.0f, ");
        emit_hf(sb, op->u.exponent.e[c]);
        toc_sb_puts(sb, ");\n");
    }
}

static void op_exp_linear(toc_sb *sb, const toc_op *op) {
    static const char *cn[3] = {"r", "g", "b"};
    int c;
    for (c = 0; c < 3; ++c) {
        float scale = op->u.exp_linear.scale[c], off = op->u.exp_linear.offset[c];
        float g = op->u.exp_linear.gamma[c], brk = op->u.exp_linear.breakpoint[c];
        float slope = op->u.exp_linear.slope[c];
        toc_sb_puts(sb, "  ");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, " = (");
        if (!op->u.exp_linear.inverse) {
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, ">");
            emit_hf(sb, brk); toc_sb_puts(sb, ") ? tc_powf(");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "*"); emit_hf(sb, scale);
            toc_sb_puts(sb, "+"); emit_hf(sb, off); toc_sb_puts(sb, ", ");
            emit_hf(sb, g); toc_sb_puts(sb, ") : ");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "*"); emit_hf(sb, slope);
        } else {
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, ">");
            emit_hf(sb, brk * slope); toc_sb_puts(sb, ") ? (tc_powf(");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, ", "); emit_hf(sb, 1.0f / g);
            toc_sb_puts(sb, ")-"); emit_hf(sb, off); toc_sb_puts(sb, ")/");
            emit_hf(sb, scale); toc_sb_puts(sb, " : ");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "/"); emit_hf(sb, slope);
        }
        toc_sb_puts(sb, ";\n");
    }
}

static void op_log(toc_sb *sb, const toc_op *op) {
    static const char *cn[3] = {"r", "g", "b"};
    int c;
    for (c = 0; c < 3; ++c) {
        toc_sb_puts(sb, "  ");
        toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, " = ");
        if (!op->u.log.inverse) {
            toc_sb_puts(sb, "("); emit_hf(sb, op->u.log.log_slope[c]);
            toc_sb_puts(sb, "*tc_log2f((");
            emit_hf(sb, op->u.log.lin_slope[c]); toc_sb_puts(sb, "*");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "+");
            emit_hf(sb, op->u.log.lin_offset[c]); toc_sb_puts(sb, "))/tc_log2f(");
            emit_hf(sb, op->u.log.base); toc_sb_puts(sb, ")+");
            emit_hf(sb, op->u.log.log_offset[c]); toc_sb_puts(sb, ")");
        } else {
            toc_sb_puts(sb, "(tc_raisef(");
            emit_hf(sb, op->u.log.base); toc_sb_puts(sb, ", (");
            toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "-");
            emit_hf(sb, op->u.log.log_offset[c]); toc_sb_puts(sb, ")/");
            emit_hf(sb, op->u.log.log_slope[c]); toc_sb_puts(sb, ")-");
            emit_hf(sb, op->u.log.lin_offset[c]); toc_sb_puts(sb, ")/");
            emit_hf(sb, op->u.log.lin_slope[c]);
        }
        toc_sb_puts(sb, ";\n");
    }
}

static void op_cdl(toc_sb *sb, const toc_op *op) {
    const float *L = op->u.cdl.luma;
    float lr = L[0], lg = L[1], lb = L[2];
    int c;
    static const char *cn[3] = {"r", "g", "b"};
    if (lr == 0 && lg == 0 && lb == 0) { lr = 0.2126f; lg = 0.7152f; lb = 0.0722f; }
    toc_sb_puts(sb, "  { float v0,v1,v2,L;\n");
    for (c = 0; c < 3; ++c) {
        toc_sb_puts(sb, "  v"); toc_sb_int(sb, c); toc_sb_puts(sb, " = ");
        toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "*"); emit_hf(sb, op->u.cdl.slope[c]);
        toc_sb_puts(sb, "+"); emit_hf(sb, op->u.cdl.offset[c]); toc_sb_puts(sb, ";");
        if (op->u.cdl.clamp) {
            toc_sb_puts(sb, " if(v"); toc_sb_int(sb, c);
            toc_sb_puts(sb, "<0.0f)v"); toc_sb_int(sb, c);
            toc_sb_puts(sb, "=0.0f; if(v"); toc_sb_int(sb, c);
            toc_sb_puts(sb, ">1.0f)v"); toc_sb_int(sb, c); toc_sb_puts(sb, "=1.0f;");
        }
        toc_sb_puts(sb, " v"); toc_sb_int(sb, c); toc_sb_puts(sb, "=v");
        toc_sb_int(sb, c); toc_sb_puts(sb, ">=0.0f?tc_powf(v"); toc_sb_int(sb, c);
        toc_sb_puts(sb, ", "); emit_hf(sb, op->u.cdl.power[c]);
        toc_sb_puts(sb, "):v"); toc_sb_int(sb, c); toc_sb_puts(sb, ";\n");
    }
    toc_sb_puts(sb, "  L = "); emit_hf(sb, lr); toc_sb_puts(sb, "*v0+");
    emit_hf(sb, lg); toc_sb_puts(sb, "*v1+"); emit_hf(sb, lb);
    toc_sb_puts(sb, "*v2;\n");
    for (c = 0; c < 3; ++c) {
        toc_sb_puts(sb, "  "); toc_sb_puts(sb, cn[c]);
        toc_sb_puts(sb, " = L+"); emit_hf(sb, op->u.cdl.saturation);
        toc_sb_puts(sb, "*(v"); toc_sb_int(sb, c); toc_sb_puts(sb, "-L);");
        if (op->u.cdl.clamp) {
            toc_sb_puts(sb, " if("); toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, "<0.0f)"); toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, "=0.0f; if("); toc_sb_puts(sb, cn[c]);
            toc_sb_puts(sb, ">1.0f)"); toc_sb_puts(sb, cn[c]); toc_sb_puts(sb, "=1.0f;");
        }
        toc_sb_puts(sb, "\n");
    }
    toc_sb_puts(sb, "  }\n");
}

/* LUT ops call emitted helpers with the embedded array name. */
static void emit_lut_array(toc_sb *sb, int idx, const float *data, size_t n) {
    size_t i;
    toc_sb_puts(sb, "static const float tc_lut");
    toc_sb_int(sb, idx);
    toc_sb_puts(sb, "[] = {");
    for (i = 0; i < n; ++i) {
        if (i) toc_sb_putc(sb, ',');
        emit_hf(sb, data[i]);
    }
    toc_sb_puts(sb, "};\n");
}

/* ---- helper source snippets --------------------------------------------- */
static const char *MATH_SRC =
    "static float tc_log2f(float x){union{float f;unsigned u;}v;int e;float "
    "m,t,t2,ln;if(!(x>0.0f))return x<0.0f?(0.0f/0.0f):-1e38f;v.f=x;e=(int)((v.u"
    ">>23)&0xffu)-127;v.u=(v.u&0x007fffffu)|0x3f800000u;m=v.f;t=(m-1.0f)/(m+1.0"
    "f);t2=t*t;ln=2.0f*t*(1.0f+t2*(1.0f/3.0f+t2*(1.0f/5.0f+t2*(1.0f/7.0f+t2*(1."
    "0f/9.0f)))));return (float)e+ln*1.4426950408889634f;}\n"
    "static float tc_exp2f(float x){union{unsigned u;float f;}v;float "
    "k,f,g,p;int ki;if(x>127.0f)return 1e38f;if(x<-126.0f)return 0.0f;k=(x>=0.0"
    "f)?(float)(int)(x+0.5f):(float)(int)(x-0.5f);f=x-k;g=f*0.6931471805599453f"
    ";p=1.0f+g*(1.0f+g*(0.5f+g*(1.0f/6.0f+g*(1.0f/24.0f+g*(1.0f/120.0f+g*(1.0f/"
    "720.0f))))));ki=(int)k;v.u=(unsigned)((ki+127)<<23);return p*v.f;}\n"
    "static float tc_powf(float x,float y){if(!(x>0.0f))return "
    "x==0.0f?(y>0.0f?0.0f:1.0f):(0.0f/0.0f);return tc_exp2f(y*tc_log2f(x));}\n"
    "static float tc_raisef(float b,float x){if(!(b>0.0f))return "
    "0.0f;return tc_exp2f(x*tc_log2f(b));}\n";

static const char *LUT1D_SRC =
    "static void tc_lut1d(const float*d,int N,int ch,float lo,float hi,float*c)"
    "{int k;float den=hi-lo;for(k=0;k<3;++k){float u=den!=0.0f?(c[k]-lo)/den:0."
    "0f,g,f,a,b;int i;if(!(u>0.0f))u=0.0f;if(u>1.0f)u=1.0f;g=u*(float)(N-1);i=("
    "int)g;if(i>=N-1)i=N-2;f=g-(float)i;if(ch==1){a=d[i];b=d[i+1];}else{a=d[i*3"
    "+k];b=d[(i+1)*3+k];}c[k]=a+(b-a)*f;}}\n";

static const char *LUT3D_SRC =
    "static void tc_lut3d(const float*d,int N,const float*dn,const float*dx,int"
    " tet,float*c){int ic[3],k;float f[3];for(k=0;k<3;++k){float den=dx[k]-dn[k"
    "],u=den!=0.0f?(c[k]-dn[k])/den:0.0f,g;if(!(u>0.0f))u=0.0f;if(u>1.0f)u=1.0f"
    ";g=u*(float)(N-1);ic[k]=(int)g;if(ic[k]>=N-1)ic[k]=N-2;f[k]=g-(float)ic[k]"
    ";}\n"
    "#define C(R,G,B) (d+((((size_t)(B)*N+(G))*N+(R))*3))\n"
    "{const float*c000=C(ic[0],ic[1],ic[2]);const float*c100=C(ic[0]+1,ic[1],ic"
    "[2]);const float*c010=C(ic[0],ic[1]+1,ic[2]);const float*c110=C(ic[0]+1,ic"
    "[1]+1,ic[2]);const float*c001=C(ic[0],ic[1],ic[2]+1);const float*c101=C(ic"
    "[0]+1,ic[1],ic[2]+1);const float*c011=C(ic[0],ic[1]+1,ic[2]+1);const float"
    "*c111=C(ic[0]+1,ic[1]+1,ic[2]+1);float fr=f[0],fg=f[1],fb=f[2];for(k=0;k<3"
    ";++k){float v;if(tet){if(fr>=fg&&fg>=fb)v=c000[k]+fr*(c100[k]-c000[k])+fg*"
    "(c110[k]-c100[k])+fb*(c111[k]-c110[k]);else if(fr>=fb&&fb>=fg)v=c000[k]+fr"
    "*(c100[k]-c000[k])+fb*(c101[k]-c100[k])+fg*(c111[k]-c101[k]);else if(fb>=f"
    "r&&fr>=fg)v=c000[k]+fb*(c001[k]-c000[k])+fr*(c101[k]-c001[k])+fg*(c111[k]-"
    "c101[k]);else if(fg>=fr&&fr>=fb)v=c000[k]+fg*(c010[k]-c000[k])+fr*(c110[k]"
    "-c010[k])+fb*(c111[k]-c110[k]);else if(fg>=fb&&fb>=fr)v=c000[k]+fg*(c010[k"
    "]-c000[k])+fb*(c011[k]-c010[k])+fr*(c111[k]-c011[k]);else v=c000[k]+fb*(c0"
    "01[k]-c000[k])+fg*(c011[k]-c001[k])+fr*(c111[k]-c011[k]);}else{float "
    "x00=c000[k]+(c100[k]-c000[k])*fr,x01=c001[k]+(c101[k]-c001[k])*fr,x10=c010"
    "[k]+(c110[k]-c010[k])*fr,x11=c011[k]+(c111[k]-c011[k])*fr,y0=x00+(x10-x00)"
    "*fg,y1=x01+(x11-x01)*fg;v=y0+(y1-y0)*fb;}c[k]=v;}}\n#undef C\n}\n";

toc_result toc_emit_c(const toc_op_list *ops, const toc_codegen_c_opts *opts,
                      const toc_allocator *a, char **out_src, size_t *out_len) {
    toc_sb sb;
    const char *fname = (opts && opts->func_name) ? opts->func_name : "tocio_apply";
    int need_math = 0, need_lut1d = 0, need_lut3d = 0, lut_idx = 0;
    size_t k;
    if (!ops || !out_src) return TOC_ERROR_INVALID_ARGUMENT;
    if (!a) a = toc_default_allocator();
    for (k = 0; k < ops->count; ++k) {
        switch (ops->ops[k].kind) {
            case TOC_OP_EXPONENT: case TOC_OP_EXP_LINEAR: case TOC_OP_LOG:
            case TOC_OP_CDL: need_math = 1; break;
            case TOC_OP_LUT1D: need_lut1d = 1; break;
            case TOC_OP_LUT3D: need_lut3d = 1; break;
            default: break;
        }
    }
    toc_sb_init(&sb, a);
    toc_sb_puts(&sb,
                "/* Generated by tocio. SPDX-License-Identifier: BSD-3-Clause.\n"
                " * Color math reimplemented from OpenColorIO (BSD-3-Clause).\n"
                " */\n#include <stddef.h>\n");
    if (need_math) toc_sb_puts(&sb, MATH_SRC);
    if (need_lut1d) toc_sb_puts(&sb, LUT1D_SRC);
    if (need_lut3d) toc_sb_puts(&sb, LUT3D_SRC);
    /* embed LUT arrays */
    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        if (op->kind == TOC_OP_LUT1D)
            emit_lut_array(&sb, (int)k, op->u.lut1d.data,
                           (size_t)op->u.lut1d.length *
                               (op->u.lut1d.channels == 1 ? 1 : 3));
        else if (op->kind == TOC_OP_LUT3D)
            emit_lut_array(&sb, (int)k, op->u.lut3d.data,
                           (size_t)op->u.lut3d.size * op->u.lut3d.size *
                               op->u.lut3d.size * 3);
    }
    toc_sb_puts(&sb, "void ");
    toc_sb_puts(&sb, fname);
    toc_sb_puts(&sb, "(float *rgba, size_t npix){\n  size_t i;\n"
                     "  for(i=0;i<npix;++i){\n"
                     "  float *px=rgba+i*4;\n"
                     "  float r=px[0],g=px[1],b=px[2],a=px[3];\n");
    (void)lut_idx;
    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        switch (op->kind) {
            case TOC_OP_MATRIX: op_matrix(&sb, op); break;
            case TOC_OP_RANGE: op_range(&sb, op); break;
            case TOC_OP_EXPONENT: op_exponent(&sb, op); break;
            case TOC_OP_EXP_LINEAR: op_exp_linear(&sb, op); break;
            case TOC_OP_LOG: op_log(&sb, op); break;
            case TOC_OP_CDL: op_cdl(&sb, op); break;
            case TOC_OP_LUT1D:
                toc_sb_puts(&sb, "  { float c[3]={r,g,b}; tc_lut1d(tc_lut");
                toc_sb_int(&sb, (int)k);
                toc_sb_puts(&sb, ", ");
                toc_sb_int(&sb, op->u.lut1d.length);
                toc_sb_puts(&sb, ", ");
                toc_sb_int(&sb, op->u.lut1d.channels);
                toc_sb_puts(&sb, ", ");
                emit_hf(&sb, op->u.lut1d.domain_min);
                toc_sb_puts(&sb, ", ");
                emit_hf(&sb, op->u.lut1d.domain_max);
                toc_sb_puts(&sb, ", c); r=c[0];g=c[1];b=c[2]; }\n");
                break;
            case TOC_OP_LUT3D:
                toc_sb_puts(&sb, "  { float c[3]={r,g,b}; static const float dn[3]={");
                emit_hf(&sb, op->u.lut3d.domain_min[0]); toc_sb_putc(&sb, ',');
                emit_hf(&sb, op->u.lut3d.domain_min[1]); toc_sb_putc(&sb, ',');
                emit_hf(&sb, op->u.lut3d.domain_min[2]);
                toc_sb_puts(&sb, "},dx[3]={");
                emit_hf(&sb, op->u.lut3d.domain_max[0]); toc_sb_putc(&sb, ',');
                emit_hf(&sb, op->u.lut3d.domain_max[1]); toc_sb_putc(&sb, ',');
                emit_hf(&sb, op->u.lut3d.domain_max[2]);
                toc_sb_puts(&sb, "}; tc_lut3d(tc_lut");
                toc_sb_int(&sb, (int)k);
                toc_sb_puts(&sb, ", ");
                toc_sb_int(&sb, op->u.lut3d.size);
                toc_sb_puts(&sb, ", dn, dx, ");
                toc_sb_int(&sb,
                           op->u.lut3d.interp == TOC_INTERP_TETRAHEDRAL ? 1 : 0);
                toc_sb_puts(&sb, ", c); r=c[0];g=c[1];b=c[2]; }\n");
                break;
            default: break;
        }
    }
    toc_sb_puts(&sb, "  px[0]=r;px[1]=g;px[2]=b;px[3]=a;\n  }\n}\n");
    if (sb.oom) { toc_sb_free(&sb); return TOC_ERROR_OUT_OF_MEMORY; }
    *out_src = toc_sb_take(&sb, out_len);
    return *out_src ? TOC_SUCCESS : TOC_ERROR_OUT_OF_MEMORY;
}
