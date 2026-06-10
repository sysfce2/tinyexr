/*
 * tocio - x86-64 JIT backend. Emits SSE2 machine code for an op chain into
 * executable memory and runs it directly (no disk / dlopen). Hosted-only
 * (needs OS executable memory via mmap); excluded from the freestanding core.
 *
 * The emitted function is `void fn(float *rgba, size_t npix)` over interleaved
 * RGBA. matrix/range are inlined as SSE (bit-exact with the SSE2 interpreter
 * tier); every other op calls toc_apply_op_pixel(op, px, ch) with the op's
 * address baked in. `channels` is baked at compile time.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "toc_internal.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <stddef.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#define TOC_HAVE_MMAP 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

struct toc_jit {
    void *mem;     /* executable mapping */
    size_t mapsz;  /* rounded map size */
    toc_jit_fn fn;
    toc_allocator alloc;
};

/* ---- growable code buffer ------------------------------------------------ */
typedef struct {
    uint8_t *buf;
    size_t len, cap;
    const toc_allocator *a;
    int oom;
} codebuf;

static int cb_reserve(codebuf *c, size_t extra) {
    if (c->oom) return 0;
    if (c->len + extra <= c->cap) return 1;
    {
        size_t ncap = c->cap ? c->cap * 2 : 1024;
        uint8_t *n;
        while (ncap < c->len + extra) ncap *= 2;
        n = (uint8_t *)toc_malloc(c->a, ncap);
        if (!n) { c->oom = 1; return 0; }
        if (c->buf) { memcpy(n, c->buf, c->len); toc_free(c->a, c->buf); }
        c->buf = n;
        c->cap = ncap;
    }
    return 1;
}
static void e1(codebuf *c, unsigned b) {
    if (cb_reserve(c, 1)) c->buf[c->len++] = (uint8_t)b;
}
static void en(codebuf *c, const uint8_t *p, size_t n) {
    if (cb_reserve(c, n)) { memcpy(c->buf + c->len, p, n); c->len += n; }
}
static void e4(codebuf *c, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16),
                    (uint8_t)(v >> 24)};
    en(c, b, 4);
}
static void e8(codebuf *c, uint64_t v) {
    uint8_t b[8];
    int i;
    for (i = 0; i < 8; ++i) b[i] = (uint8_t)(v >> (8 * i));
    en(c, b, 8);
}

/* ---- instruction emitters (SSE2; xmm regs limited to 0,1,5,6 = no REX) --- */
static void mov_rax_imm64(codebuf *c, uint64_t v) { e1(c, 0x48); e1(c, 0xB8); e8(c, v); }
static void mov_rdi_imm64(codebuf *c, uint64_t v) { e1(c, 0x48); e1(c, 0xBF); e8(c, v); }
static void mov_edx_imm32(codebuf *c, uint32_t v) { e1(c, 0xBA); e4(c, v); }
static void mov_rsi_r12(codebuf *c) { e1(c, 0x4C); e1(c, 0x89); e1(c, 0xE6); }
static void call_rax(codebuf *c) { e1(c, 0xFF); e1(c, 0xD0); }

/* movups xmm<reg>, [rax + disp32] */
static void ld_xmm_rax(codebuf *c, int reg, uint32_t disp) {
    e1(c, 0x0F); e1(c, 0x10);
    e1(c, 0x80u | ((unsigned)reg << 3)); /* mod=10, rm=000 (rax) */
    e4(c, disp);
}
/* movups xmm<reg>, [r12]  /  movups [r12], xmm<reg> */
static void ld_xmm_r12(codebuf *c, int reg) {
    e1(c, 0x41); e1(c, 0x0F); e1(c, 0x10);
    e1(c, 0x04u | ((unsigned)reg << 3)); e1(c, 0x24);
}
static void st_xmm_r12(codebuf *c, int reg) {
    e1(c, 0x41); e1(c, 0x0F); e1(c, 0x11);
    e1(c, 0x04u | ((unsigned)reg << 3)); e1(c, 0x24);
}
static void pshufd(codebuf *c, int dst, int src, unsigned imm) {
    e1(c, 0x66); e1(c, 0x0F); e1(c, 0x70);
    e1(c, 0xC0u | ((unsigned)dst << 3) | (unsigned)src); e1(c, imm);
}
static void sse_rr(codebuf *c, unsigned op2, int dst, int src) {
    e1(c, 0x0F); e1(c, op2);
    e1(c, 0xC0u | ((unsigned)dst << 3) | (unsigned)src);
}
/* 66-prefixed packed-integer reg,reg form. */
static void sse66_rr(codebuf *c, unsigned op2, int dst, int src) {
    e1(c, 0x66); e1(c, 0x0F); e1(c, op2);
    e1(c, 0xC0u | ((unsigned)dst << 3) | (unsigned)src);
}
#define mulps(c, d, s) sse_rr(c, 0x59, d, s)
#define addps(c, d, s) sse_rr(c, 0x58, d, s)
#define subps(c, d, s) sse_rr(c, 0x5C, d, s)
#define divps(c, d, s) sse_rr(c, 0x5E, d, s)
#define maxps(c, d, s) sse_rr(c, 0x5F, d, s)
#define minps(c, d, s) sse_rr(c, 0x5D, d, s)
#define movaps(c, d, s) sse_rr(c, 0x28, d, s)
#define cvtdq2ps(c, d, s) sse_rr(c, 0x5B, d, s)
#define cvtps2dq(c, d, s) sse66_rr(c, 0x5B, d, s)
#define pand(c, d, s) sse66_rr(c, 0xDB, d, s)
#define por(c, d, s) sse66_rr(c, 0xEB, d, s)
#define paddd(c, d, s) sse66_rr(c, 0xFE, d, s)
#define psubd(c, d, s) sse66_rr(c, 0xFA, d, s)
/* psrld/pslld xmm, imm8 (66 0F 72 /2 or /6) */
static void psrld_i(codebuf *c, int reg, unsigned imm) {
    e1(c, 0x66); e1(c, 0x0F); e1(c, 0x72);
    e1(c, 0xD0u | (unsigned)reg); e1(c, imm);
}
static void pslld_i(codebuf *c, int reg, unsigned imm) {
    e1(c, 0x66); e1(c, 0x0F); e1(c, 0x72);
    e1(c, 0xF0u | (unsigned)reg); e1(c, imm);
}
/* broadcast a 32-bit constant to all 4 lanes of xmm<reg> via eax. */
static void bcast_bits(codebuf *c, int reg, uint32_t bits) {
    e1(c, 0xB8); e4(c, bits);                 /* mov eax, imm32 */
    e1(c, 0x66); e1(c, 0x0F); e1(c, 0x6E);    /* movd xmm<reg>, eax */
    e1(c, 0xC0u | ((unsigned)reg << 3));
    e1(c, 0x66); e1(c, 0x0F); e1(c, 0x70);    /* pshufd xmm,xmm,0 */
    e1(c, 0xC0u | ((unsigned)reg << 3) | (unsigned)reg); e1(c, 0x00);
}
static void bcast_f(codebuf *c, int reg, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    bcast_bits(c, reg, u);
}

static uint64_t addr_of(const void *p) { return (uint64_t)(uintptr_t)p; }

/* 4-wide log2 in place on xmm1; scratch xmm2,xmm3,xmm4,xmm7. Matches the
 * scalar toc_log2f formulation (atanh series). */
static void emit_log2_xmm1(codebuf *c) {
    movaps(c, 2, 1);                 /* xmm2 = bits copy for exponent */
    psrld_i(c, 2, 23);
    bcast_bits(c, 7, 0xffu); pand(c, 2, 7);
    bcast_bits(c, 7, 127u);  psubd(c, 2, 7);
    cvtdq2ps(c, 2, 2);               /* xmm2 = e_f */
    bcast_bits(c, 7, 0x7fffffu); pand(c, 1, 7);
    bcast_bits(c, 7, 0x3f800000u); por(c, 1, 7);  /* xmm1 = m in [1,2) */
    bcast_f(c, 7, 1.0f);
    movaps(c, 3, 1); subps(c, 3, 7); /* m-1 */
    addps(c, 1, 7);                  /* m+1 */
    divps(c, 3, 1);                  /* xmm3 = t */
    movaps(c, 4, 3); mulps(c, 4, 4); /* xmm4 = t2 */
    bcast_f(c, 1, 1.0f / 9.0f);
    mulps(c, 1, 4); bcast_f(c, 7, 1.0f / 7.0f); addps(c, 1, 7);
    mulps(c, 1, 4); bcast_f(c, 7, 1.0f / 5.0f); addps(c, 1, 7);
    mulps(c, 1, 4); bcast_f(c, 7, 1.0f / 3.0f); addps(c, 1, 7);
    mulps(c, 1, 4); bcast_f(c, 7, 1.0f);        addps(c, 1, 7); /* poly */
    bcast_f(c, 7, 2.0f); mulps(c, 3, 7);        /* 2t */
    mulps(c, 1, 3);                              /* ln = poly*2t */
    bcast_f(c, 7, 1.4426950408889634f); mulps(c, 1, 7);
    addps(c, 1, 2);                  /* xmm1 = log2 */
}

/* 4-wide exp2 in place on xmm1; scratch xmm2,xmm3,xmm4,xmm5,xmm7. */
static void emit_exp2_xmm1(codebuf *c) {
    bcast_f(c, 7, 127.0f);  minps(c, 1, 7);
    bcast_f(c, 7, -126.0f); maxps(c, 1, 7);
    cvtps2dq(c, 2, 1);               /* xmm2 = k_int (round nearest) */
    cvtdq2ps(c, 3, 2);               /* xmm3 = k_f */
    movaps(c, 4, 1); subps(c, 4, 3); /* xmm4 = f */
    bcast_f(c, 7, 0.6931471805599453f); mulps(c, 4, 7); /* g = f*ln2 */
    bcast_f(c, 5, 1.0f / 720.0f);
    mulps(c, 5, 4); bcast_f(c, 7, 1.0f / 120.0f); addps(c, 5, 7);
    mulps(c, 5, 4); bcast_f(c, 7, 1.0f / 24.0f);  addps(c, 5, 7);
    mulps(c, 5, 4); bcast_f(c, 7, 1.0f / 6.0f);   addps(c, 5, 7);
    mulps(c, 5, 4); bcast_f(c, 7, 0.5f);          addps(c, 5, 7);
    mulps(c, 5, 4); bcast_f(c, 7, 1.0f);          addps(c, 5, 7);
    mulps(c, 5, 4); bcast_f(c, 7, 1.0f);          addps(c, 5, 7); /* p */
    bcast_bits(c, 7, 127u); paddd(c, 2, 7); pslld_i(c, 2, 23); /* 2^k bits */
    mulps(c, 5, 2);                  /* p * 2^k */
    movaps(c, 1, 5);                 /* xmm1 = exp2 */
}

/* dispatch one op via the SSE (1-pixel) path */
static void emit_op_sse(codebuf *c, const toc_op *op, int channels);

/* EXPONENT (ch==4): v = pow(max(0,v), e[4]) per lane. NOTE bcast_* clobbers
 * eax (= low 32 of rax), so the op pointer is reloaded after the log2 pass. */
static void emit_exponent(codebuf *c, const toc_op *op) {
    uint32_t E = (uint32_t)offsetof(toc_op, u.exponent.e);
    ld_xmm_r12(c, 0);
    bcast_f(c, 7, 0.0f); maxps(c, 0, 7);             /* max(v,0) */
    bcast_f(c, 7, 1.17549435e-38f); maxps(c, 0, 7);  /* >= FLT_MIN for log2 */
    movaps(c, 1, 0);
    emit_log2_xmm1(c);
    mov_rax_imm64(c, addr_of(op));                   /* reload (bcast hit eax) */
    ld_xmm_rax(c, 2, E); mulps(c, 1, 2);             /* y = e*log2(v) */
    emit_exp2_xmm1(c);
    movaps(c, 0, 1);
    st_xmm_r12(c, 0);
}

static void emit_matrix(codebuf *c, const toc_op *op) {
    uint32_t M = (uint32_t)offsetof(toc_op, u.matrix.m);
    uint32_t O = (uint32_t)offsetof(toc_op, u.matrix.off);
    mov_rax_imm64(c, addr_of(op));
    ld_xmm_r12(c, 0);            /* xmm0 = v(r,g,b,a) */
    pshufd(c, 1, 0, 0x00);       /* xmm1 = r */
    ld_xmm_rax(c, 5, M + 0);  mulps(c, 5, 1);
    pshufd(c, 1, 0, 0x55);       /* g */
    ld_xmm_rax(c, 6, M + 16); mulps(c, 6, 1); addps(c, 5, 6);
    pshufd(c, 1, 0, 0xAA);       /* b */
    ld_xmm_rax(c, 6, M + 32); mulps(c, 6, 1); addps(c, 5, 6);
    pshufd(c, 1, 0, 0xFF);       /* a */
    ld_xmm_rax(c, 6, M + 48); mulps(c, 6, 1); addps(c, 5, 6);
    ld_xmm_rax(c, 6, O);      addps(c, 5, 6);
    st_xmm_r12(c, 5);
}

/* ---- AVX (256-bit, 2 pixels/iter) encoders + kernels -------------------- */
/* vmovups ymm<r>,[r12] / [r12],ymm<r> (3-byte VEX, B bit for r12 base) */
static void vld_ymm_r12(codebuf *c, int r) {
    e1(c, 0xC4); e1(c, 0xC1); e1(c, 0x7C); e1(c, 0x10);
    e1(c, 0x04u | ((unsigned)r << 3)); e1(c, 0x24);
}
static void vst_ymm_r12(codebuf *c, int r) {
    e1(c, 0xC4); e1(c, 0xC1); e1(c, 0x7C); e1(c, 0x11);
    e1(c, 0x04u | ((unsigned)r << 3)); e1(c, 0x24);
}
/* vbroadcastf128 ymm<r>, [rax+disp32] */
static void vbcast128_rax(codebuf *c, int r, uint32_t disp) {
    e1(c, 0xC4); e1(c, 0xE2); e1(c, 0x7D); e1(c, 0x1A);
    e1(c, 0x80u | ((unsigned)r << 3)); e4(c, disp);
}
/* vpermilps ymm<d>, ymm<s>, imm8 */
static void vpermilps(codebuf *c, int d, int s, unsigned imm) {
    e1(c, 0xC4); e1(c, 0xE3); e1(c, 0x7D); e1(c, 0x04);
    e1(c, 0xC0u | ((unsigned)d << 3) | (unsigned)s); e1(c, imm);
}
/* 3-operand 256-bit float op (C5 form): dst = op(src1, src2). */
static void vex_rrr(codebuf *c, unsigned op2, int d, int s1, int s2) {
    e1(c, 0xC5);
    e1(c, 0x80u | (((~(unsigned)s1) & 0xfu) << 3) | 0x04u); /* vvvv=~s1, L=1 */
    e1(c, op2);
    e1(c, 0xC0u | ((unsigned)d << 3) | (unsigned)s2);
}
#define vmulps(c, d, a, b) vex_rrr(c, 0x59, d, a, b)
#define vaddps(c, d, a, b) vex_rrr(c, 0x58, d, a, b)
#define vmaxps(c, d, a, b) vex_rrr(c, 0x5F, d, a, b)
#define vminps(c, d, a, b) vex_rrr(c, 0x5D, d, a, b)
static void vzeroupper(codebuf *c) { e1(c, 0xC5); e1(c, 0xF8); e1(c, 0x77); }

static void emit_matrix_avx2(codebuf *c, const toc_op *op) {
    uint32_t M = (uint32_t)offsetof(toc_op, u.matrix.m);
    uint32_t O = (uint32_t)offsetof(toc_op, u.matrix.off);
    mov_rax_imm64(c, addr_of(op));
    vld_ymm_r12(c, 0);                 /* ymm0 = 2 pixels */
    vpermilps(c, 1, 0, 0x00); vbcast128_rax(c, 5, M + 0);  vmulps(c, 5, 5, 1);
    vpermilps(c, 1, 0, 0x55); vbcast128_rax(c, 6, M + 16); vmulps(c, 6, 6, 1); vaddps(c, 5, 5, 6);
    vpermilps(c, 1, 0, 0xAA); vbcast128_rax(c, 6, M + 32); vmulps(c, 6, 6, 1); vaddps(c, 5, 5, 6);
    vpermilps(c, 1, 0, 0xFF); vbcast128_rax(c, 6, M + 48); vmulps(c, 6, 6, 1); vaddps(c, 5, 5, 6);
    vbcast128_rax(c, 6, O); vaddps(c, 5, 5, 6);
    vst_ymm_r12(c, 5);
}

static void emit_range_avx2(codebuf *c, const toc_op *op) {
    uint32_t S = (uint32_t)offsetof(toc_op, u.range.scale);
    uint32_t OF = (uint32_t)offsetof(toc_op, u.range.offset);
    uint32_t MN = (uint32_t)offsetof(toc_op, u.range.min);
    uint32_t MX = (uint32_t)offsetof(toc_op, u.range.max);
    mov_rax_imm64(c, addr_of(op));
    vld_ymm_r12(c, 0);
    vbcast128_rax(c, 1, S);  vmulps(c, 0, 0, 1);
    vbcast128_rax(c, 1, OF); vaddps(c, 0, 0, 1);
    if (op->u.range.clamp_lo) { vbcast128_rax(c, 1, MN); vmaxps(c, 0, 0, 1); }
    if (op->u.range.clamp_hi) { vbcast128_rax(c, 1, MX); vminps(c, 0, 0, 1); }
    vst_ymm_r12(c, 0);
}

#if defined(__GNUC__) || defined(__clang__)
static int host_has_avx(void) {
    unsigned a, b, cc, d, eax, edx;
    if (!__get_cpuid(1, &a, &b, &cc, &d)) return 0;
    if (!(cc & (1u << 27)) || !(cc & (1u << 28))) return 0; /* OSXSAVE + AVX */
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((eax & 0x6u) == 0x6u) ? 1 : 0; /* XMM+YMM enabled */
}
#else
static int host_has_avx(void) { return 0; }
#endif

static void emit_range(codebuf *c, const toc_op *op) {
    uint32_t S = (uint32_t)offsetof(toc_op, u.range.scale);
    uint32_t OF = (uint32_t)offsetof(toc_op, u.range.offset);
    uint32_t MN = (uint32_t)offsetof(toc_op, u.range.min);
    uint32_t MX = (uint32_t)offsetof(toc_op, u.range.max);
    mov_rax_imm64(c, addr_of(op));
    ld_xmm_r12(c, 0);
    ld_xmm_rax(c, 1, S);  mulps(c, 0, 1);
    ld_xmm_rax(c, 1, OF); addps(c, 0, 1);
    if (op->u.range.clamp_lo) { ld_xmm_rax(c, 1, MN); maxps(c, 0, 1); }
    if (op->u.range.clamp_hi) { ld_xmm_rax(c, 1, MX); minps(c, 0, 1); }
    st_xmm_r12(c, 0);
}

static void patch_rel32(codebuf *c, size_t site, size_t target) {
    int32_t r = (int32_t)((int64_t)target - (int64_t)(site + 4));
    if (c->oom) return;
    c->buf[site] = (uint8_t)r; c->buf[site + 1] = (uint8_t)(r >> 8);
    c->buf[site + 2] = (uint8_t)(r >> 16); c->buf[site + 3] = (uint8_t)(r >> 24);
}

/* call toc_apply_op_pixel(op, r12, ch) */
static void emit_helper_call(codebuf *c, const toc_op *op, int ch) {
    mov_rdi_imm64(c, addr_of(op));
    mov_rsi_r12(c);
    mov_edx_imm32(c, (uint32_t)ch);
    mov_rax_imm64(c, addr_of((const void *)&toc_apply_op_pixel));
    call_rax(c);
}

static void emit_op_sse(codebuf *c, const toc_op *op, int channels) {
    if (channels == 4 && op->kind == TOC_OP_MATRIX) emit_matrix(c, op);
    else if (channels == 4 && op->kind == TOC_OP_RANGE) emit_range(c, op);
    else if (channels == 4 && op->kind == TOC_OP_EXPONENT) emit_exponent(c, op);
    else if (op->kind != TOC_OP_NOOP) emit_helper_call(c, op, channels);
}

toc_result toc_jit_compile(const toc_op_list *ops, int channels,
                           const toc_allocator *a, toc_jit **out) {
    codebuf c;
    size_t k;
    int avx_pure;
    toc_jit *j;
#if !defined(TOC_HAVE_MMAP)
    (void)ops; (void)channels; (void)a; (void)out;
    return TOC_ERROR_UNSUPPORTED;
#else
    size_t pgsz = 4096, mapsz;
    void *mem;
    if (!ops || !out || (channels != 3 && channels != 4))
        return TOC_ERROR_INVALID_ARGUMENT;
    if (!a) a = toc_default_allocator();
    *out = NULL;
    memset(&c, 0, sizeof(c));
    c.a = a;

    /* prologue: save callee-saved, load args, align stack for calls.
     * SysV entry rsp%16==8; push r12,r13 then sub 8 -> rsp%16==0. */
    e1(&c, 0x41); e1(&c, 0x54);            /* push r12 */
    e1(&c, 0x41); e1(&c, 0x55);            /* push r13 */
    e1(&c, 0x48); e1(&c, 0x83); e1(&c, 0xEC); e1(&c, 0x08); /* sub rsp,8 */
    e1(&c, 0x49); e1(&c, 0x89); e1(&c, 0xFC); /* mov r12, rdi (px = rgba) */
    e1(&c, 0x49); e1(&c, 0x89); e1(&c, 0xF5); /* mov r13, rsi (counter = npix) */

    /* Use the AVX 2-pixel path only for pure matrix/range pipelines (the common
     * colorspace-conversion case): it avoids mixing AVX/SSE inline and keeps the
     * inline-pow SSE path for everything else. */
    avx_pure = (channels == 4 && host_has_avx());
    for (k = 0; avx_pure && k < ops->count; ++k) {
        toc_op_kind kk = ops->ops[k].kind;
        if (kk != TOC_OP_MATRIX && kk != TOC_OP_RANGE && kk != TOC_OP_NOOP)
            avx_pure = 0;
    }

    if (avx_pure) {
        size_t main_pos, jb_at, jmp_at, jz_at;
        main_pos = c.len;
        e1(&c, 0x49); e1(&c, 0x83); e1(&c, 0xFD); e1(&c, 0x02); /* cmp r13,2 */
        e1(&c, 0x0F); e1(&c, 0x82); jb_at = c.len; e4(&c, 0);   /* jb -> tail */
        for (k = 0; k < ops->count; ++k) {
            const toc_op *op = &ops->ops[k];
            if (op->kind == TOC_OP_MATRIX) emit_matrix_avx2(&c, op);
            else if (op->kind == TOC_OP_RANGE) emit_range_avx2(&c, op);
        }
        e1(&c, 0x49); e1(&c, 0x83); e1(&c, 0xC4); e1(&c, 0x20); /* add r12,32 */
        e1(&c, 0x49); e1(&c, 0x83); e1(&c, 0xED); e1(&c, 0x02); /* sub r13,2 */
        e1(&c, 0xE9); jmp_at = c.len; e4(&c, 0);                /* jmp -> main */
        patch_rel32(&c, jmp_at, main_pos);
        patch_rel32(&c, jb_at, c.len);                          /* tail: */
        e1(&c, 0x4D); e1(&c, 0x85); e1(&c, 0xED);               /* test r13,r13 */
        e1(&c, 0x0F); e1(&c, 0x84); jz_at = c.len; e4(&c, 0);   /* jz -> done */
        for (k = 0; k < ops->count; ++k) emit_op_sse(&c, &ops->ops[k], channels);
        patch_rel32(&c, jz_at, c.len);                          /* done: */
        vzeroupper(&c);
    } else {
        size_t loop_pos, jz_at, jmp_at;
        loop_pos = c.len;
        e1(&c, 0x4D); e1(&c, 0x85); e1(&c, 0xED);               /* test r13,r13 */
        e1(&c, 0x0F); e1(&c, 0x84); jz_at = c.len; e4(&c, 0);   /* jz -> done */
        for (k = 0; k < ops->count; ++k) emit_op_sse(&c, &ops->ops[k], channels);
        e1(&c, 0x49); e1(&c, 0x83); e1(&c, 0xC4);               /* add r12,imm8 */
        e1(&c, (unsigned)(channels * 4));
        e1(&c, 0x49); e1(&c, 0xFF); e1(&c, 0xCD);               /* dec r13 */
        e1(&c, 0xE9); jmp_at = c.len; e4(&c, 0);                /* jmp -> loop */
        patch_rel32(&c, jmp_at, loop_pos);
        patch_rel32(&c, jz_at, c.len);                          /* done: */
    }

    e1(&c, 0x48); e1(&c, 0x83); e1(&c, 0xC4); e1(&c, 0x08); /* add rsp,8 */
    e1(&c, 0x41); e1(&c, 0x5D);            /* pop r13 */
    e1(&c, 0x41); e1(&c, 0x5C);            /* pop r12 */
    e1(&c, 0xC3);                          /* ret */

    if (c.oom) { toc_free(a, c.buf); return TOC_ERROR_OUT_OF_MEMORY; }

    /* copy into an executable mapping (W^X) */
    mapsz = (c.len + pgsz - 1) & ~(pgsz - 1);
    mem = mmap(NULL, mapsz, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) { toc_free(a, c.buf); return TOC_ERROR_UNSUPPORTED; }
    memcpy(mem, c.buf, c.len);
    toc_free(a, c.buf);
    if (mprotect(mem, mapsz, PROT_READ | PROT_EXEC) != 0) {
        munmap(mem, mapsz);
        return TOC_ERROR_UNSUPPORTED;
    }
    j = (toc_jit *)toc_malloc(a, sizeof(*j));
    if (!j) { munmap(mem, mapsz); return TOC_ERROR_OUT_OF_MEMORY; }
    j->mem = mem;
    j->mapsz = mapsz;
    j->alloc = *a;
    memcpy(&j->fn, &mem, sizeof(void *)); /* avoid object<->fn ptr warning */
    *out = j;
    return TOC_SUCCESS;
#endif
}

toc_jit_fn toc_jit_func(const toc_jit *j) { return j ? j->fn : NULL; }

void toc_jit_destroy(toc_jit *j) {
    toc_allocator a;
    if (!j) return;
    a = j->alloc;
#if defined(TOC_HAVE_MMAP)
    if (j->mem) munmap(j->mem, j->mapsz);
#endif
    toc_free(&a, j);
}

#else /* non x86-64 */

#include <stddef.h>
struct toc_jit { int unused; };
toc_result toc_jit_compile(const toc_op_list *ops, int channels,
                           const toc_allocator *a, toc_jit **out) {
    (void)ops; (void)channels; (void)a;
    if (out) *out = NULL;
    return TOC_ERROR_UNSUPPORTED;
}
toc_jit_fn toc_jit_func(const toc_jit *j) { (void)j; return NULL; }
void toc_jit_destroy(toc_jit *j) { (void)j; }

#endif
