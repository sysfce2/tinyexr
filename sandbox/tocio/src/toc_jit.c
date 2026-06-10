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
#define mulps(c, d, s) sse_rr(c, 0x59, d, s)
#define addps(c, d, s) sse_rr(c, 0x58, d, s)
#define maxps(c, d, s) sse_rr(c, 0x5F, d, s)
#define minps(c, d, s) sse_rr(c, 0x5D, d, s)

static uint64_t addr_of(const void *p) { return (uint64_t)(uintptr_t)p; }

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

/* call toc_apply_op_pixel(op, r12, ch) */
static void emit_helper_call(codebuf *c, const toc_op *op, int ch) {
    mov_rdi_imm64(c, addr_of(op));
    mov_rsi_r12(c);
    mov_edx_imm32(c, (uint32_t)ch);
    mov_rax_imm64(c, addr_of((const void *)&toc_apply_op_pixel));
    call_rax(c);
}

toc_result toc_jit_compile(const toc_op_list *ops, int channels,
                           const toc_allocator *a, toc_jit **out) {
    codebuf c;
    size_t k, loop_pos, jz_disp_at, jmp_rel;
    int32_t rel;
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

    loop_pos = c.len;
    e1(&c, 0x4D); e1(&c, 0x85); e1(&c, 0xED);  /* test r13, r13 */
    e1(&c, 0x0F); e1(&c, 0x84);                /* jz rel32 -> done */
    jz_disp_at = c.len;
    e4(&c, 0);                                 /* patched below */

    for (k = 0; k < ops->count; ++k) {
        const toc_op *op = &ops->ops[k];
        if (channels == 4 && op->kind == TOC_OP_MATRIX) emit_matrix(&c, op);
        else if (channels == 4 && op->kind == TOC_OP_RANGE) emit_range(&c, op);
        else if (op->kind != TOC_OP_NOOP) emit_helper_call(&c, op, channels);
    }

    e1(&c, 0x49); e1(&c, 0x83); e1(&c, 0xC4);  /* add r12, imm8 (stride) */
    e1(&c, (unsigned)(channels * 4));
    e1(&c, 0x49); e1(&c, 0xFF); e1(&c, 0xCD);  /* dec r13 */
    e1(&c, 0xE9);                              /* jmp rel32 -> loop */
    jmp_rel = c.len;
    e4(&c, 0);
    rel = (int32_t)((int64_t)loop_pos - (int64_t)(jmp_rel + 4));
    if (!c.oom) { c.buf[jmp_rel] = (uint8_t)rel; c.buf[jmp_rel + 1] = (uint8_t)(rel >> 8);
                  c.buf[jmp_rel + 2] = (uint8_t)(rel >> 16); c.buf[jmp_rel + 3] = (uint8_t)(rel >> 24); }

    /* done: patch the jz to here, then epilogue + ret */
    {
        int32_t r = (int32_t)((int64_t)c.len - (int64_t)(jz_disp_at + 4));
        if (!c.oom) { c.buf[jz_disp_at] = (uint8_t)r; c.buf[jz_disp_at + 1] = (uint8_t)(r >> 8);
                      c.buf[jz_disp_at + 2] = (uint8_t)(r >> 16); c.buf[jz_disp_at + 3] = (uint8_t)(r >> 24); }
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
    (void)jz_disp_at;
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
