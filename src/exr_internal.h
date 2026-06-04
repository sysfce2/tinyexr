/*
 * TinyEXR - internal shared declarations (not installed).
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef TINYEXR_INTERNAL_H_
#define TINYEXR_INTERNAL_H_

#include "exr.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ============================================================================
 * EXR format constants
 * ========================================================================== */

#define EXR_MAGIC 20000630 /* 0x01312f76 */
#define EXR_VERSION_NUMBER 2

/* Version field flag bits (byte 1..3 of the 4-byte version word). */
#define EXR_VERSION_FLAG_TILED (1 << 9)
#define EXR_VERSION_FLAG_LONG_NAMES (1 << 10)
#define EXR_VERSION_FLAG_NON_IMAGE (1 << 11) /* deep data present */
#define EXR_VERSION_FLAG_MULTIPART (1 << 12)

/* Hard safety limits to bound allocations from hostile files. */
#define EXR_MAX_PARTS 1048576
#define EXR_MAX_CHANNELS 1048576
#define EXR_MAX_ATTRIBUTES 1048576
#define EXR_MAX_ATTR_SIZE (256u * 1024u * 1024u)
#define EXR_MAX_DIMENSION (1 << 20) /* 1,048,576 px per axis */

/* ============================================================================
 * Checked integer arithmetic (size_t domain)
 * ========================================================================== */

static inline int exr_mul_ovf(size_t a, size_t b, size_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > SIZE_MAX / a) return 1;
    *out = a * b;
    return 0;
#endif
}

static inline int exr_add_ovf(size_t a, size_t b, size_t *out) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, out);
#else
    if (b > SIZE_MAX - a) return 1;
    *out = a + b;
    return 0;
#endif
}

/* ============================================================================
 * Little-endian readers / writers (EXR is always little-endian on disk)
 * ========================================================================== */

static inline uint16_t exr_rd_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static inline uint32_t exr_rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static inline int32_t exr_rd_i32(const uint8_t *p) {
    return (int32_t)exr_rd_u32(p);
}
static inline uint64_t exr_rd_u64(const uint8_t *p) {
    return (uint64_t)exr_rd_u32(p) | ((uint64_t)exr_rd_u32(p + 4) << 32);
}
static inline float exr_rd_f32(const uint8_t *p) {
    float f;
    uint32_t u = exr_rd_u32(p);
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline void exr_wr_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}
static inline void exr_wr_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static inline void exr_wr_i32(uint8_t *p, int32_t v) { exr_wr_u32(p, (uint32_t)v); }
static inline void exr_wr_u64(uint8_t *p, uint64_t v) {
    exr_wr_u32(p, (uint32_t)v);
    exr_wr_u32(p + 4, (uint32_t)(v >> 32));
}
static inline void exr_wr_f32(uint8_t *p, float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    exr_wr_u32(p, u);
}

/* ============================================================================
 * Allocator helpers
 * ========================================================================== */

const exr_allocator *exr_default_allocator(void);
void *exr_malloc(const exr_allocator *a, size_t size);
void *exr_calloc(const exr_allocator *a, size_t count, size_t size);
void exr_free(const exr_allocator *a, void *ptr);
char *exr_strdup(const exr_allocator *a, const char *s);

/* ============================================================================
 * SIMD dispatch
 * ========================================================================== */

/* Runtime-detected CPU features (bit flags mirror exr_simd_caps). */
uint32_t exr_cpu_caps(void);

/* Function-pointer table populated once by exr_simd_init() (idempotent). */
typedef struct {
    void (*half_to_float)(const uint16_t *src, float *dst, size_t count);
    void (*float_to_half)(const float *src, uint16_t *dst, size_t count);
    void (*interleave)(const uint8_t *src, uint8_t *dst, size_t n); /* de-split */
} exr_simd_vtbl;

extern exr_simd_vtbl exr_simd;
void exr_simd_init(void);
/* Force a kernel tier for benchmarking: 0=scalar, 1=sse2/neon, 2=avx2/f16c. */
void exr_simd_force(int level);

/* Scalar kernels (always built). */
void exr_half_to_float_scalar(const uint16_t *src, float *dst, size_t count);
void exr_float_to_half_scalar(const float *src, uint16_t *dst, size_t count);
void exr_interleave_scalar(const uint8_t *src, uint8_t *dst, size_t n);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define EXR_X86 1
void exr_interleave_sse2(const uint8_t *src, uint8_t *dst, size_t n);
void exr_interleave_avx2(const uint8_t *src, uint8_t *dst, size_t n);
void exr_half_to_float_f16c(const uint16_t *src, float *dst, size_t count);
void exr_float_to_half_f16c(const float *src, uint16_t *dst, size_t count);
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define EXR_NEON 1
void exr_interleave_neon(const uint8_t *src, uint8_t *dst, size_t n);
#endif

/* ============================================================================
 * Pixel helpers
 * ========================================================================== */

static inline size_t exr_pixel_size(exr_pixel_type t) {
    return (t == EXR_PIXEL_HALF) ? 2u : 4u;
}

/* Floor division and OpenEXR sample-count over an inclusive range [lo,hi]. */
static inline int64_t exr_floordiv(int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) q--;
    return q;
}
static inline int exr_num_samples(int lo, int hi, int s) {
    if (s <= 1) return hi - lo + 1;
    return (int)(exr_floordiv(hi, s) - exr_floordiv((int64_t)lo - 1, s));
}

/* Scanlines compressed together per chunk for a given compression. */
int exr_lines_per_block(exr_compression c);

/* ============================================================================
 * Attribute list (opaque exr_attr_list defined here)
 * ========================================================================== */

typedef struct exr_attr {
    char *name;      /* owned */
    char *type_name; /* owned */
    uint8_t *data;   /* owned raw bytes */
    uint32_t size;
} exr_attr;

struct exr_attr_list {
    exr_attr *items;
    uint32_t count;
    uint32_t capacity;
};

void exr_attr_list_free(const exr_allocator *a, exr_attr_list *list);
const exr_attr *exr_attr_find(const exr_attr_list *list, const char *name);

/* Parse one channel-list ("chlist") attribute into a freshly allocated channel
 * array. Returns EXR_SUCCESS and sets *out_channels / *out_count. */
exr_result exr_parse_chlist(const exr_allocator *a, const uint8_t *data,
                            uint32_t size, exr_channel **out_channels,
                            int32_t *out_count);

/* ============================================================================
 * Header deep-copy / free
 * ========================================================================== */

exr_result exr_header_copy(const exr_allocator *a, exr_header *dst,
                           const exr_header *src);
void exr_header_free(const exr_allocator *a, exr_header *hdr);

/* ============================================================================
 * Internal per-part state held by the reader
 * ========================================================================== */

typedef struct exr_int_part {
    exr_header header; /* owns channels + attrs */

    uint64_t *offsets; /* chunk offset table (file byte offsets) */
    uint32_t num_chunks;

    /* geometry */
    int32_t width;  /* data window width  */
    int32_t height; /* data window height */

    /* tiling (valid when header.tiled) */
    int32_t num_x_levels;
    int32_t num_y_levels;
    /* per-level pixel sizes and tile counts; allocated when tiled. */
    int32_t *level_width;   /* [num_x_levels] */
    int32_t *level_height;  /* [num_y_levels] */
    int32_t *level_x_tiles; /* [num_x_levels] tiles across */
    int32_t *level_y_tiles; /* [num_y_levels] tiles down */
} exr_int_part;

/* ============================================================================
 * Reader (shared between exr_core.c and exr_reader.c)
 * ========================================================================== */

typedef enum exr_src_kind {
    EXR_SRC_MEMORY = 0,
    EXR_SRC_CALLBACK = 1
} exr_src_kind;

struct exr_reader {
    exr_allocator alloc;

    exr_src_kind kind;
    const uint8_t *mem; /* memory path: whole file (may be owned) */
    size_t mem_size;
    int free_mem; /* if set, exr_reader_close frees (void*)mem */
    exr_data_source src; /* callback path */

    int parsed;
    uint32_t version_flags;
    int is_tiled;
    int is_multipart;
    int is_deep;
    int long_names;

    exr_int_part *parts;
    int32_t num_parts;

    /* streaming suspend state (callback path). */
    int have_pending;
    exr_pending_read pending;
};

/*
 * Random-access read of [offset, offset+size) from the source.
 *   - memory path: returns a direct pointer (zero-copy) via *out_ptr, *out_ptr
 *     stays valid for the reader lifetime. scratch is ignored.
 *   - callback path: copies into scratch (caller-provided, >= size) and sets
 *     *out_ptr = scratch.
 * Returns EXR_WOULD_BLOCK if the callback path needs bytes (Phase 9).
 */
exr_result exr_reader_fetch(exr_reader *r, uint64_t offset, size_t size,
                            void *scratch, const uint8_t **out_ptr);

/* ============================================================================
 * Codec layer (exr_codec.c and friends). Each decodes one chunk's raw
 * compressed bytes into the canonical uncompressed scanline/tile block.
 * `channels` is the part's sorted channel list; the block layout is, per
 * scanline then per channel, sample data of width/x_sampling samples.
 * ========================================================================== */

typedef struct exr_codec_ctx {
    const exr_allocator *alloc;
    exr_compression compression;
    const exr_channel *channels;
    int32_t num_channels;
    int32_t x; /* data window x origin (for sampling alignment) */
    int32_t y; /* first scanline of this block (data window coords) */
    int32_t width;
    int32_t num_lines; /* scanlines in this block */
} exr_codec_ctx;

/* Decompress `src_size` bytes from `src` into `dst` (capacity dst_size, which
 * must equal the exact uncompressed block size). */
exr_result exr_decompress_block(const exr_codec_ctx *ctx, const uint8_t *src,
                                size_t src_size, uint8_t *dst, size_t dst_size);

/* Uncompressed byte size of one scanline block / tile region. */
exr_result exr_block_uncompressed_size(const exr_channel *channels,
                                       int32_t num_channels, int32_t x,
                                       int32_t y, int32_t width,
                                       int32_t num_lines, size_t *out_size);

/* Individual codecs (decode). ZIP/RLE need scratch for the pre-reconstruction
 * buffer, hence the allocator. */
exr_result exr_rle_decompress(const exr_allocator *a, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size);
exr_result exr_zip_decompress(const exr_allocator *a, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size);
exr_result exr_pxr24_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                                size_t src_size, uint8_t *dst, size_t dst_size);
exr_result exr_piz_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size);
exr_result exr_b44_decompress(const exr_codec_ctx *ctx, const uint8_t *src,
                              size_t src_size, uint8_t *dst, size_t dst_size,
                              int optimize_flat);

/* Raw zlib (DEFLATE) inflate used by ZIP/ZIPS/PXR24. Returns the number of
 * decoded bytes in *out_size; fails on truncation/corruption. */
exr_result exr_inflate_zlib(const uint8_t *src, size_t src_size, uint8_t *dst,
                            size_t dst_cap, size_t *out_size);

/* Apply EXR's two post-DEFLATE reconstruction passes (predictor + byte
 * interleave) in place over `n` bytes. */
void exr_predictor_decode(uint8_t *p, size_t n);
void exr_interleave_decode(const uint8_t *src, uint8_t *dst, size_t n);

/* Forward (encode-side) inverses: byte split, then delta predictor. */
void exr_interleave_encode(const uint8_t *src, uint8_t *dst, size_t n);
void exr_predictor_encode(uint8_t *p, size_t n);

/* Raw zlib (DEFLATE) compress. Allocates *out_data (caller frees with the same
 * allocator); *out_size is the exact compressed length. */
exr_result exr_deflate_zlib(const exr_allocator *a, const uint8_t *src,
                            size_t n, uint8_t **out_data, size_t *out_size);

/* Adler-32 (zlib trailer). */
uint32_t exr_adler32(const uint8_t *data, size_t n, uint32_t adler);

/* fpnge-derived literal DEFLATE encoder with a PSHUFB Huffman-table lookup.
 * The PSHUFB-friendly table: symbols 0-15 and 240-255 are looked up by low
 * nibble, and 16-239 share one code length (so the code is computable from the
 * byte's nibbles). */
typedef struct {
    uint8_t nbits[286];
    uint16_t end_bits; /* end-of-block (symbol 256) code */
    uint8_t first16_nbits[16], first16_bits[16];
    uint8_t last16_nbits[16], last16_bits[16];
    uint8_t mid_lowbits[16];
    uint8_t mid_nbits;
} exr_fpnge_table;

/* Build the PSHUFB-friendly table from per-symbol frequencies[286]. */
int exr_fpnge_build_table(const exr_allocator *a, const uint64_t *collected,
                          exr_fpnge_table *t);

/* Per-byte Huffman lookup: fills nb[count] and the 16-bit code as blo|bhi<<8.
 * Scalar reference plus SSE4.1/AVX2 PSHUFB kernels (selected by exr_fpnge_deflate). */
void exr_fpnge_lookup_scalar(const exr_fpnge_table *t, const uint8_t *src,
                             size_t count, uint8_t *nb, uint8_t *blo, uint8_t *bhi);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void exr_fpnge_lookup_sse41(const exr_fpnge_table *t, const uint8_t *src,
                            size_t count, uint8_t *nb, uint8_t *blo, uint8_t *bhi);
void exr_fpnge_lookup_avx2(const exr_fpnge_table *t, const uint8_t *src,
                           size_t count, uint8_t *nb, uint8_t *blo, uint8_t *bhi);
#endif

/* Literal-only zlib stream over src[0..n). use_simd selects the PSHUFB lookup
 * when SSE4.1 is available. Allocates *out_data (caller frees). */
exr_result exr_fpnge_deflate(const exr_allocator *a, const uint8_t *src, size_t n,
                             uint8_t **out_data, size_t *out_size, int use_simd);

/* Codec encoders (compress one canonical block). Each allocates *out_data
 * (caller frees) holding the chunk payload; if compression does not shrink the
 * data the codec stores it verbatim and *out_size == src_size. */
exr_result exr_rle_compress(const exr_allocator *a, const uint8_t *src,
                            size_t n, uint8_t **out_data, size_t *out_size);
exr_result exr_zip_compress(const exr_allocator *a, const uint8_t *src,
                            size_t n, uint8_t **out_data, size_t *out_size);
exr_result exr_piz_compress(const exr_codec_ctx *ctx, const uint8_t *block,
                            size_t n, uint8_t **out_data, size_t *out_size);

/* Encode dispatch: compress one canonical block per ctx->compression. */
exr_result exr_compress_block(const exr_codec_ctx *ctx, const uint8_t *block,
                              size_t n, uint8_t **out_data, size_t *out_size);

#endif /* TINYEXR_INTERNAL_H_ */
