/*
 * TinyEXR private Zstandard wrapper declarations.
 *
 * The implementation is generated from upstream zstd and uses zstd's BSD
 * license option. See deps/zstd/LICENSE.
 */

#ifndef TINYEXR_ZSTD_H_
#define TINYEXR_ZSTD_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t tinyexr_zstd_compress(void *dst, size_t dst_capacity, const void *src,
                             size_t src_size, int compression_level);
size_t tinyexr_zstd_decompress(void *dst, size_t dst_capacity, const void *src,
                               size_t src_size);
size_t tinyexr_zstd_compress_bound(size_t src_size);
unsigned tinyexr_zstd_is_error(size_t code);

#ifdef __cplusplus
}
#endif

#endif /* TINYEXR_ZSTD_H_ */
