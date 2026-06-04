# TinyEXR v3 benchmark

Standalone micro/throughput benchmark for the pure-C11 EXR codec.

```sh
make bench                 # build + run on asakusa.exr
./build/bench foo.exr bar.exr   # run on your own files
```

It reports three things:

1. **Codec throughput** — for each codec the writer supports
   (NONE/RLE/ZIPS/ZIP/PIZ), the encode and decode rate in megapixels/second
   and the compressed size. Decode covers all codecs in the file; the
   auto-generated set only exercises the writable ones (pass PXR24/B44/DWA
   files on the command line to time those decoders too).

2. **ZIP decode by forced SIMD tier** — the same ZIP decode run with the
   byte de-interleave kernel forced to scalar / SSE2 / AVX2 (via
   `exr_simd_force`), showing the end-to-end effect of that dispatch.

3. **SIMD kernels** — direct throughput (GB/s) and speedup for the byte
   de-interleave (scalar / SSE2 / AVX2) and half→float (scalar / F16C)
   primitives.

## SIMD tiers

The kernels are runtime-dispatched from CPUID (`exr_simd_info()` prints the
selected path). Tiers measured:

| tier   | de-interleave | half↔float |
|--------|---------------|------------|
| scalar | ✅            | ✅         |
| SSE2   | ✅            | —          |
| AVX2   | ✅            | F16C       |

**PSHUFB note:** the SSE4.1/AVX2 PSHUFB Huffman-emit kernel (ported from
fpnge) for the DEFLATE/ZIP *encoder* is not yet wired in — the ZIP encoder
currently uses the scalar fpng Huffman path. When added it will appear as a
fourth tier on the encode side.
