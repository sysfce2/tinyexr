# TinyEXR v3 benchmark

Standalone micro/throughput benchmark for the pure-C11 EXR codec.

```sh
make bench                 # build + run on asakusa.exr
./build/bench foo.exr bar.exr   # run on your own files
```

It reports three things:

1. **Codec throughput** — for each codec the writer supports
   (NONE/RLE/ZIPS/ZIP/PIZ/HTJ2K32/HTJ2K256), the encode and decode rate in
   megapixels/second and the compressed size. Decode covers all codecs in the
   file; the auto-generated set only exercises the writable ones (pass
   PXR24/B44/DWA files on the command line to time those decoders too).

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

**PSHUFB Huffman-emit:** the `DEFLATE encode on EXR bytes` section compares
three encoders on real predictor+split EXR bytes: the default fpng LZ77 encoder,
and the fpnge-derived literal encoder with its PSHUFB per-byte Huffman-table
lookup (scalar vs SSE4.1). The PSHUFB lookup is the SSE4.1 (`pshufb`) tier. Note
that fpnge's literal-only, PNG-tuned table tends to *expand* EXR data, so it is
an opt-in/benchmark encoder — the default ZIP codec keeps the fpng LZ77 path.
