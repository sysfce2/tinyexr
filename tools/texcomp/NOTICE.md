# TinyEXR Texcomp Notices

This directory contains TinyEXR BSD-3-Clause code plus CLI-only use of
`examples/common/stb_image.h` for PNG loading.

The BC7 API and QuickBC7 option names are designed for a pure-C11 port of the
QuickBC7-enabled etcpak fork:

- Paper: QuickBC7: Fast BC7 Texture Compression Heuristics, Hyeon-ki Lee and
  Jae-Ho Nah, Computer & Graphics 2026.
- Reference source: https://github.com/gusrlLee/etcpak, inspected at commit
  `b88c8f4`.

The current C11 encoder emits valid BC7 modes 0 through 7 and selects the
lowest reconstructed-error candidate per block using a scalar endpoint search.
Quick mode ports the QuickBC7 luma/alpha mode predecision and LUT partition
prediction for two-subset modes 1 and 7. The upstream least-squares endpoint
optimizer and SIMD kernels remain future optimization work.

BC5 support is a scalar BC4-pair encoder for unsigned RG data. BC6H support
currently emits valid unsigned-float and signed-float BC6H using one-region mode
11 with 4-bit selectors. It evaluates per-channel bounds and luma-axis endpoint
candidates per block; full multi-mode BC6H rate-distortion search remains
future work.

The ASTC LDR encoder covers all fourteen 2D footprints with decimated weight
grids, one to four partitions (per-partition weight fitting on a shared grid),
dual-plane alpha, and CEM 0/4/6/8/10/12 including mixed per-partition formats.
Endpoints are refined with an integer least-squares solve and all encoding
paths compete on quantization-aware reconstruction error. Conformance is
verified against a reference decoder in test/astc_ref_decode.h and against
Arm's astcenc (`make texcomp-astc-arm-smoke`); quality is tracked by
`make texcomp-astc-psnr`. The partition-pattern hash and the ISE/unquant data
tables follow the Khronos ASTC specification (cross-checked against
https://github.com/ARM-software/astc-encoder). The projection and
reconstruction kernels of the encoder search have SSE2 and NEON variants
that are bit-exact with the scalar forms (enforced by cross-backend parity
tests); AVX2-widened kernels remain future optimization work.
