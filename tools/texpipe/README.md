# texpipe — resize-aware texture compression

`texpipe` ties together the two standalone libraries in this repo — **`tir`**
(content-aware image resize) and **`texcomp`** (BC/ETC/ASTC block compression) —
to build content-aware mip chains and serialize them into multi-mip GPU texture
containers (DDS, KTX2). It is pure C11; only `texpipe_cli.c` does file I/O.

## Why

`tir` can resize with content awareness (premultiplied alpha, normal/height
modes) but has no mip-chain generator. `texcomp` has every block encoder but only
writes single-surface, single-mip files. Neither knows about the other. `texpipe`
is the missing layer: *resize → per-mip content passes → compress → container*.

## Status (Phases 0–3 — all four capabilities)

Implemented:

- **Mip-chain generation**, resample **from base** by default (one filtering
  error per level, no accumulated blur; `--mip-source previous` for the classic
  half-step chain).
- **Alpha-aware RGBA** resize+compress: `tir` premultiplied alpha feeding
  BC7/BC3/ASTC alpha-capable encoders (capability #4).
- **Alpha-coverage preservation** across LODs (Castaño): rescales each mip's
  alpha so its alpha-tested coverage matches the base level, so cutouts/foliage
  don't thin out with distance (`--content alpha`, capability #2).
- **Seam-free cubemap LOD** (capability #1): 6-face input (separate files or
  cross/strip layouts), per-mip AMD-CubeMapGen-style edge + corner fixup so
  adjacent-face borders are bit-identical at every level, written to cubemap
  DDS/KTX2 (`DDSCAPS2_CUBEMAP` / `faceCount = 6`). See the caveat below.
- **Normal / height LOD coherence** (capability #3): `--content normal` filters
  and renormalizes normals per mip (stays unit-length across LODs, verified to
  ~1e-7), `--content height` is mean-preserving. `--bake-roughness` captures the
  pre-renormalize |N| and maps it to a **Toksvig** roughness (Toksvig 2005),
  written as a companion EAC_R11 KTX2 mip chain so specular highlights don't
  alias into shimmer at distance.
- **Multi-mip DDS** (DX10) for the BC family and **multi-mip KTX2** (Vulkan
  formats, native cube/mip/array) for BC/ETC2/EAC/ASTC.
- Codecs: BC1/BC3/BC5/BC7/BC6H, ETC2 RGB/RGBA, EAC R11/RG11, ASTC LDR (any
  block), ASTC HDR 4x4.

### Cubemap seam caveat (honest limit)

The edge/corner fixup makes adjacent-face borders **bit-identical before
compression** (the unit test measures max border deviation ≈ 0.006 on a
direction field, and the fixup is idempotent). Block codecs (BC/ETC/ASTC)
compress each face independently, so a residual seam of up to **one quantization
step** can reappear after encoding even with identical float borders. Nothing in
a standard independent-block codec removes this fully; rely on hardware seamless
cube filtering at runtime, or prefer higher-bit-depth codecs (BC7/ASTC) on cube
borders. `--no-seam-fixup` disables the pass.

## Build

```sh
make texpipe           # build/libtexpipe.a + build/texpipe/texpipe CLI
make texpipe-c11-gate  # strict pure-C11 gate (no <stdio.h> outside the CLI)
make texpipe-test      # unit tests (per-mip PSNR, alpha coverage, containers)
```

## CLI

```sh
texpipe -i in.{exr,png} -o out.{ktx2,dds} --format bc7 [opts]
```

Key options: `--format`, `--content color|alpha|normal|height`,
`--container dds|ktx2`, `--filter`, `--edge clamp|wrap|reflect`, `--levels N`,
`--mip-source base|previous`, `--srgb`, `--alpha`, `--alpha-threshold`,
`--astc-block WxH`, `--threads`, `--part` (see `texpipe --help`).

HDR codecs (`bc6h`, `astc_hdr`) require an EXR input; LDR codecs take EXR or PNG.

Cubemaps — give all six faces (order `+x -x +y -y +z -z`) or split one packed
image:

```sh
# 6 separate square faces
texpipe --cube-face +x px.png --cube-face -x nx.png ... -o cube.ktx2 --format bc7
# one cross/strip image
texpipe -i strip.png --cube-layout strip_h -o cube.ktx2 --format astc --astc-block 6x6
```

Normal maps with a baked Toksvig roughness companion:

```sh
texpipe -i normal.png -o normal.ktx2 --format bc5 --content normal \
        --normal-enc unorm --bake-roughness --base-roughness 0.15
# also writes normal.ktx2.rough.ktx2 (EAC_R11 roughness mip chain)
texpipe -i height.png -o height.ktx2 --format bc7 --content height   # mean-preserving
```

## API

The staged C API (see `include/texpipe.h`) is:

```c
tp_build_mips()      base image  -> content-aware float mip chain
tp_compress_chain()  float chain -> compressed block payloads
tp_write_container() blocks      -> DDS / KTX2 bytes
tp_process()         one-shot of the three above
```

plus leaf helpers: `tp_alpha_coverage()` / `tp_alpha_scale_to_coverage()`
(coverage), `tp_cube_seam_fixup()` / `tp_cube_split()` (cubemap),
`tp_toksvig_roughness()` / `tp_build_roughness_chain()` (normal roughness).

## License

Apache-2.0 (matches `texcomp`). Depends on `tir` (BSD-3-Clause) and `texcomp`
(Apache-2.0); both permissive and compatible for combined works.
