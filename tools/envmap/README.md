# envmap — environment-map projections, SH & spherical gaussians

Pure-C11 tool for image-based-lighting representations. Converts between
environment-map projections and fits compact lighting bases (spherical
harmonics, spherical gaussians). Links `tir + texcomp + texpipe + libtinyexr3`;
only the CLI does HDR EXR I/O.

## What it does

- **Projection conversion** (`convert`) — resample between **equirectangular**
  (Y-up), **cube** (KTX/D3D/GL face order `+X,-X,+Y,-Y,+Z,-Z`, matching
  `texpipe`'s cube convention), and **octahedral** (Y-up pole). Cube output
  writes 6 face EXRs that `texpipe --cube-face` consumes directly; octa output
  feeds `texpipe --octa`.
- **Spherical harmonics** (`sh`) — project an env onto real SH up to order 4
  ((L+1)² RGB coeffs), with an optional Hanning window to curb ringing. Order 2
  (9 coeffs) is the classic diffuse-irradiance ambient term.
- **Spherical gaussians** (`sg`) — fit K SG lobes (Fibonacci axes, shared
  sharpness ≈ 0.35·K, non-negative least-squares amplitudes) — a compact
  all-frequency-ish lighting representation.

`sh`/`sg` also write `<out>_recon.exr`, an equirect reconstruction from the
coefficients for eyeballing quality.

## Build

```sh
make envmap            # build/envmap/envmap
make envmap-c11-gate   # strict pure-C11 (no <stdio.h> outside the CLI)
make envmap-test       # projection round-trips, solid-angle, SH/SG
```

## CLI

```sh
# equirect HDR -> 6 cube face EXRs (feed texpipe --cube-face)
envmap convert -i env.exr --from equirect --to cube --size 256 -o cube.exr
# equirect -> octahedral (feed texpipe --octa)
envmap convert -i env.exr --to octa --size 512 -o octa.exr

# order-2 SH (9 coeffs) + reconstruction
envmap sh -i env.exr --order 2 -o env.sh
# 24-lobe spherical gaussians + reconstruction
envmap sg -i env.exr --lobes 24 -o env.sg
```

## Pipeline into texpipe (HDR mip + compress)

```sh
envmap convert -i env.exr --to octa --size 512 -o octa.exr
texpipe -i octa.exr -o octa.ktx2 --format bc6h --octa --container ktx2
```

## Library API (`include/envmap.h`, prefix `em_`)

`em_convert`, `em_dir_to_uv`/`em_uv_to_dir`, `em_texel_solid_angle`,
`em_sample`, `em_foreach_texel`; `em_sh_project`/`em_sh_eval`/`em_sh_window`;
`em_sg_fit`/`em_sg_eval`; plus `em_hammersley` and hemisphere/sphere sampling.

## Verification (from the tests)

Direction round-trips < 1e-6 (all 3 projections); per-texel solid angles sum to
4π within 0.03%; equirect→cube→equirect resample ≈ 87 dB PSNR; SH white-furnace
(constant env → DC only) and SH-4 smooth-field RMS ≈ 4e-4; SG constant-energy
preserved with bounded ripple.

## Notes / deferred

- SH/SG use the equirect solid-angle weights (projection-agnostic; the input is
  treated on the sphere). Convention: SH polar axis = +Y, self-consistent
  project/eval (not matched to an external SH library's axis).
- `--asg` (anisotropic SG) is reserved; the fit is isotropic today.
- Prefiltered specular IBL (GGX) and diffuse-cube irradiance were out of scope
  for this effort (see the plan catalog).

## License

Apache-2.0. Depends on `tir` (BSD-3), `texcomp`/`texpipe` (Apache-2.0),
TinyEXR core (BSD-3).
