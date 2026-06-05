# TinyEXR v3 JPH Remaining Tasks

Current state:
- `src/exr_jph.c` is a clean-room C11 HTJ2K/JPH front end under BSD-3-Clause.
- OpenEXR HT wrapper parsing is implemented.
- JPEG 2000 marker/profile validation is implemented for the intended OpenEXR subset: SOC/SIZ/CAP/COD/QCD/QCC/NLT/SOT/SOD/EOC handling, RPCL, one layer, 5 reversible 5/3 decompositions, 128x32 HT codeblocks, default precincts, and single tile part.
- Packet parsing validates precinct order, inclusion tag trees, missing-MSB tag trees, pass counts, pass lengths, codeblock byte bounds, cleanup `Scup`, and HT stream stuffing.
- Internal HT forward/reverse/MEL readers exist and are covered by tests.
- Codeblock callback boundary exists. Validated codeblock descriptors include component, resolution, band, subband coordinates, dimensions, pass lengths, missing MSBs, and byte ranges.
- Component coefficient planes are allocated with the codec allocator.
- All-empty JPH codestreams decode to zero blocks through the intended postprocess pipeline: inverse 5/3, inverse RCT if enabled, NLT type 3, and TinyEXR block layout write.
- Non-empty codeblocks still return `EXR_ERROR_UNSUPPORTED` after validation.
- `make c11-gate` passes.
- `make test-c` passes with `129 passed, 0 failed`.
- `src/exr_jph.c` has no `assert()`, `abort()`, or `exit()` calls.

Important files:
- `src/exr_jph.c`: JPH parser, packet parser, HT readers, transform helpers, coefficient-plane scaffolding.
- `src/exr_internal.h`: internal JPH helper declarations used by tests.
- `test/unit/test_exr_v3.c`: synthetic JPH profile tests, transform tests, packet/HT reader tests, allocator balance tests.
- `jph.md`: earlier OpenEXR JPH investigation and implementation plan.

Remaining implementation tasks:

1. Add HT block decoder tables.
   - Decide whether to clean-room generate VLC/UVLC tables from spec-derived logic or import/adapt OpenJPH table generation with BSD-2 attribution.
   - If any OpenJPH code/table data is copied or derived, keep its copyright and BSD-2 license notice.
   - Keep public TinyEXR licensing permissive; do not introduce GPL or incompatible code.
   - Prefer compact generated static tables or deterministic init without global mutable races.

2. Implement cleanup pass entropy decode.
   - Implement scalar C11 decoder for MagSgn, MEL, and VLC streams.
   - Decode HT quads into a bounded scratch representation.
   - Avoid OpenJPH assertions and unchecked pointer reads.
   - All reads must go through checked helpers and return `EXR_ERROR_CORRUPT` or `EXR_ERROR_UNSUPPORTED`.
   - Validate `missing_msbs`, `p = 30 - missing_msbs`, `Scup`, stream lengths, and block dimensions before decoding.

3. Implement coefficient reconstruction for cleanup pass.
   - Convert decoded quad metadata and MagSgn bits into signed int32 coefficients.
   - Write only inside the target codeblock rectangle in the correct subband plane.
   - Handle edge codeblocks smaller than 128x32.
   - Keep all arithmetic checked for int32 overflow where relevant.

4. Implement optional SPP/MRP refinement pass support.
   - Current pass parser accepts active pass counts 1..3 and validates lengths.
   - Add SPP when `active_passes >= 2`.
   - Add MRP when `active_passes >= 3`.
   - Respect OpenEXR/OpenJPH tolerances already mirrored in validation, such as zero refinement length downgrading to one pass where appropriate.

5. Map decoded subbands into component coefficient planes.
   - Current callback validates coordinates but does not write coefficients.
   - Add codeblock decode output placement for LL and high-frequency subbands across all 5 resolutions.
   - Confirm band coordinate conventions against the inverse 5/3 layout already used by `exr_jph_inverse_53_2d_i32`.

6. Complete postprocess/output for real non-empty blocks.
   - After all codeblocks decode, run inverse 5/3 per component.
   - Apply inverse RCT for `mc_trans == 1`.
   - Apply NLT type 3 where present.
   - Store HALF/UINT/FLOAT channels into TinyEXR canonical block layout.
   - For FLOAT, confirm whether OpenEXR JPH stores signed 32-bit integer-transformed samples or needs bit-preserving handling before writing.

7. Add real-file interoperability tests.
   - Generate or locate tiny OpenEXR HTJ2K/JPH fixtures from OpenEXR/OpenJPH.
   - Add tests for grayscale HALF, RGB HALF with RCT, UINT, and FLOAT if supported.
   - Compare decoded pixels against OpenEXR output for a few small deterministic images.
   - Keep fixtures small.

8. Add malformed/fuzz tests for entropy decode.
   - Bad `Scup`, truncated MagSgn/MEL/VLC, malformed stuffing, overlong UVLC, excessive missing MSBs, invalid refinement lengths, and edge codeblock sizes.
   - Ensure ASan/UBSan stays clean.
   - Keep hardened behavior: no OOB, no crash, no assert.

9. Decide encoder scope.
   - `exr_jph_compress()` currently returns `EXR_ERROR_UNSUPPORTED`.
   - If decode is the only near-term requirement, leave encoder unsupported.
   - If encoding is needed, implement only after decoder is verified.

10. Documentation and cleanup.
    - Update comments at the top of `src/exr_jph.c` once non-empty entropy decode is implemented.
    - Document supported subset and explicit unsupported features.
    - Keep implementation single-file unless tables make a split clearer.
    - Re-run `make c11-gate`, `make test-c`, and `rg -n "assert\\(|abort\\(|exit\\(" src/exr_jph.c`.

Resume prompt:

```
Continue implementing TinyEXR v3 JPH/HTJ2K decode in /mnt/nvme02/work/tinyexr.

Read tasks.md, jph.md, src/exr_jph.c, src/exr_internal.h, and test/unit/test_exr_v3.c first. The current JPH front end parses OpenEXR HT wrapper/JPEG2000 profile, validates RPCL packet metadata and HT codeblock segment structure, has HT forward/reverse/MEL readers, allocates coefficient planes, decodes complete all-empty JPH codestreams through inverse 5/3/RCT/NLT and block store, and returns EXR_ERROR_UNSUPPORTED for non-empty codeblocks.

Continue from the non-empty codeblock boundary. Implement the next smallest verifiable piece toward HT cleanup entropy decode in pure C11, keeping it portable, low-memory, and hardened. Do not use assert(), abort(), or exit(); return exr_result errors for all malformed input. Preserve permissive licensing: if referencing/copying OpenJPH tables or code from /mnt/nvme02/work/openexr/external/OpenJPH, keep BSD-2 copyright/license attribution and do not introduce GPL code.

Use apply_patch for edits. Keep unrelated dirty files untouched. After each slice run:
- make c11-gate
- make test-c
- rg -n "assert\\(|abort\\(|exit\\(" src/exr_jph.c

Current known passing state before this continuation: make c11-gate passes, make test-c reports 129 passed, 0 failed.
```
