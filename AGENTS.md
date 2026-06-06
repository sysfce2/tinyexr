# AGENTS.md — TinyEXR v3 (pure-C11 rewrite)

Guidance for AI agents working on the pure-C11 EXR loader/writer under
`include/exr.h` + `src/*.c` (private decls in `src/exr_internal.h`). The legacy
v1 `tinyexr.h` at the repo root is a dependency of older tests — leave it
untouched.

## Scope / non-goals

- **DWA (DWAA / DWAB) is intentionally NOT supported — do not implement it.**
  We do not plan to support the lossy DCT DWA codecs. Leave
  `EXR_COMPRESSION_DWAA` / `EXR_COMPRESSION_DWAB` returning
  `EXR_ERROR_UNSUPPORTED` in the codec dispatch; do not add a DCT/zigzag/AC-DC
  decoder. Do not spend effort here.
- HTJ2K / JPH (`src/exr_jph.c`, `EXR_COMPRESSION_HTJ2K*`) is owned separately —
  don't refactor it without coordination.

## Build / test / validate

- `make lib`        — build `build/libtinyexr3.a` (`-std=c11 -Wall -Wextra -Werror`).
- `make c11-gate`   — strict pure-C11 syntax gate over all `src/*.c` (must stay green).
- `make test-c`     — unit tests under ASan+UBSan (note: runs with `detect_leaks=0`).
- `make fuzz-corpus`— replay `test/unit/regression/*` under ASan+UBSan+**LSan**
  (this is what catches error-path leaks; keep it green).
- `make fuzz-corpus-asan` — same corpus replay with `ASAN_OPTIONS=detect_leaks=0`;
  use only in ptrace/sandboxed local sessions where LSan aborts with
  "LeakSanitizer does not work under ptrace". It is a crash/ASan/UBSan fallback,
  not a replacement for the LSan gate.
- `make fuzz`       — clang/libFuzzer coverage-guided target (`build/fuzz_v3`).
- `make bench`      — codec + SIMD-kernel throughput.

Validation method: cross-check against the legacy v1 loader
(`LoadEXRImageFromFile` / `LoadDeepEXR`) and OpenEXR CLI tools
(`exrheader`, `exrmaketiled`, `exrmultipart`). `oiiotool` is broken here
(missing libboost). For lossy codecs the bar is "OpenEXR reads my output and my
own decode matches"; for lossless it is byte-identical.

## Secret-scanning audit (run before/after committing)

Audit new commits for accidentally-committed credentials with **both**
gitleaks and trufflehog. Tools default to `~/go/bin/` (trufflehog lives there;
gitleaks may also be on `PATH`). Both must exit 0 (no findings) before pushing.

```sh
# Scan only the new commits: RANGE = <last-known-clean>..HEAD (e.g. a tag or the
# upstream commit you branched from). Use --all / drop --log-opts for a full sweep.
RANGE=origin/release..HEAD

# gitleaks: history scan, redact matches, fail (non-zero) on any leak.
gitleaks detect --source . --log-opts="$RANGE" --no-banner --redact

# trufflehog: scan the same range; --fail makes it exit non-zero on a finding.
~/go/bin/trufflehog git "file://$(pwd)" --since-commit "${RANGE%%..*}" \
    --results=verified,unknown --no-update --fail
```

Both should report "no leaks found" / `verified_secrets: 0` and exit 0. If either
flags something, do not push — rewrite history to drop the secret and rotate it.

`.gitleaks.toml` (auto-loaded from the repo root) allowlists the vendored
`deps/zstd/` amalgamation, whose xxHash key-mixing intrinsics (`key_lo`/`key_hi`)
trip the generic-api-key heuristic — those are upstream constants, not secrets.
Keep first-party code out of the allowlist.

## Conventions

- Every new file gets the BSD-3-Clause header. Ported code keeps upstream
  attribution (fpng = public domain, fpnge = Apache-2.0; see `NOTICE`).
- All hostile-input arithmetic goes through `exr_mul_ovf` / `exr_add_ovf`.
  Error paths must leave outputs owning nothing (`exr_part_free` on failure).
- SIMD kernels use `__attribute__((target(...)))` + a runtime CPUID vtable so
  everything compiles at baseline; scalar fallback is always present and is the
  source of truth (SIMD must be bit-identical).
