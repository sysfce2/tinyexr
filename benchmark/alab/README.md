# ALab texture_pack decode benchmark

`decode_bench.py` exercises TinyEXR v3's decoder over Animal Logic's **ALab**
`texture_pack` corpus and reports decode **speed** and decode **failures**.

The corpus (`/mnt/disk1/data/usd/alab/texture_pack/fragment/...`) is **6831 EXR
files, ~74 GB**, all 4096×4096 **tiled, ZIP-compressed** surfacing textures
(ao / ior / metallic / roughness / surfaceColor / ntu).

## What it does

The runner is a stdlib-only Python orchestrator that drives the two existing v3
C harnesses (no decode logic is reimplemented):

- **Speed** — pipes the file list to `test/v3/bench_decode`, which slurps each
  file into RAM and times only `exr_load_from_memory` (in-memory, decode-only,
  **single-threaded**). Reports per-codec and total Mpix/s and input MB/s.
- **Failures** — runs `test/v3/parse_harness` per file (thread pool) and
  classifies each load as PASS / XFAIL / XPASS / FAIL, naming every failing file
  with its load message and compression. DWAA/DWAB are treated as expected
  failures (v3 intentionally does not implement the lossy DWA codecs).

The sampled set is therefore decoded twice — once for timing, once for
correctness — which is cheap at the default sample size.

## Usage

```sh
# 200-file random sample of the default corpus (seeded, reproducible):
benchmark/alab/decode_bench.py

# Build the harnesses first (make lib + compile bench_decode & parse_harness):
benchmark/alab/decode_bench.py --build

# Quick iteration with live progress:
benchmark/alab/decode_bench.py --sample 50 --progress

# Whole corpus (slow: ~74 GB / 6831 files):
benchmark/alab/decode_bench.py --all

# Point at any other tree:
benchmark/alab/decode_bench.py /path/to/exr/tree
```

Key flags: `--sample N` (default 200), `--all`, `--seed S` (default 0),
`--jobs N` (failure-pass workers), `--report PATH`
(default `benchmark/alab/last-report.md`), `--bench-decode` / `--harness` to
override binary locations.

Exit status is non-zero if any unexpected `FAIL` (or `XPASS`) is found.

## Output

Prints a per-codec throughput table and a PASS/XFAIL/FAIL summary, and writes a
markdown report to `benchmark/alab/last-report.md` (corpus stats, throughput
table, and the named failure list). See that file for the latest run.

## Notes

- Throughput is **single-threaded** decode (from `bench_decode`); the
  `--jobs` workers only parallelize the per-file correctness pass.
- A full `--all` sweep decodes ~74 GB twice and takes a long time; use the
  default sample for routine checks.
