#!/usr/bin/env python3
"""
Generate the hand-styled perf SVGs in doc/ from inline data (stdlib only).

The charts in doc/ were originally hand-authored SVG; this reproduces that exact
minimal style (620x400, blue=in-tree / green=libdeflate / amber=OpenEXR, value
labels above bars, light gridlines) so the data-bearing ones can be regenerated.

    python3 doc/gen_perf_charts.py        # writes the *.svg next to this script

Colors:  in-tree #2563eb, libdeflate #16a34a, OpenEXR #f59e0b
"""

import os

W, H = 620, 400
X0, X1 = 60.0, 602.0          # plot x-range
Y0, YTOP = 328.0, 64.0        # y of value 0 and of y-max
CY = {"intree": "#2563eb", "libdeflate": "#16a34a", "openexr": "#f59e0b",
      "before": "#9ca3af", "zstd": "#7c3aed"}


def _fmt(v):
    return ("%.1f" % v).rstrip("0").rstrip(".") if v % 1 else "%d" % v


def chart(path, title, subtitle, cats, series, ymax, ytick):
    """series: list of (label, color, [value per cat]). cats: list of names."""
    span = Y0 - YTOP
    def y(v):
        return Y0 - (v / ymax) * span
    out = []
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
               'viewBox="0 0 %d %d" font-family="-apple-system,Segoe UI,Roboto,'
               'sans-serif">' % (W, H, W, H))
    out.append('<rect width="%d" height="%d" fill="#fff"/>' % (W, H))
    out.append('<text x="60" y="26" font-size="17" font-weight="700" '
               'fill="#111">%s</text>' % title)
    out.append('<text x="60" y="44" font-size="12" fill="#666">%s</text>' % subtitle)
    # gridlines + y labels
    t = 0
    while t <= ymax + 1e-9:
        gy = y(t)
        out.append('<line x1="60" y1="%.1f" x2="602" y2="%.1f" stroke="#eee"/>'
                   % (gy, gy))
        out.append('<text x="52" y="%.1f" font-size="11" fill="#888" '
                   'text-anchor="end">%s</text>' % (gy + 4, _fmt(t)))
        t += ytick
    out.append('<text x="16" y="196" font-size="12" fill="#666" '
               'transform="rotate(-90 16 196)" text-anchor="middle">'
               'megapixels / s</text>')
    # bars
    ncat = len(cats)
    nser = len(series)
    group_w = (X1 - X0) / ncat
    bw = min(42.6, group_w * 0.78 / nser)
    for ci, cat in enumerate(cats):
        gx = X0 + ci * group_w
        inner = nser * bw
        start = gx + (group_w - inner) / 2.0
        for si, (label, color, vals) in enumerate(series):
            v = vals[ci]
            bx = start + si * bw
            by = y(v)
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                       'fill="%s" rx="2"/>' % (bx, by, bw, Y0 - by, color))
            out.append('<text x="%.1f" y="%.1f" font-size="9" fill="#444" '
                       'text-anchor="middle">%s</text>'
                       % (bx + bw / 2.0, by - 4, _fmt(v)))
        out.append('<text x="%.1f" y="346" font-size="12" fill="#333" '
                   'text-anchor="middle">%s</text>'
                   % (gx + group_w / 2.0, cat))
    out.append('<line x1="60" y1="328.0" x2="602" y2="328.0" stroke="#bbb"/>')
    # legend
    lx = 60
    for label, color, _ in series:
        out.append('<rect x="%d" y="370" width="12" height="12" fill="%s" '
                   'rx="2"/>' % (lx, color))
        out.append('<text x="%d" y="380" font-size="12" fill="#333">%s</text>'
                   % (lx + 17, label))
        lx += 24 + int(7.0 * len(label))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # Decode throughput, in-tree vs libdeflate, openexr-images natural corpus.
    # Measured with test/v3/bench_decode (single thread, decode-only, Mpix/s).
    chart(
        os.path.join(here, "perf-libdeflate-corpus-decode.svg"),
        "Decode throughput - in-tree vs libdeflate (natural-image corpus)",
        "openexr-images; libdeflate is the DEFLATE=auto hosted default.",
        ["zip", "zips", "pxr24"],
        [("tinyexr (in-tree)", CY["intree"], [226, 171, 309]),
         ("tinyexr (libdeflate)", CY["libdeflate"], [391, 278, 479])],
        ymax=500, ytick=100,
    )

    # PIZ decode before/after the in-tree optimization (best-of-8 on the 36
    # openexr-images PIZ files; inline Huffman literal + tighter canonical scan).
    chart(
        os.path.join(here, "perf-piz-decode.svg"),
        "PIZ decode - in-tree optimization (+14%)",
        "openexr-images; inline Huffman literal store + tighter canonical scan.",
        ["piz decode"],
        [("before", CY["before"], [80.2]),
         ("after", CY["intree"], [91.6])],
        ymax=100, ytick=20,
    )

    # ZSTD vs ZIP decode on identical images (6 openexr-images ScanLines files
    # re-encoded to each; both decoded with the default hosted build).
    chart(
        os.path.join(here, "perf-zstd-decode.svg"),
        "ZSTD vs ZIP decode - identical images",
        "6 openexr-images ScanLines files re-encoded to each codec.",
        ["decode"],
        [("zstd (vendored)", CY["zstd"], [410]),
         ("zip (libdeflate)", CY["libdeflate"], [272])],
        ymax=450, ytick=90,
    )


if __name__ == "__main__":
    main()
