"""
test_paper_figure_palette.py

Checks that the paper's Figs. 2, 3 and 4 use one colour scheme: the plasma map as fig_lib.jl's
cmap3 samples it (from 0.95, gold, down to 0, indigo). It reads the colours the PDFs actually
draw (the `r g b rg` / `r g b RG` operators of their content streams), so it tests the files
the paper includes, not the code that should have made them.

Usage:
  python3 testing/test_paper_figure_palette.py <figures_dir>
where <figures_dir> holds figure2.pdf, figure3.pdf and figure4.pdf.

Checks (all must pass, exit code 1 otherwise):
  1. every non-gray colour drawn in each figure is a plasma colour, within 2/255 per channel
     of the map at some point of [0, 0.95];
  2. Fig. 2's colours include the 10 per-system colours cmap3(11)[1:10];
  3. Fig. 3 draws exactly those 10 per-system colours, and no other non-gray colour;
  4. Fig. 4 draws exactly cmap3(7), the 7 colours of m = 1, 2, 3, 4, 6, 8 and infinity
     (in particular no red, which the old "circuit" curve used).
"""
import re, sys, zlib
import numpy as np
from matplotlib import colormaps

PLASMA = colormaps["plasma"]
TOL = 2.5 / 255


def cmap3(n):
    return [np.array(PLASMA(z)[:3]) for z in np.linspace(0.95, 0, n)]


def pdf_colours(path):
    raw = open(path, "rb").read()
    cols = set()
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            s = zlib.decompress(m.group(1))
        except zlib.error:
            continue
        for c in re.finditer(rb"([\d.]+) ([\d.]+) ([\d.]+) (?:rg|RG)\b", s):
            cols.add(tuple(round(float(x), 4) for x in c.groups()))
    return {c for c in cols if not (abs(c[0] - c[1]) < 1e-3 and abs(c[1] - c[2]) < 1e-3)}  # drop grays


def on_plasma(c, grid=np.array([PLASMA(z)[:3] for z in np.linspace(0, 0.95, 4001)])):
    return np.abs(grid - np.array(c)).max(axis=1).min() <= TOL


def same_set(found, expected):
    unmatched_found = [c for c in found if not any(np.abs(np.array(c) - e).max() <= TOL for e in expected)]
    unmatched_exp = [e for e in expected if not any(np.abs(np.array(c) - e).max() <= TOL for c in found)]
    return unmatched_found, unmatched_exp


def main(d):
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    cols = {f: pdf_colours(f"{d}/figure{f}.pdf") for f in (2, 3, 4)}
    for f, cs in cols.items():
        off = [c for c in cs if not on_plasma(c)]
        check(f"Fig. {f}: all {len(cs)} non-gray colours lie on plasma[0, 0.95]", not off, f"off-map: {off}" if off else "")
    systems = cmap3(11)[:10]
    _, missing = same_set(cols[2], systems)
    check("Fig. 2: draws the 10 per-system colours cmap3(11)[1:10]", not missing, f"missing: {missing}" if missing else "")
    extra, missing = same_set(cols[3], systems)
    check("Fig. 3: draws exactly Fig. 2's 10 per-system colours", not extra and not missing,
          f"extra: {extra} missing: {missing}" if (extra or missing) else "")
    extra, missing = same_set(cols[4], cmap3(7))
    check("Fig. 4: draws exactly cmap3(7) (m = 1, 2, 3, 4, 6, 8, inf), nothing else", not extra and not missing,
          f"extra: {extra} missing: {missing}" if (extra or missing) else "")
    for f, cs in cols.items():
        print(f"  Fig. {f} colours (0-255): {sorted(tuple(int(round(255 * x)) for x in c) for c in cs)}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
