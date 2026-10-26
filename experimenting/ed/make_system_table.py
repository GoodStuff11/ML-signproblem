#!/usr/bin/env python3
"""Build the system-summary table (CSV + LaTeX) for the ED datasets.

For every dataset folder in data_h5_fixed, reports the lattice, electron numbers,
the spin sector, the degenerate ground-state momentum sectors, the momentum
sector the optimization actually runs in, the corresponding Hilbert space
dimensions, and P, the number of independent angles in one layer of the
momentum-conserving two-body ansatz (`n_params`).

Sector selection replicates `find_best_energy_sector` in ed_functions.jl:
sectors within atol=1e-6 of the lowest energy are tallied across the U scan and
the highest tally wins (for folders split into one file per sector, the file is
chosen first, then the sector within it). Ties are broken arbitrarily in the
Julia code, so the canonical (lexicographically smallest) member of the
degenerate set is reported here; all tied sectors are symmetry-equivalent and,
except for the 4x4 N=(4,4) case, share the same dimension.

The spin sector is Sz = (N_up - N_dn)/2: the ED and the optimization both work at
fixed (N_up, N_dn) and do not resolve total S (`su2_symmetry=false` is the
default and no run_*.jl script overrides it).

P = (3N^3 - 6N^2 + (g+2)N)/4 with N = Lx*Ly sites and g = 2^(number of even sides)
(HANDOFF §2); it equals the number of gates enumerate_ferm_excitations(2, Lvec;
conserve_mom=true, conserve_sz=true, include_diagonal=false) returns (114 for 3x2,
300 for 4x2, 432 for 3x3, 1092 for 4x3, 2712 for 4x4), independent of the filling.

The LaTeX table lists only the systems that appear in a paper figure, read from the figures'
ledgers data-summary/fig2.csv, fig3a.csv, fig3b.csv and fig4.csv (so it follows the figures
automatically); the CSV keeps every computed system, with `in_figures` marking those.

The LaTeX table uses the paper's notation (N sites, n_up/n_dn electrons, total
momentum K) and is a table* so it spans both columns of the two-column paper.

The table caption and footnotes are the CAPTION and FOOTNOTES text blocks near the top of this
file; paste replacement text there (line breaks inside a block are ignored).

Outputs: system_table.csv, system_table.tex  (alongside this script)
"""
import h5py, numpy as np, os, glob, csv, re

# =============================================================================
# CAPTION AND FOOTNOTES -- paste replacement text straight into these blocks.
# Write plain LaTeX; line breaks and indentation inside a block are ignored (all
# whitespace is collapsed to single spaces), so the text can be one long line or wrapped.
# =============================================================================

CAPTION = r"""
Table describing the Hubbard ground states examined. $N=L_x\times L_y$ is the number of sites, and $n_\uparrow$ and $n_\downarrow$ are the number of spin up and down electrons respectively, with an overall magnetization $S_z=(n_\uparrow-n_\downarrow)/2$. The optimization is performed at fixed $(n_\uparrow, n_\downarrow)$, with unresolved total spin $S$, and fixed momentum sector $\mathbf{K}=2\pi(n_x/L_x,n_y/L_y)$ for integers $(n_x, n_y)$. The momentum sectors that contain a state degenerate to the ground state for all interaction strengths $U$ are listed, along with the dimension of the subspace $\text{dim}\mathcal{H}_\mathbf{K}$ the dimension of the full Hilbert space $\text{dim}\mathcal{H}_{n_\uparrow n_\downarrow}$ and the number of independent parameters in one layer of the ansatz $P$, where $P=\tfrac14\big(3N^3-6N^2+(g+2)N\big)$ and $g=2^{\#\{d:\,L_d\ \mathrm{even}\}}$
"""

# One footnote per system, keyed by the data folder name ("N=(nu, nd)_LxxLy"). A system in the
# table gets a footnote mark only if it has an entry here; marks are numbered in table order.
# If a system in the table has an automatically detected note (ground-state manifold change,
# sectors of different dimension; see the CSV's `notes` column) but no entry here, the script
# warns and uses the automatic text.
FOOTNOTES = {
    "N=(6, 6)_4x4": r"""
The degenerate ground-state changes momentum sectors from $\{(0,0),\,(0,2),\,(2,0)\}$ to $\{(0,0),\,(2,2)\}$ at $U/t\approx4.936$. The $(0,0)$ sector has a level crossing within the momentum sector.
""",
}


def flat(s):
    """Collapse all whitespace (including newlines) in a pasted text block to single spaces."""
    return " ".join(s.split())
from fractions import Fraction
from math import comb

ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
HERE = os.path.dirname(os.path.abspath(__file__))
ATOL = 1e-6     # matches find_best_energy_sector
DEGTOL = 1e-8   # sector-degeneracy tolerance
SKIP = {"N=(4, 4)_3x3_separate"}   # duplicate of N=(4, 4)_3x3

FIGURE_LEDGERS = ["fig2.csv", "fig3a.csv", "fig3b.csv", "fig4.csv"]   # in data-summary/

def system_key(s):
    """(Lx, Ly, n_up, n_dn) from a folder name 'N=(nu, nd)_LxxLy' or a figure label '$Lx\\times Ly,\\;(nu,nd)$'."""
    m = re.search(r"N=\((\d+), (\d+)\)_(\d+)x(\d+)", s)
    if m:
        nu, nd, lx, ly = map(int, m.groups()); return (lx, ly, nu, nd)
    m = re.search(r"(\d+)\\times ?(\d+)[^(]*\((\d+),(\d+)\)", s)
    if m:
        lx, ly, nu, nd = map(int, m.groups()); return (lx, ly, nu, nd)
    raise ValueError(f"cannot parse system from {s!r}")

def figure_systems():
    used = set()
    for name in FIGURE_LEDGERS:
        with open(os.path.join(HERE, "data-summary", name)) as fh:
            used |= {system_key(r["system_size"]) for r in csv.DictReader(fh)}
    return used

def n_params(Lx, Ly):
    """Independent angles in one layer of the momentum-conserving two-body ansatz (HANDOFF §2)."""
    N = Lx * Ly
    g = 2 ** sum(1 for L in (Lx, Ly) if L % 2 == 0)
    P, r = divmod(3 * N**3 - 6 * N**2 + (g + 2) * N, 4)
    assert r == 0
    return P

def fmtq(qs):     return ";".join(f"({a},{b})" for a, b in qs)
def fmtq_tex(qs): return ",\\,".join(f"({a},{b})" for a, b in qs)
def num_tex(n):
    # comma thousands separators, as in the paper's text (32,670); {,} keeps math mode from
    # adding space after the comma
    return f"{n:,}".replace(",", "{,}")

rows = []
for sysdir in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, sysdir)
    fs = sorted(glob.glob(os.path.join(d, "HubbardED_XDiag_*.h5"))) if os.path.isdir(d) else []
    if not fs or sysdir in SKIP:
        continue

    uvec = None; sec = {}; fileE = {}; Lvec = nu = nd = None
    for fp in fs:
        with h5py.File(fp, 'r') as f:
            uvec = np.array(f['data/uvec']); q = np.array(f['metadata/qvecs'])
            Lvec = tuple(int(x) for x in np.array(f['metadata/Lvec']))
            nu = int(np.array(f['metadata/nu'])); nd = int(np.array(f['metadata/nd']))
            per = []
            for k in sorted(f['data/energies'].keys(), key=int):
                i = int(k)
                qt = tuple(int(x) for x in (q[i] if q.shape[0] > i else q[0]))
                E0 = np.array(f['data/energies'][k])[:, 0]
                sec[qt] = dict(E0=E0, dim=int(f['data/evecs'][k].shape[2]),
                               file=os.path.basename(fp))
                per.append(E0)
            fileE[os.path.basename(fp)] = np.min(np.stack(per, axis=1), axis=1)

    qs = sorted(sec); nU = len(uvec)
    E = np.stack([sec[q]['E0'] for q in qs], axis=1)
    mins = E.min(axis=1)

    # degenerate ground-state manifold(s)
    gsets = [frozenset(np.where(E[i] <= mins[i] + DEGTOL)[0]) for i in range(nU)]
    manifolds = []; s = 0
    for i in range(1, nU + 1):
        if i == nU or gsets[i] != gsets[s]:
            manifolds.append((float(uvec[s]), float(uvec[i - 1]),
                              sorted(qs[j] for j in gsets[s]))); s = i

    def tally(curves, labels):
        if len(curves) == 1:
            return [labels[0]]
        cnt = {}
        for t in range(nU):
            e = np.array([c[t] for c in curves]); m = e.min()
            for j in np.argsort(e):
                if abs(e[j] - m) <= ATOL: cnt[labels[j]] = cnt.get(labels[j], 0) + 1
                else: break
        mx = max(cnt.values())
        return [l for l in labels if cnt.get(l, 0) == mx]

    fl = sorted(fileE); best_files = tally([fileE[f] for f in fl], fl)
    if len(best_files) > 1:
        tied = [q for q in qs if sec[q]['file'] in best_files]
    else:
        inf = [q for q in qs if sec[q]['file'] == best_files[0]]
        tied = inf if len(inf) == 1 else tally([sec[q]['E0'] for q in inf], inf)
    used = sorted(tied)[0]

    Ns = Lvec[0] * Lvec[1]
    sz = Fraction(nu - nd, 2)
    dims_tied = sorted({sec[q]['dim'] for q in tied})
    note = ""; note_tex = ""
    if len(manifolds) > 1:
        note = "ground-state manifold changes with U -- " + "; ".join(
            f"U in [{a},{b}]: {fmtq(m)}" for a, b, m in manifolds)
        note_tex = ("The ground-state manifold changes across the level crossing: "
                    + "; ".join(f"${fmtq_tex(m)}$ for $U/t\\in[{a},{b}]$"
                                for a, b, m in manifolds) + ".")
    if len(dims_tied) > 1:
        bysize = {}
        for q in tied: bysize.setdefault(sec[q]['dim'], []).append(q)
        note += (" | " if note else "") + \
                "degenerate sectors differ in dimension: " + \
                "; ".join(f"{fmtq(sorted(v))}: {k}" for k, v in sorted(bysize.items()))
        note_tex += (" " if note_tex else "") + \
                    "The degenerate sectors do not all have the same dimension: " + \
                    "; ".join(f"${num_tex(k)}$ for ${fmtq_tex(sorted(v))}$"
                              for k, v in sorted(bysize.items())) + "."

    rows.append(dict(
        system=sysdir, lattice=f"{Lvec[0]}x{Lvec[1]}", Lx=Lvec[0], Ly=Lvec[1], n_sites=Ns,
        n_up=nu, n_dn=nd, n_electrons=nu + nd, filling=round((nu + nd) / Ns, 4),
        Sz=str(sz), n_momentum_sectors=len(qs),
        n_degenerate_gs_sectors=len(manifolds[0][2]),
        degenerate_gs_momentum_sectors=fmtq(manifolds[0][2]),
        momentum_sector_used=f"({used[0]},{used[1]})",
        hilbert_dim_sector_used=sec[used]['dim'],
        hilbert_dim_full_nu_nd=comb(Ns, nu) * comb(Ns, nd),
        n_params=n_params(Lvec[0], Lvec[1]),
        notes=note, _note_tex=note_tex,
        _tex_deg=fmtq_tex(manifolds[0][2]), _tex_used=f"({used[0]},{used[1]})",
        _sortkey=(Ns, Lvec[0], Lvec[1], nu, nd)))

rows.sort(key=lambda r: r.pop('_sortkey'))

in_figs = figure_systems()
for r in rows:
    r['in_figures'] = system_key(r['system']) in in_figs
missing = in_figs - {system_key(r['system']) for r in rows}
assert not missing, f"systems in figures but without ED data: {sorted(missing)}"

# ---------------- CSV ----------------
order = ["system", "lattice", "Lx", "Ly", "n_sites", "n_up", "n_dn", "n_electrons",
         "filling", "Sz", "n_momentum_sectors", "n_degenerate_gs_sectors",
         "degenerate_gs_momentum_sectors", "momentum_sector_used",
         "hilbert_dim_sector_used", "hilbert_dim_full_nu_nd", "n_params", "in_figures", "notes"]
csv_path = os.path.join(HERE, "system_table.csv")
with open(csv_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=order, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)

# ---------------- LaTeX ----------------
tex_rows = [r for r in rows if r['in_figures']]
unknown = set(FOOTNOTES) - {r['system'] for r in tex_rows}
if unknown:
    print(f"warning: FOOTNOTES entries for systems not in the table, ignored: {sorted(unknown)}")
marks, footnote_text = {}, {}
for r in tex_rows:
    if r['system'] in FOOTNOTES:
        text = flat(FOOTNOTES[r['system']])
    elif r['notes']:
        print(f"warning: {r['system']} has an automatic note but no FOOTNOTES entry; using: {r['_note_tex']}")
        text = r['_note_tex']
    else:
        continue
    marks[r['system']] = len(marks) + 1
    footnote_text[r['system']] = text

L = []
L.append(r"% Requires \usepackage{booktabs}.")
L.append(r"\begin{table*}[t]")
L.append(r"  \centering")
L.append(r"  \setlength{\tabcolsep}{9pt}")
L.append("  \\caption{" + flat(CAPTION) + "}")
L.append(r"  \label{tab:systems}")
L.append(r"  \begin{tabular}{lcccccrrr}")
L.append(r"    \toprule")
L.append(r"    Lattice & $N$ & $n_\uparrow$ & $n_\downarrow$ & $S_z$ & Degenerate $\mathbf{K}$ sectors"
         r" & $\dim\mathcal{H}_\mathbf{K}$ & $\dim\mathcal{H}_{n_\uparrow n_\downarrow}$ & $P$ \\")
L.append(r"    \midrule")
for r in tex_rows:
    sz = r['Sz']
    sz_tex = "0" if sz == "0" else (r"\tfrac{1}{2}" if sz == "1/2" else sz)
    mark = f"\\textsuperscript{{{marks[r['system']]}}}" if r['system'] in marks else ""
    L.append(f"    ${r['Lx']}\\times{r['Ly']}${mark} & {r['n_sites']} & {r['n_up']} & {r['n_dn']} "
             f"& ${sz_tex}$ & ${r['_tex_deg']}$ "
             f"& ${num_tex(r['hilbert_dim_sector_used'])}$ & ${num_tex(r['hilbert_dim_full_nu_nd'])}$ "
             f"& ${num_tex(r['n_params'])}$ \\\\")
L.append(r"    \bottomrule")
L.append(r"  \end{tabular}")
for r in tex_rows:
    if r['system'] in marks:
        L.append(f"  \\\\[2pt] \\footnotesize\\textsuperscript{{{marks[r['system']]}}} {footnote_text[r['system']]}")
L.append(r"\end{table*}")

tex_path = os.path.join(HERE, "system_table.tex")
open(tex_path, 'w').write("\n".join(L) + "\n")
print(f"wrote {csv_path} ({len(rows)} systems)\nwrote {tex_path} ({len(tex_rows)} systems in figures)")
