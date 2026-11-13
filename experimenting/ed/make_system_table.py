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

S is the total spin of the target state, i.e. of the ED ground state of the momentum sector
the optimization runs in, read off from <S^2> = S(S+1) at every U in the scan
(`ground_state_spin`). S^2 = Sz^2 + Sz + S^- S^+ with S^+ = sum_k c^dag_{k up} c_{k dn}; the
form is basis independent, so it is evaluated directly on the stored momentum-space Slater
determinant eigenvectors (`data/evecs`, bit k of `metadata/basis_labels` = momentum orbital
k, spin-first Jordan-Wigner order: all up orbitals before all down orbitals, as in
`create_S2!`/`compute_swap_sign` in ed_functions.jl). If the sector's ground level is
degenerate (the two stored eigenvalues agree within DEGTOL) S^2 is diagonalized in that
2-dimensional space. Every <S^2> eigenvalue must be within S2TOL of S(S+1) for a
half-integer S, or the script stops. If S changes along the U scan, the crossing U is
estimated by linear interpolation of the two stored levels, labelled by their S, and the
system gets an automatic note.

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
Table describing the Hubbard ground states examined. $N=L_x\times L_y$ is the number of sites, and $n_\uparrow$ and $n_\downarrow$ are the number of spin up and down electrons respectively, with an overall magnetization $S_z=(n_\uparrow-n_\downarrow)/2$. The optimization is performed at fixed $(n_\uparrow, n_\downarrow)$, with unresolved total spin $S$, and fixed momentum sector $\mathbf{K}=2\pi(n_x/L_x,n_y/L_y)$ for integers $(n_x, n_y)$. The momentum sectors that contain a state degenerate to the ground state for all interaction strengths $U$ are listed, along with the dimension of the subspace $\text{dim}\mathcal{H}_\mathbf{K}$ (which will elsewhere be denoted as $\text{dim}\mathcal{H}$), the dimension of the full Hilbert space $\text{dim}\mathcal{H}_{n_\uparrow n_\downarrow}$ and the number of independent parameters in one layer of the ansatz $P$, where $P=\tfrac14\big(3N^3-6N^2+(g+2)N\big)$ and $g=2^{\#\{d:\,L_d\ \mathrm{even}\}}$.
"""

# One footnote per system, keyed by the data folder name ("N=(nu, nd)_LxxLy"). A system in the
# table gets a footnote mark only if it has an entry here; marks are numbered in table order.
# If a system in the table has an automatically detected note (ground-state manifold change,
# sectors of different dimension; see the CSV's `notes` column) but no entry here, the script
# warns and uses the automatic text.
FOOTNOTES = {
    "N=(4, 3)_3x3": r"""
The total spin of the ground state in the $(0,1)$ sector changes from $S=\tfrac{3}{2}$ to $S=\tfrac{1}{2}$ at $U/t\approx5.900$.
""",
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
S2TOL = 1e-8    # |<S^2> - S(S+1)| allowed for the target state
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

def popcount(x):
    x = x.astype(np.uint64); c = np.zeros(x.shape, np.int64)
    while np.any(x):
        c += (x & np.uint64(1)).astype(np.int64); x = x >> np.uint64(1)
    return c

def splus_structure(up, dn, N, nu):
    """Nonzero elements of S^+ = sum_k c^dag_{k up} c_{k dn} acting on the Slater determinants
    (up, dn bit strings) of one sector, in spin-first Jordan-Wigner order (up orbital k is mode
    k, down orbital k is mode N+k). Returns (source row, target slot, sign, number of slots)."""
    src, keys, sgn = [], [], []
    for k in range(N):
        b = np.uint64(1) << np.uint64(k); below = b - np.uint64(1)
        m = np.nonzero(((dn & b) != 0) & ((up & b) == 0))[0]
        u, d = up[m], dn[m]
        # c_{k dn} passes all nu up modes and the down modes below k; c^dag_{k up} the up modes below k
        par = nu + popcount(d & below) + popcount(u & below)
        src.append(m); sgn.append(1.0 - 2.0 * (par % 2))
        keys.append(((u | b) << np.uint64(32)) | (d & ~b))
    src, keys, sgn = np.concatenate(src), np.concatenate(keys), np.concatenate(sgn)
    _, tgt = np.unique(keys, return_inverse=True)
    return src, tgt.ravel(), sgn, (int(tgt.max()) + 1 if len(tgt) else 0)

def s2_eigvals(vecs, st, nu, nd):
    """Eigenvalues of S^2 restricted to span(vecs) (rows of vecs are orthonormal)."""
    src, tgt, sgn, M = st; sz = (nu - nd) / 2
    W = []
    for v in vecs:
        a = v[src] * sgn
        W.append(np.bincount(tgt, a.real, M) + 1j * np.bincount(tgt, a.imag, M))
    W = np.array(W)
    return np.linalg.eigvalsh((sz * sz + sz) * np.eye(len(vecs)) + (W.conj() @ W.T).real)

def spin_of(s2, where):
    S = Fraction(int(round(np.sqrt(1 + 4 * s2) - 1)), 2)      # S(S+1) = s2, S half-integer
    assert abs(s2 - float(S * (S + 1))) < S2TOL, f"{where}: <S^2>={s2} is not S(S+1) for half-integer S"
    return S

def ground_state_spin(path, key, uvec):
    """Total spin of the ground state of one momentum sector at every U.
    Returns (per U: sorted tuple of the distinct S in the (possibly degenerate) ground level,
             crossings: list of (U_left, U_right, U_cross estimate or None, S_before, S_after))."""
    with h5py.File(path, 'r') as f:
        N = int(np.prod(np.array(f['metadata/Lvec']))); nu = int(np.array(f['metadata/nu']))
        nd = int(np.array(f['metadata/nd'])); lab = np.array(f['metadata/basis_labels'][key])
        st = splus_structure(lab[:, 0].astype(np.uint64), lab[:, 1].astype(np.uint64), N, nu)
        En = np.array(f['data/energies'][key]); ev = f['data/evecs'][key]
        per_u, levels = [], []          # levels[i] = [(E, S) for each stored eigenvector]
        for i in range(len(uvec)):
            V = np.array(ev[i])         # (n_stored, dim)
            where = f"{path}[{key}] U={uvec[i]}"
            if abs(En[i, 1] - En[i, 0]) <= DEGTOL:   # degenerate pair: diagonalize S^2 in it
                Ss = [spin_of(x, where) for x in s2_eigvals(V, st, nu, nd)]
            else:
                Ss = [spin_of(s2_eigvals(V[j:j + 1], st, nu, nd)[0], where) for j in range(V.shape[0])]
            levels.append(list(zip(En[i], Ss)))
            per_u.append(tuple(sorted({S for E, S in levels[-1] if E <= En[i, 0] + DEGTOL})))
    crossings = []
    for i in range(len(uvec) - 1):
        if per_u[i] != per_u[i + 1] and len(per_u[i]) == 1 and len(per_u[i + 1]) == 1:
            Sa, Sb = per_u[i][0], per_u[i + 1][0]
            Ea = {S: E for E, S in levels[i]}; Eb = {S: E for E, S in levels[i + 1]}
            uc = None
            if Sa in Ea and Sb in Ea and Sa in Eb and Sb in Eb:   # both levels stored on both sides
                d0, d1 = Ea[Sa] - Ea[Sb], Eb[Sa] - Eb[Sb]
                uc = float(uvec[i] + (uvec[i + 1] - uvec[i]) * d0 / (d0 - d1))
            crossings.append((float(uvec[i]), float(uvec[i + 1]), uc, Sa, Sb))
    return per_u, crossings

def fmtS(S):     return str(S)
def fmtS_tex(S): return str(S.numerator) if S.denominator == 1 else f"\\tfrac{{{S.numerator}}}{{{S.denominator}}}"
def spin_text(per_u, tex=False):
    """Ground-state S along the U scan, distinct values in order of first appearance
    (a degenerate ground level of mixed S is written S1=S2)."""
    f = fmtS_tex if tex else fmtS
    out = []
    for gs in per_u:
        x = "=".join(f(S) for S in gs)
        if x not in out: out.append(x)
    return ("\\to " if tex else "->").join(out)

def crossing_text(c, tex):
    a, b, uc, Sa, Sb = c
    if tex:
        where = f"U/t\\approx{uc:.3f}" if uc is not None else f"U/t\\in({a},{b})"
        return f"$S={fmtS_tex(Sa)}\\to{fmtS_tex(Sb)}$ at ${where}$"
    where = f"U~{uc:.3f}" if uc is not None else f"U in ({a},{b})"
    return f"S {Sa}->{Sb} at {where}"

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
                               file=os.path.basename(fp), path=fp, key=k)
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

    spin_u, spin_cross = ground_state_spin(sec[used]['path'], sec[used]['key'], uvec)
    if len(set(spin_u)) > 1:
        note += (" | " if note else "") + "ground-state total spin changes with U in the sector used: " + \
                "; ".join(crossing_text(c, False) for c in spin_cross)
        note_tex += (" " if note_tex else "") + \
                    f"The total spin of the ground state in the $({used[0]},{used[1]})$ sector changes with $U$: " + \
                    "; ".join(crossing_text(c, True) for c in spin_cross) + "."

    rows.append(dict(
        system=sysdir, lattice=f"{Lvec[0]}x{Lvec[1]}", Lx=Lvec[0], Ly=Lvec[1], n_sites=Ns,
        n_up=nu, n_dn=nd, n_electrons=nu + nd, filling=round((nu + nd) / Ns, 4),
        Sz=str(sz), S_ground_state=spin_text(spin_u), _tex_S=spin_text(spin_u, tex=True),
        _spin_u=spin_u, n_momentum_sectors=len(qs),
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
         "filling", "Sz", "S_ground_state", "n_momentum_sectors", "n_degenerate_gs_sectors",
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
L.append("% Generated by ML-signproblem/experimenting/ed/make_system_table.py -- edit the CAPTION and FOOTNOTES "
         "blocks there and regenerate; do not edit by hand.")
L.append(r"% Requires \usepackage{booktabs}.")
L.append(r"\begin{table*}[t]")
L.append(r"  \centering")
L.append(r"  \setlength{\tabcolsep}{9pt}")
L.append("  \\caption{" + flat(CAPTION) + "}")
L.append(r"  \label{tab:systems}")
L.append(r"  \begin{tabular}{lccccccrrr}")
L.append(r"    \toprule")
L.append(r"    Lattice & $N$ & $n_\uparrow$ & $n_\downarrow$ & $S_z$ & $S$ & Degenerate $\mathbf{K}$ sectors"
         r" & $\dim\mathcal{H}_\mathbf{K}$ & $\dim\mathcal{H}_{n_\uparrow n_\downarrow}$ & $P$ \\")
L.append(r"    \midrule")
for r in tex_rows:
    sz = r['Sz']
    sz_tex = "0" if sz == "0" else (r"\tfrac{1}{2}" if sz == "1/2" else sz)
    mark = f"\\textsuperscript{{{marks[r['system']]}}}" if r['system'] in marks else ""
    L.append(f"    ${r['Lx']}\\times{r['Ly']}${mark} & {r['n_sites']} & {r['n_up']} & {r['n_dn']} "
             f"& ${sz_tex}$ & ${r['_tex_S']}$ & ${r['_tex_deg']}$ "
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
