#!/usr/bin/env python3
"""Build the system-summary table (CSV + LaTeX) for the ED datasets.

For every dataset folder in data_h5_fixed, reports the lattice, electron numbers,
the spin sector, the degenerate ground-state momentum sectors, the momentum
sector the optimization actually runs in, and the corresponding Hilbert space
dimensions.

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

Outputs: system_table.csv, system_table.tex  (alongside this script)
"""
import h5py, numpy as np, os, glob, csv
from fractions import Fraction
from math import comb

ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
HERE = os.path.dirname(os.path.abspath(__file__))
ATOL = 1e-6     # matches find_best_energy_sector
DEGTOL = 1e-8   # sector-degeneracy tolerance
SKIP = {"N=(4, 4)_3x3_separate"}   # duplicate of N=(4, 4)_3x3

def fmtq(qs):     return ";".join(f"({a},{b})" for a, b in qs)
def fmtq_tex(qs): return ",\\,".join(f"({a},{b})" for a, b in qs)
def num_tex(n):
    s = f"{n:,}".replace(",", "\\,")
    return s

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
        notes=note, _note_tex=note_tex,
        _tex_deg=fmtq_tex(manifolds[0][2]), _tex_used=f"({used[0]},{used[1]})",
        _sortkey=(Ns, Lvec[0], Lvec[1], nu, nd)))

rows.sort(key=lambda r: r.pop('_sortkey'))

# ---------------- CSV ----------------
order = ["system", "lattice", "Lx", "Ly", "n_sites", "n_up", "n_dn", "n_electrons",
         "filling", "Sz", "n_momentum_sectors", "n_degenerate_gs_sectors",
         "degenerate_gs_momentum_sectors", "momentum_sector_used",
         "hilbert_dim_sector_used", "hilbert_dim_full_nu_nd", "notes"]
csv_path = os.path.join(HERE, "system_table.csv")
with open(csv_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=order, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)

# ---------------- LaTeX ----------------
marks = {}
for r in rows:
    if r['notes']:
        marks[r['system']] = len(marks) + 1

L = []
L.append(r"% Requires \usepackage{booktabs} (and \usepackage{amsmath} for \text).")
L.append(r"\begin{table}[htbp]")
L.append(r"  \centering")
L.append(r"  \caption{Hubbard systems used in this work. $N_s=L_x\times L_y$ is the number of")
L.append(r"  lattice sites and $N_\uparrow$, $N_\downarrow$ the numbers of spin-up and spin-down")
L.append(r"  electrons; the optimization is carried out at fixed $(N_\uparrow,N_\downarrow)$, i.e.\ in")
L.append(r"  the spin sector $S_z=(N_\uparrow-N_\downarrow)/2$, without resolving total spin $S$.")
L.append(r"  Momentum sectors are labelled by integers $(n_x,n_y)$ with")
L.append(r"  $\mathbf{k}=2\pi(n_x/L_x,\,n_y/L_y)$; the listed sectors are those degenerate with")
L.append(r"  the ground state. The optimization uses a single representative of that")
L.append(r"  degenerate set, whose Hilbert space dimension is $\dim\mathcal{H}_\mathbf{k}$;")
L.append(r"  $\dim\mathcal{H}_{N_\uparrow N_\downarrow}=\binom{N_s}{N_\uparrow}\binom{N_s}{N_\downarrow}$")
L.append(r"  is the dimension of the full fixed-particle-number sector.}")
L.append(r"  \label{tab:systems}")
L.append(r"  \begin{tabular}{lcccccrr}")
L.append(r"    \toprule")
L.append(r"    Lattice & $N_s$ & $N_\uparrow$ & $N_\downarrow$ & $S_z$ & Degenerate $\mathbf{k}$ sectors"
         r" & $\dim\mathcal{H}_\mathbf{k}$ & $\dim\mathcal{H}_{N_\uparrow N_\downarrow}$ \\")
L.append(r"    \midrule")
for r in rows:
    sz = r['Sz']
    sz_tex = "0" if sz == "0" else (r"\tfrac{1}{2}" if sz == "1/2" else sz)
    mark = f"\\textsuperscript{{{marks[r['system']]}}}" if r['system'] in marks else ""
    L.append(f"    ${r['Lx']}\\times{r['Ly']}${mark} & {r['n_sites']} & {r['n_up']} & {r['n_dn']} "
             f"& ${sz_tex}$ & ${r['_tex_deg']}$ "
             f"& ${num_tex(r['hilbert_dim_sector_used'])}$ & ${num_tex(r['hilbert_dim_full_nu_nd'])}$ \\\\")
L.append(r"    \bottomrule")
L.append(r"  \end{tabular}")
for r in rows:
    if r['system'] in marks:
        L.append(f"  \\\\[2pt] \\footnotesize\\textsuperscript{{{marks[r['system']]}}} {r['_note_tex']}")
L.append(r"\end{table}")

tex_path = os.path.join(HERE, "system_table.tex")
open(tex_path, 'w').write("\n".join(L) + "\n")
print(f"wrote {csv_path}\nwrote {tex_path}\n({len(rows)} systems)")
