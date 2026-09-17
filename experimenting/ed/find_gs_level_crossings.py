#!/usr/bin/env python3
"""Identify ground-state level crossings vs U for every system in data_h5_fixed.

Two crossing channels are detected; energies alone only reveal the first.

1. INTER-SECTOR: the global ground state moves to a different momentum sector.
   Detected from the lowest eigenvalue per sector. Sectors related by point-group
   symmetry are exactly degenerate, so the ground manifold at each U is the set of
   sectors within TOL of the minimum, and a crossing requires a genuine two-way
   exchange (a sector in the manifold at U_i lies strictly above the minimum at
   U_{i+1}, and vice versa).

2. INTRA-SECTOR: the two lowest levels *inside* the ground-state sector swap.
   The ground-state momentum never changes, so energies alone are blind to this.
   Detected from the stored eigenvectors: |<v0(U_i)|v0(U_{i+1})>| collapses to ~0
   while the span of the two stored eigenvectors is preserved, i.e. the ground
   state and the first excited state of that sector exchange character.
   If the sector's two lowest levels are exactly degenerate, the individual
   eigenvectors are gauge-arbitrary and the 2D subspace overlap is used instead.

4x4 lattices are excluded (per request).
"""
import h5py, numpy as np, os, glob, csv, sys

ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
OUT  = "/home/jek354/research/ML-signproblem/experimenting/ed/ground_state_level_crossings.csv"
TOL     = 1e-8    # inter-sector degeneracy tolerance (result stable 1e-11..1e-8)
DEGTOL  = 1e-10   # intra-sector degeneracy tolerance
OVTHR   = 0.5     # overlap below this = level exchange
SKIP_GEOM = {"4x4"}

fmtq = lambda qs: ";".join(f"({a}|{b})" for a, b in sorted(qs))
rows = []

for sysdir in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, sysdir)
    files = sorted(glob.glob(os.path.join(d, "HubbardED_XDiag_*.h5"))) if os.path.isdir(d) else []
    if not files:
        continue

    uvec = None; E = {}; OV0 = {}; OVSUB = {}; Lvec = nu = nd = None
    for fp in files:
        with h5py.File(fp, 'r') as f:
            Lvec = tuple(int(x) for x in np.array(f['metadata/Lvec']))
            if f"{Lvec[0]}x{Lvec[1]}" in SKIP_GEOM:
                break
            u = np.array(f['data/uvec'])
            if uvec is None:
                uvec = u
            elif len(u) != len(uvec) or not np.allclose(u, uvec):
                print(f"WARNING: U-grid mismatch, skipping {fp}", file=sys.stderr); continue
            q = np.array(f['metadata/qvecs'])
            nu = int(np.array(f['metadata/nu'])); nd = int(np.array(f['metadata/nd']))
            for k in f['data/energies'].keys():
                i = int(k)
                qt = tuple(int(x) for x in (q[i] if q.shape[0] > i else q[0]))
                E[qt] = np.array(f['data/energies'][k])          # (nU, 2)
                dset = f['data/evecs'][k]                        # (nU, 2, dim) complex
                nU = dset.shape[0]
                ov0 = np.zeros(nU - 1); ovs = np.zeros(nU - 1)
                prev = np.array(dset[0])
                for t in range(nU - 1):
                    cur = np.array(dset[t + 1])
                    ov0[t] = abs(np.vdot(prev[0], cur[0]))
                    M = prev.conj() @ cur.T                      # <prev_a|cur_b>
                    ovs[t] = np.sum(np.abs(M) ** 2) / 2.0
                    prev = cur
                OV0[qt] = ov0; OVSUB[qt] = ovs
    if not E:
        continue

    qs = sorted(E)
    E0 = np.stack([E[q][:, 0] for q in qs], axis=1)
    E1 = np.stack([E[q][:, 1] for q in qs], axis=1)
    nU = E0.shape[0]; step = float(uvec[1] - uvec[0])
    mins = E0.min(axis=1)
    gset = [set(np.where(E0[i] <= mins[i] + TOL)[0]) for i in range(nU)]

    cross = []
    for i in range(nU - 1):
        A, B = gset[i], gset[i + 1]

        # channel 1: ground state changes momentum sector
        if A != B:
            lost   = [a for a in A if E0[i + 1, a] > mins[i + 1] + TOL]
            gained = [b for b in B if E0[i, b] > mins[i] + TOL]
            if lost and gained:
                sep = max(max(E0[i + 1, a] - mins[i + 1] for a in lost),
                          max(E0[i, b] - mins[i] for b in gained))
                cross.append(dict(kind="inter_sector", i=i,
                                  before=[qs[a] for a in A], after=[qs[b] for b in B],
                                  sectors=[qs[a] for a in sorted(set(lost) | set(gained))],
                                  ov="", sub="", gapa="", gapb="", ustar="",
                                  sep=f"{sep:.6e}"))

        # channel 2: levels swap inside a sector that is the ground state at both ends
        for j in sorted(A & B):
            q = qs[j]
            ga = E1[i, j] - E0[i, j]; gb = E1[i + 1, j] - E0[i + 1, j]
            degenerate = ga < DEGTOL and gb < DEGTOL
            metric = OVSUB[q][i] if degenerate else OV0[q][i]
            if metric < OVTHR:
                ustar = uvec[i] + step * ga / (ga + gb) if (ga + gb) > 0 else ""
                cross.append(dict(kind="intra_sector", i=i,
                                  before=[qs[a] for a in A], after=[qs[b] for b in B],
                                  sectors=[q], ov=f"{OV0[q][i]:.3e}", sub=f"{OVSUB[q][i]:.6f}",
                                  gapa=f"{ga:.6e}", gapb=f"{gb:.6e}",
                                  ustar=f"{ustar:.4f}" if ustar != "" else "", sep=""))

    # merge crossings that occur in the same U step (symmetry-equivalent sectors)
    merged = {}
    for c in cross:
        key = (c['i'], c['kind'])
        if key in merged:
            merged[key]['sectors'] = sorted(set(merged[key]['sectors']) | set(c['sectors']))
        else:
            merged[key] = c

    base = dict(system=sysdir, geometry=f"{Lvec[0]}x{Lvec[1]}", n_sites=Lvec[0] * Lvec[1],
                nu=nu, nd=nd, n_momentum_sectors=len(qs),
                U_min=uvec[0], U_max=uvec[-1], U_step=step,
                gs_momentum_sectors=fmtq([qs[j] for j in sorted(gset[0])]))
    if merged:
        for key in sorted(merged):
            c = merged[key]; i = c['i']
            rows.append({**base, "has_level_crossing": "yes", "crossing_type": c['kind'],
                         "U_interval": f"({uvec[i]} - {uvec[i+1]}]",
                         "U_lower": uvec[i], "U_upper": uvec[i + 1],
                         "U_crossing_estimate": c['ustar'],
                         "crossing_sectors": fmtq(c['sectors']),
                         "gs_momenta_before": fmtq(c['before']), "gs_momenta_after": fmtq(c['after']),
                         "overlap_v0": c['ov'], "subspace_overlap": c['sub'],
                         "gap_before": c['gapa'], "gap_after": c['gapb'],
                         "intersector_separation": c['sep']})
    else:
        rows.append({**base, "has_level_crossing": "no", "crossing_type": "",
                     "U_interval": "", "U_lower": "", "U_upper": "", "U_crossing_estimate": "",
                     "crossing_sectors": "", "gs_momenta_before": "", "gs_momenta_after": "",
                     "overlap_v0": "", "subspace_overlap": "", "gap_before": "", "gap_after": "",
                     "intersector_separation": ""})

order = ["system", "geometry", "n_sites", "nu", "nd", "n_momentum_sectors",
         "U_min", "U_max", "U_step", "gs_momentum_sectors", "has_level_crossing",
         "crossing_type", "U_interval", "U_lower", "U_upper", "U_crossing_estimate",
         "crossing_sectors", "gs_momenta_before", "gs_momenta_after",
         "overlap_v0", "subspace_overlap", "gap_before", "gap_after",
         "intersector_separation"]
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=order); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} rows)")
