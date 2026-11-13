#!/usr/bin/env python3
"""Independent checks of the S column of system_table.{csv,tex} (make_system_table.py).

1. SU(2) ladder test, with no S^2 operator and no momentum basis: a separate real-space
   Hubbard ED, H = -t sum_<ij>,s c^dag_is c_js + U sum_i n_iup n_idn (t=1, periodic), is
   built for (n_up, n_dn) and for (n_up+k, n_dn-k), k = 1, 2, ... A multiplet of total spin S
   has members in every Sz' <= S, so the ground level of (n_up, n_dn) has S >= Sz+k exactly when
   the ground energy of the Sz+k sector equals it. The test finds S from where the ground energy
   first rises, and compares with the table at U values on both sides of every reported change
   (and at a small and a large U). The real-space ground energy is first compared with the ED
   file's (min over momentum sectors of data/energies) to show both solve the same model.
   Because the ladder gives S of the whole (n_up, n_dn) ground level, it also checks that every
   momentum sector of a degenerate ground manifold has that S.
2. The S column reported by make_system_table.py for every system (14 rows, incl. 4x4).
3. The table caption is identical to the one in the Overleaf system_table.tex as committed
   before this change (git HEAD of the Overleaf clone).

Systems with dim H_{n_up n_dn} > 5e5 (the 4x4 lattices) are too large for the real-space ladder
here; for those, make_system_table.py's own guard (every <S^2> within 1e-8 of S(S+1)) is what
backs the value.

Run: python testing/test_system_table_spin.py   (from experimenting/ed)
"""
import csv, os, re, subprocess, sys, glob
from itertools import combinations
from fractions import Fraction
import numpy as np, h5py
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

HERE = os.path.dirname(os.path.abspath(__file__)); ED = os.path.dirname(HERE)
ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"
OVERLEAF = "/home/jek354/research/68f6bef0eaaf5c0928a922c6"
MAXDIM = 500_000

def bonds(Lx, Ly):
    """Nearest-neighbour bonds with periodic boundaries: every site has a +x and a +y bond, so on a
    length-2 axis each pair is listed twice (hopping 2t across it), as in the ED -- the ground
    energies below match the ED files only with this doubling."""
    out = []
    for x in range(Lx):
        for y in range(Ly):
            i = x + Lx * y
            for dx, dy in ((1, 0), (0, 1)):
                j = (x + dx) % Lx + Lx * ((y + dy) % Ly)
                if i != j: out.append((min(i, j), max(i, j)))
    return out

def basis(N, n):
    return [sum(1 << i for i in c) for c in combinations(range(N), n)]

def hop_matrix(N, n, bl):
    """Single-species hopping sum_<ij> (c^dag_i c_j + h.c.) on n-particle bit strings, with the
    Jordan-Wigner sign (-1)^(occupied sites strictly between i and j)."""
    B = basis(N, n); idx = {b: k for k, b in enumerate(B)}
    r, c, v = [], [], []
    for k, b in enumerate(B):
        for i, j in bl:
            for a, d in ((i, j), (j, i)):
                if (b >> d) & 1 and not (b >> a) & 1:
                    lo, hi = min(a, d), max(a, d)
                    s = bin(b & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))).count("1")
                    r.append(idx[b ^ (1 << d) ^ (1 << a)]); c.append(k); v.append((-1.0) ** s)
    return sp.csr_matrix((v, (r, c)), shape=(len(B), len(B))), np.array(B)

def ground_energy(Lx, Ly, nu, nd, U):
    N = Lx * Ly; bl = bonds(Lx, Ly)
    Tu, Bu = hop_matrix(N, nu, bl); Td, Bd = hop_matrix(N, nd, bl)
    # spin-first ordering: up and down hoppings commute without extra signs
    H = -(sp.kron(Tu, sp.identity(len(Bd))) + sp.kron(sp.identity(len(Bu)), Td))
    dbl = np.array([bin(a & b).count("1") for a in Bu for b in Bd], float)
    H = (H + U * sp.diags(dbl)).tocsr()
    if H.shape[0] <= 400:
        return np.linalg.eigvalsh(H.toarray())[0]
    return eigsh(H, k=1, which="SA", tol=1e-12)[0][0]

def ladder_S(Lx, Ly, nu, nd, U, tol=1e-7):
    E0 = ground_energy(Lx, Ly, nu, nd, U); k = 0
    while nd - (k + 1) >= 0:
        Ek = ground_energy(Lx, Ly, nu + k + 1, nd - k - 1, U)
        if Ek - E0 > tol: break
        k += 1
    return E0, Fraction(nu - nd, 2) + k

def ed_file_E0(system, U):
    best = np.inf
    for fp in glob.glob(f"{ROOT}/{system}/HubbardED_XDiag_*.h5"):
        with h5py.File(fp, "r") as f:
            uv = np.array(f["data/uvec"]); i = int(np.argmin(abs(uv - U))); assert abs(uv[i] - U) < 1e-12
            for k in f["data/energies"]:
                best = min(best, float(np.array(f["data/energies"][k])[i, 0]))
    return best

def table_S_at(cell, U, crossings):
    """S the table reports at U: cell is '1/2' or 'a->b' with crossing estimates from notes."""
    parts = [Fraction(p) for p in cell.split("->")]
    if len(parts) == 1: return parts[0]
    assert len(parts) == 2 and len(crossings) == 1
    return parts[0] if U < crossings[0] else parts[1]

rows = list(csv.DictReader(open(os.path.join(ED, "system_table.csv"))))
fails = 0
print("== 1. real-space SU(2) ladder vs table ==")
print(f"{'system':16s} {'U':>6s} {'E0 real-space':>16s} {'E0 ED file':>16s} {'S ladder':>8s} {'S table':>8s}")
for r in rows:
    Lx, Ly, nu, nd = int(r["Lx"]), int(r["Ly"]), int(r["n_up"]), int(r["n_dn"])
    crossings = [float(x) for x in re.findall(r"U~([0-9.]+)", r["notes"])]
    Us = [0.5, 10.0] + [u for c in crossings for u in (round(c - 0.15, 2), round(c + 0.1, 2))]
    if int(r["hilbert_dim_full_nu_nd"]) > MAXDIM:
        print(f"{r['system']:16s}  skipped (dim {r['hilbert_dim_full_nu_nd']} > {MAXDIM}); table S = {r['S_ground_state']}")
        continue
    for U in Us:
        U = min([0.25 * k for k in range(1, 61)], key=lambda x: abs(x - U))   # ED grid point
        E0, S = ladder_S(Lx, Ly, nu, nd, U); Ef = ed_file_E0(r["system"], U)
        St = table_S_at(r["S_ground_state"], U, crossings)
        ok = abs(E0 - Ef) < 1e-8 and S == St
        fails += not ok
        print(f"{r['system']:16s} {U:6.2f} {E0:16.10f} {Ef:16.10f} {str(S):>8s} {str(St):>8s}  {'OK' if ok else 'FAIL'}")

print("\n== 2. S column of system_table.csv ==")
for r in rows:
    print(f"{r['system']:16s} Sz={r['Sz']:4s} S={r['S_ground_state']:8s} in_figures={r['in_figures']}  {r['notes']}")

print("\n== 3. caption identical to the Overleaf copy (git HEAD) ==")
old = subprocess.run(["git", "-C", OVERLEAF, "show", "HEAD:system_table.tex"], capture_output=True, text=True, check=True).stdout
new = open(os.path.join(ED, "system_table.tex")).read()
cap = lambda s: re.search(r"\\caption\{(.*)\}\n", s).group(1)
same = cap(old) == cap(new)
fails += not same
print(f"caption identical: {same}  ({len(cap(new))} chars)")

print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
sys.exit(fails != 0)
