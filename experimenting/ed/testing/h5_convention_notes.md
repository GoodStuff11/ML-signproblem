# HDF5 vs TamFermion Basis Convention — Investigation Notes

## Goal

Make `HubbardMomentumBasis(q_target=2)` return a Hamiltonian H such that the raw HDF5 eigenvector `state = evecs_dataset[:,1,1]` satisfies `state' * H * state ≈ E0` without any manual transformation.

---

## What Was Confirmed Working

`HubbardMomentumBasis(q_target=2)` already produces a Hamiltonian H with:
- Correct eigenvalues matching the HDF5 stored energies to floating-point precision
- Correct sector dimension (50 states for 3×2, n_up=3, n_dn=2, sector "2")

The problem is purely about **basis ordering and sign convention**, not about the physics.

---

## Key Discoveries

### 1. `kvecs` Uses C-order (Not F-order)

The HDF5 file stores `metadata/kvecs` where orbital index `o` (0-based) maps to:

| Orbital | kx | ky |
|---------|----|----|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |
| 4 | 2 | 0 |
| 5 | 2 | 1 |

The orbital ordering is **C-order** (kx is the slow index): `orbital o → kx = o ÷ Ly, ky = o % Ly` with `Ly = 2`.

TamFermion uses **F-order** (Fortran-order): bit `k` → `(kx, ky) = unravel_f(k, (3,2))` = `kx = k % 3, ky = k ÷ 3`. So:
- HDF5 orbital `o` maps to F-order bit `f_bit = kx + ky * Lx` (NOT directly to `o`)

The **correct conversion** from HDF5 orbital to TamFermion bit is:
```julia
kx = kvecs[1, o+1]
ky = kvecs[2, o+1]
bit = ravel_f((kx, ky), dims)   # = kx + ky * Lx
```

This was verified numerically: HDF5 state int=587 (h5_idx=1) matches the TamFermion basis when using this conversion.

### 2. There Is a Sign Mismatch Between Conventions

TamFermion creates basis states by applying creation operators in **ascending F-bit order**:
```
|state⟩_tam = c†_0 c†_1 c†_3 |0⟩   (ascending F-bit: 0,1,3)
```

HDF5 creates basis states by applying creation operators in **ascending C-orbital order**:
```
|state⟩_h5 = c†_0 c†_3 c†_1 |0⟩   (C-orbitals 0,1,2 → F-bits 0,3,1)
```

For state h5_idx=1 (up=[0,1,2]): the F-bit sequence in C-orbital order is `[0, 3, 1]`, which requires **1 swap** to sort to ascending `[0, 1, 3]`. Sign = -1.

The state-level sign is: `|h5_i⟩ = sgn_i |tam_{perm[i]}⟩`, where sgn_i = `permutation_parity(sortperm(up_c_orbs)) * permutation_parity(sortperm(dn_c_orbs))`.

### 3. The HDF5 Sector Label "2" Is Not Total Momentum Index 2

**Critical open question:** when computing the total momentum of the HDF5 basis states using kvecs, states within `slater_labels/2` do NOT all have the same total momentum. For example:
- h5_idx=1 (up=[0,1,2], dn=[0,1]) → total k = (kx=1, ky=0) → C-order index = 2
- h5_idx=3 (up=[0,1,3], dn=[0,3]) → total k = (kx=2, ky=1) → C-order index = 5

This means **the HDF5 `slater_labels/2` does not represent a single fixed-k sector.** The sector label "2" may be a particle-number sector or some other grouping, and the 50-state eigenbasis is stored together regardless of individual state momenta.

However, because `HubbardMomentumBasis(q_target=2)` produces the correct eigenvalues, the sectors ARE physically equivalent — the relationship between `q_target` and HDF5 sector label is correct at the eigenvalue level.

### 4. `DtMb = UInt32`, Combined Integers Are UInt16

`combineSpinInts(ints_up, ints_dn, N)` calls `uint_for_bits(2N)`. For N=6, `2N=12`, so `T = UInt16`. The combined spin integers in `basis_dict["ints"]` are `UInt16`. Lookups must use `UInt16` keys, not `UInt64`.

---

## What Was Implemented

### New Helper Functions (TamFermion.jl)

**`permutation_parity(perm)`** — Computes ±1 parity of a permutation using cycle decomposition.

**`f_bit_to_c_orbital(k, Lvec)`** — Converts 0-based F-order bit index to 0-based C-order orbital index: `(kx, ky) = unravel_f(k, dims)` → `c_orbital = kx * Ly + ky`.

**`h5_ordering_and_signs(basis_ints, N, Lvec)`** — For each TamFermion basis state, computes:
1. Its C-order key: `(sort(up_c_orbs), sort(dn_c_orbs))`
2. Its sign correction: `permutation_parity(sortperm(up_c_orbs)) * permutation_parity(sortperm(dn_c_orbs))`

Then returns `(perm_h5_to_tam, sgns)` sorted by C-order key.

**`HubbardMomentumBasis` modification** — When `h5_convention=true` and no `basis_sector` override: applies `H_h5 = D * H_tam[perm, perm] * D` where `D = Diagonal(sgns)`.

**`reorder_to_h5::Bool=true` parameter** — Bypass switch added to `HubbardMomentumBasis` for debugging (set `false` to get raw TamFermion internal ordering).

---

## Outstanding Problems

### Sign Correction Appears Wrong (or h5 Sector Is Not Fixed-k)

After applying `H_h5 = D * H_tam[perm, perm] * D`:
- `state' * H_h5 * state = +2.75` (should be -10.754)
- Fidelity `|⟨state|gs⟩|² = 2.4e-6` (should be ~1)

The analytical sign is worse than no sign correction at all. There are two likely causes:

1. **The permutation is wrong**: `slater_labels/2` may NOT index the same states as TamFermion's sector `q_target=2` at all, causing the entire perm to be invalid. The int=1113 lookup failure confirms this — state h5_idx=3 with total momentum (kx=2, ky=1) does not exist in the `q_target=2` subspace.

2. **The HDF5 sector "2" is not a fixed-k sector**: If the HDF5 stores all (N_up=3, N_dn=2) states together (300 total, but only 50 per k-sector shown due to file structure), the slater_labels/2 might list a PARTICLE-NUMBER sector not a MOMENTUM sector.

---

## Next Steps

1. **Inspect HDF5 structure more carefully**: determine what `slater_labels/2` actually represents — is "2" a momentum index, a particle number, or something else?

2. **Verify correspondence**: Check whether `q_target=2` in TamFermion and "sector 2" in HDF5 actually contain the SAME set of physical states (not just the same eigenvalues by coincidence).

3. **Check alternative mapping**: If the HDF5 sector label has a different meaning, the correct `q_target` for TamFermion may be different, and the slater_labels need to be interpreted differently for the basis alignment.

4. **Compare eigenvectors directly**: Once the correct perm is established (step 1-2), compare TamFermion eigenvectors with HDF5 eigenvectors component-by-component to find the true sign vector numerically.
