# Exact Diagonalization and Optimization File Formats
Hierarchical file schemas for `.h5` Lanczos outputs and `.jld2` Trotter optimization checkpoints.

---
## 1. Lanczos ED HDF5 File Schema (`.h5`)
Files located in `/home/jek354/research/data/new_data/data_h5_fixed/` adhere to the structure:
- `/data/`:
  - `energies`: Array of ground-state eigenvalues per interaction value $U$.
  - `evecs`: Eigenvector coefficients matching basis sector representations.
  - `uvec`: Interaction strengths $U/t$ sampled during ED scan.
  - `runtime`: Wall-clock computation duration for Lanczos solver.
  - `ED_methods`: Diagonalization configuration and parameters.
- `/metadata/`:
  - `Lvec`: Lattice geometry vector $[L_x, L_y]$.
  - `basis_labels`: Determinant configurations or momentum/symmetry state labels.
  - `nd`: Number of spin-down electrons.
  - `nu`: Number of spin-up electrons.
  - `qvecs`: Momentum vectors corresponding to translation symmetry sectors.

Verified details for the `HubbardED_XDiag_momSlater_*.h5` files (used by `experimenting/ed/make_system_table.py`, checked by `experimenting/ed/testing/test_system_table_spin.py`):
- Sectors are keyed by string index `k` = "0", "1", ...; sector `k` has momentum `qvecs[k]` (integers $(n_x,n_y)$, $\mathbf{K}=2\pi(n_x/L_x,n_y/L_y)$). Files named `..._sectors_(k).h5` hold one sector each.
- `data/energies/<k>`: shape `(n_U, 2)`, the two lowest eigenvalues of sector `k` at each `uvec` entry, ascending.
- `data/evecs/<k>`: shape `(n_U, 2, dim)`, complex128; `evecs[i, j, :]` is the eigenvector of `energies[i, j]`.
- `metadata/basis_labels/<k>`: shape `(dim, 2)`, uint64; row `r` = (up bit string, down bit string) of basis state `r`, bit `m` set = momentum orbital `m` occupied (Slater determinant in momentum orbitals).
- Fermion order is spin-first Jordan–Wigner (all up orbitals, then all down orbitals), as in `create_S2!`/`compute_swap_sign` in `ed_functions.jl`: with this order the stored ground states give $\langle S^2\rangle = S(S+1)$ to $\le 10^{-11}$; the interleaved order does not.
- Hamiltonian: $H=-t\sum_{\langle ij\rangle\sigma}c^\dagger_{i\sigma}c_{j\sigma}+U\sum_i n_{i\uparrow}n_{i\downarrow}$, $t=1$, periodic; on a length-2 axis the $+\hat y$ and $-\hat y$ bonds join the same pair and both are counted (effective hopping $2t$). Real-space ED with this convention reproduces `energies` to $10^{-10}$.
- `metadata/eig_target` (string) and `metadata/eig_sigma` (float64), written by UCC_Hubbard's XDiag pipeline from 2026-11-16 on: `"lowest"`/NaN = each sector holds its m LOWEST eigenpairs; `"closest_to_sigma"`/σ = each sector holds the m eigenpairs with smallest $|E-\sigma|$ (pipeline flag `--sigma σ`; σ = 0 gives the smallest-magnitude, highly excited mid-spectrum states). Files without these fields hold the lowest eigenpairs. In both cases `energies/<k>[i, :]` is ascending. Read with `read_eig_target(data)` in `ed_functions.jl` (returns `("lowest", 0.0)` when absent). Filenames of σ files carry `_sigma_<σ>` after `_m_<m>`.

### Recorded momentum sector (`momentum_sector.toml`)
A data folder may contain `momentum_sector.toml`, recording the momentum sector its saved results were trained in when the ground state is exactly degenerate between sectors (symmetry images with identical spectra). Keys:
- `sector` (int): flat ED key $k$ of `data/energies/<k>`, $k=q_x L_y+q_y$.
- `q` ([int, int]): total momentum $(q_x,q_y)$, $\mathbf K=2\pi(q_x/L_x,q_y/L_y)$.
- `tied_sectors` ([int]): all sectors whose lowest level is the ground state at every U.
- `verified_by` (string): how the sector was established.
`load_h5_ED_data` uses `sector` as the default when no `sector=` is given and `target_state=:lowest`; `make_system_table.py` uses `q` for the `momentum_sector_used` column. Present (2026-11-17), each verified by `experimenting/ed/testing/check_paper_sectors.jl` (the saved one-layer `ref_slater` coefficients reproduce their saved loss only in this sector):
- `N=(3, 2)_3x2`: sector 4, q=(2,0) (tied 2,4)
- `N=(3, 2)_3x3`, `N=(4, 3)_3x3`, `N=(5, 4)_3x3`: sector 3, q=(1,0) (tied 1,2,3,6)
- `N=(5, 4)_4x3`: sector 2, q=(0,2) (tied 1,2)
Folders without the file use the tally + smallest-index rule of `find_best_energy_sector` (e.g. `N=(4, 4)_4x4`: tied 0,2,8 → 0; `N=(5, 4)_4x4`: tied 1,3,4,12 → 1; neither has saved optimization results).

### Excited-state folders
- `N=(4, 4)_4x3`: the ground-state data (m = 2 lowest per sector). The ground state is in sector `0`, q = (0,0), at every U. Sectors `1`,`2` (q = (0,1),(0,2)) have an exactly 2-fold degenerate lowest level, so they are unsuitable fidelity targets. Sectors `7`,`8` (q = (2,1),(2,2)) have a non-degenerate lowest state that is the "slightly excited" target (`load_ED_data(...; sector=7)`); its energy minus the ground energy and its gap to the next state are printed by `experimenting/ed/testing/test_excited_state_targets.jl`.
- `N=(4, 4)_4x3_excited_sigma0`: same lattice, filling, U grid (0.25:0.25:15) and schema, but `eig_target = "closest_to_sigma"`, `eig_sigma = 0`, m = 8 per sector (all 12 sectors), complex128 storage (`--force-complex`). File `HubbardED_XDiag_momSlater_4x3_nu_4_nd_4_t_1_m_8_sigma_0.h5`, plus `HubbardBasis_XDiag_4x3_nu_4_nd_4.h5` and `Raw/` per-U orbit-basis files. The spectrum has EXACT zero modes (|E| < 1e-8; e.g. 2 in sector 0 at U = 0.25, 4 at U = 4, from dense diagonalization), which are degenerate; `target_state=:min_abs_energy` skips them by default (`zero_mode_tol = 1e-8`) and targets the closest non-zero level. Produced by Slurm array 698368 + consolidation job 698398 (`jobs/consolidate_4x3_44_sigma0.sh`).

---
## 2. Trotter Optimization Checkpoint Schema (`.jld2`)
Optimization results saved per interaction point $u$:
- `dict`:
  - `coefficients::Vector{Float64}`: Optimized Trotter parameter vector $A$.
  - `u_idx::Int`: 1-based index into interaction vector `U_values`.
  - `norm1::Float64`: $L_1$-norm of parameter vector $\sum |A_i|$.
  - `norm2::Float64`: $L_2$-norm of parameter vector $\sqrt{\sum A_i^2}$.
  - `metrics::Dict`: Tracked history of loss and fidelity across iterations.
