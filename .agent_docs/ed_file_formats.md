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

---
## 2. Trotter Optimization Checkpoint Schema (`.jld2`)
Optimization results saved per interaction point $u$:
- `dict`:
  - `coefficients::Vector{Float64}`: Optimized Trotter parameter vector $A$.
  - `u_idx::Int`: 1-based index into interaction vector `U_values`.
  - `norm1::Float64`: $L_1$-norm of parameter vector $\sum |A_i|$.
  - `norm2::Float64`: $L_2$-norm of parameter vector $\sqrt{\sum A_i^2}$.
  - `metrics::Dict`: Tracked history of loss and fidelity across iterations.
