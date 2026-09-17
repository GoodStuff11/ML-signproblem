# Trotter GPU Acceleration Pipeline
Documents the GPU architecture, state propagation, adjoint pullback, and gate matrix streaming.

---
## 1. Mathematical and Algorithmic Foundation
The Trotter unitary ansatz evolves an initial state $|\psi_0\rangle$ under $P$ layers of $M_{\text{gates}}$ fermionic excitation gates:
$$|\psi(A)\rangle = \prod_{l=1}^{P} \prod_{k=1}^{M_{\text{gates}}} \exp(A_{l,k} \hat{\tau}_k) |\psi_0\rangle$$
where $\hat{\tau}_k$ are antihermitian generators ($\hat{\tau}_k^\dagger = -\hat{\tau}_k$) satisfying $\hat{\tau}_k^3 = -\hat{\tau}_k$ on the orthogonal complement of the kernel. The exponential expansion evaluates in closed form:
$$\exp(a \hat{\tau}_k) = I + \sin(a)\hat{\tau}_k + (1 - \cos(a))\hat{\tau}_k^2$$
For diagonal gates under antihermitian conditions ($\hat{\tau}_k = 0$), the operator reduces to the identity matrix $I$.

---
## 2. GPU Memory Architecture and `GpuGateOps`
High-level structure defined in [`trotter_gpu_kernels.jl`](file:///home/jek354/research/ML-signproblem/experimenting/ed/trotter_gpu_kernels.jl):
- `GpuGateOps`:
  - `tau_cpu::Vector{Any}`: Sparse host matrices of type `SparseMatrixCSC{datatype, Int32}`.
  - `tau_dev::Vector{Any}`: Pre-cached device sparse matrices `CUDA.CUSPARSE.CuSparseMatrixCSC`, or `nothing` when streaming mode is active.
  - `is_diag::Vector{Bool}`: Flags indicating whether each gate is diagonal in the Fock basis.
  - `sign0_val::Vector{Float64}`: Jordan-Wigner reference phase signs.
  - `w1::Any`: Scratch vector preallocated on GPU device memory for matrix-vector products.
  - `w2::Any`: Second scratch vector preallocated on GPU device memory.
  - `pre_cached_all::Bool`: Flag indicating whether all gate matrices reside in VRAM (`true`) or are streamed on-the-fly (`false`).

---
## 3. Streaming Mode (Option B) vs Full GPU Caching
- **Memory Scaling**: Storing all `tau_dev` matrices for large lattices ($4\times 4$, $d = 4,008,576$) requires $\sim 45\text{ GiB}$ of VRAM. To avoid CUDA Out-Of-Memory (OOM) errors, `get_gpu_gate_ops` evaluates total column pointer bytes:
  $$\text{colptr\_bytes} = M_{\text{gates}} \times (d + 1) \times 4$$
  When $\text{colptr\_bytes} \ge 4\text{ GiB}$, `pre_cache_all` defaults to `false`.
- **Dynamic Retrieval (`_get_gpu_tau_mat`)**:
  When `tau_dev[k] !== nothing`, returns cached device matrix. When `tau_dev[k] === nothing`, converts host matrix `tau_cpu[k]` to `CuSparseMatrixCSC` on the fly.
- **Gate Application (`gpu_apply_gate_exp!`)**:
  Applies the gate exponential to device vectors using `CUSPARSE.mv!`. Accepts an optional `tau` parameter to reuse device sparse matrices between gradient evaluation and backward propagation in [`backward_adjoint_propagation`](file:///home/jek354/research/ML-signproblem/experimenting/ed/trotter_evolution.jl#L129).
