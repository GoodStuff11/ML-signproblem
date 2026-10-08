# Documentation Index
Central index for codebase architecture, mathematical conventions, data schemas, and GPU pipelines.

---
## Domains
- [[trotter_gpu_pipeline]]: Details the GPU acceleration architecture for Trotterized unitary optimization, including host-to-device gate matrix streaming (Option B), checkpoint rematerialization, and `GpuGateOps`.
- [[ed_file_formats]]: Schema and physical/mathematical correspondence for Lanczos ED `.h5` and optimization output `.jld2` files.
- [[ucc_hardware_encoding]]: Two-qubit/single-qubit gate counts and depth of the Trotterized UCC circuit (`ucc_hardware_encoding.jl`) under the Yordanov 2020 Jordan–Wigner circuits and on the González-Cuadra 2023 fermionic tweezer processor; inputs, mode convention, optimizations, reference numbers.