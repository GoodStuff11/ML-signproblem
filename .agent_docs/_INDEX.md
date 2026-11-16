# Documentation Index
Central index for codebase architecture, mathematical conventions, data schemas, and GPU pipelines.

---
## Domains
- [[trotter_gpu_pipeline]]: Details the GPU acceleration architecture for Trotterized unitary optimization, including host-to-device gate matrix streaming (Option B), checkpoint rematerialization, and `GpuGateOps`.
- [[ed_file_formats]]: Schema and physical/mathematical correspondence for Lanczos ED `.h5` and optimization output `.jld2` files, including the `eig_target`/`eig_sigma` fields and the excited-state data folders.
- [[data_pipeline]]: `load_ED_data` loaders, `target_vecs` row layout, how the reference Slater determinant is chosen, and the `sector`/`target_state`/`reference_target` options for excited-state targets.
- [[ucc_hardware_encoding]]: Two-qubit/single-qubit gate counts and depth of the Trotterized UCC circuit, and of the HVA with the Cade et al. 2020 fermionic-swap network, (`ucc_hardware_encoding.jl`) under the Yordanov 2020 Jordan–Wigner circuits and on the González-Cuadra 2023 fermionic tweezer processor; inputs, mode convention, optimizations, reference numbers.