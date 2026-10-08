# UCC hardware encoding (`experimenting/ed/ucc_hardware_encoding.jl`)
Gate counts and depth of the Trotterized UCC circuit of `trotter.jl` on two kinds of hardware. Library and command-line script. As a library, `include` it after or instead of `trotter.jl` (it includes `trotter.jl` itself when `Main.Trotter` is not defined); its `(@main)` is defined only when the file is the program being run (`abspath(PROGRAM_FILE) == @__FILE__`), so including scripts can define their own. From the command line: `julia --project=.. ucc_hardware_encoding.jl --lattice=4x3 [options]`; every keyword of the two cost functions has a `--option` (full list in the file header), plus `--ansatz=standard|hva`, `--coefficients_file=<per-U .jld2>` (gates from the sibling `_shared.jld2`), `--encoding=yordanov|fermionic|both` and `--optimize=true|false|both`. Arguments are parsed by `parse_ucc_hardware_arguments`; output is also logged to `logs/<date>/`. Tests: `experimenting/ed/testing/test_ucc_hardware_encoding.jl`.

---
## Sources
- Yordanov, Arvidsson-Shukur, Barnes, Phys. Rev. A 102, 062612 (2020), arXiv:2005.14475 — CNOT-efficient single/double fermionic-excitation circuits for a Jordan–Wigner qubit register.
- González-Cuadra et al., PNAS 120, e2304294120 (2023), arXiv:2303.06985 — fermionic tweezer register; native gates $U^{(t)}_{i,j}(\theta_1,\theta_2,\theta_3)=e^{-i[\frac{\theta_1}{2}(e^{-i\theta_2}c^\dagger_ic_j+\text{h.c.})+\frac{\theta_3}{2}(n_i-n_j)]}$ (tunneling) and $U^{(\text{int})}_{i,j}(\theta)=e^{-i\theta n_in_j}$ (interaction).

---
## Inputs (shared by both cost functions)
- `gates::Vector{FGate}`: from `enumerate_ferm_excitations` (or `enumerate_ferm_excitations_HVA` with its `param_map`), in circuit order. For a saved run, use the gate order of that run (its `_shared.jld2` `"gates"` or the per-file `"gate_keys"`), because order changes depth and fusion.
- `N`: number of lattice sites. Spin-orbital (mode) index convention, identical to `TamFermion.excitation_operator_sector`: 0-based, spin-blocked; spin-up momentum $k$ is mode $k$, spin-down momentum $k$ is mode $N+k$; $2N$ modes in total.
- `coefficients` (optional): coefficient vector laid out as in `apply_unitary` (`A[(l-1)*length(gates)+g]` for layer `l`, gate `g`), or the reduced vector when `param_map` is given. Factors with `|a| <= drop_tol` (default `0.0`) are dropped, which removes exactly the zeros of a pruned `--target_fidelity` run (those files store the full-length vector with zeros).
- `num_exponentials`: Trotter layers; inferred from `length(coefficients)` when omitted, else 1.
- `antihermitian` (default `true`): generator convention of the run. Diagonal gates are dropped in the antihermitian convention (their generator vanishes).
- Supported factor kinds (`classify_excitation`): `:double` ($c^\dagger c^\dagger cc$), `:single` ($c^\dagger c$), `:number_pair` ($n_pn_q$), `:number` ($n_p$). Number-controlled excitations (a mode both created and annihilated), rank > 2, and diagonal products of > 2 number operators throw `ArgumentError`.

---
## `yordanov_encoding_cost(gates, N; ...)`
Per factor on $w$ Jordan–Wigner qubits ($w$ = `length(jw_support(e, position))`; the string covers the qubits between the 1st/2nd and 3rd/4th sorted qubits, so $w=n^{(df)}$ of the paper):
- double: $2w+5$ two-qubit gates (CNOT + 2 CZ), depth 11 ($w=4$) else $\max(13,2w-1)$; 21 single-qubit gates (8 parametrized $R_y$).
- single: $2w-1$ two-qubit gates, depth 3 ($w=2$) else $\max(5,2w-3)$; 7 single-qubit gates.
- `:number_pair`: 2 CNOT + 3 $R_z$, depth 2 (standard controlled phase, not from the paper); `:number`: 1 $R_z$.
- `antihermitian=false`: +2 single-qubit gates per excitation (S/S† conjugation).

Optimizations (`optimize=true` turns all on; each keyword overrides): `parity_network=:tree` (balanced parity tree, depth $\max(13,2\lceil\log_2 m\rceil+9)$ / $\max(5,2\lceil\log_2 m\rceil+3)$, $m$ = string qubits); `fuse_single_qubit` (adjacent single-qubit gates merged: 21→16, 7→6, counted from the paper's figures); `jw_ordering` (`:blocked` = all ↑ then all ↓; `:interleaved` = 0↑ 0↓ 1↑ 1↓ … via `jw_ordering_positions`; `:optimized` = `optimize_jw_ordering`, pairwise-swap hill climbing from both, minimizing $\sum w$, default when `optimize=true`); `reorder` (commutation-aware `schedule_blocks`, kept only if it beats program order). Depth model: each factor is a block occupying its whole JW support for its two-qubit depth; reported `depth` is two-qubit-gate depth.

---
## `fermionic_encoding_cost(gates, N; ...)`
One mode per register site, no strings, all-to-all connectivity (tweezers are moved). Native gates per factor (`fermionic_native_layers`):
- double = pair-tunneling gate, Fig. 3(a) of the paper: layers $T(\frac{\sqrt8\pi}{\sqrt{27}},\frac{\theta_2}{2}-\frac{\pi}{4},\frac{2\pi}{\sqrt{27}})$ on (c₁,a₁),(c₂,a₂) → $U^{(\text{int})}(-\theta_1)$ on (c₁,c₂),(a₁,a₂) → $T(\frac\pi2,\frac{\theta_2}{2}+\frac\pi2,0)$ → $U^{(\text{int})}(+\theta_1)$ → $T(\frac\pi2,\frac{\theta_2}{2}+\pi,0)$: 6 tunneling + 4 interaction, depth 5, verified to machine precision in the test.
- single = 1 tunneling gate; `:number_pair` = 1 interaction gate; `:number` = 1 single-mode phase.

Optimizations: `fuse` (peephole merge of consecutive same-kind gates on the same modes: tunneling∘tunneling is one tunneling gate, interactions and phases add angles); `choose_pairing` (pairing ((c₁,a₁),(c₂,a₂)) or ((c₁,a₂),(c₂,a₁)) — the same gate with $\theta_1\to-\theta_1$ — chosen to maximize merges); `reorder` (as above, resources = modes, `objective=:depth` or `:two_qubit`). `spin_conserving=true` (default) forbids spin-flipping tunneling pairs. Reported `n_two_qubit` = two-mode gates, `n_single_qubit` = single-mode phases, `depth` = native layers.

---
## Gates vs. free parameters
`n_excitations` counts gates (factors); `n_parameters` (`free_parameter_count`) counts independent variational parameters: one per gate without `param_map`, one per (layer, shared entry) with the HVA tie map, counting only entries with a surviving gate. HVA 4×3 open: 46 gates per layer (12 on-site + 17 bonds × 2 spins) carrying 5 / 29 / 46 parameters for `tie` = full / spin / none; tie=full over 12 layers gives the 60 EHV DOF of `gate_count_analysis.tex`. The CLI prints both lines (`free parameters`, `gates (factors)`).

---
## Gate order: `gate_order=:given` / `:best`
Factors on disjoint spin-orbitals commute, so with `:given` (default) only factors sharing a spin-orbital keep program order. With the sorted order of `enumerate_ferm_excitations`, 1081 of 1091 consecutive pairs share one at 4×3, so the given circuit is essentially serial (critical path 15454 of a 15722 sequential sum, Yordanov tree). `gate_order=:best` lets the factors go in any order and keeps the shallowest schedule from `packing_priorities` (longest first, widest first, busiest resource first, program order, `order_trials` random tie-breaks, default 32); `order_strategy` names the winner. It is a different circuit (coefficients would need re-optimizing) and a heuristic: the true optimum lies between its depth and `bounds.resource_bound`, the total load on the busiest qubit/mode, which no order can beat. 4×3: Yordanov 9629 (bound 8800), fermionic 1100 (bound 910); 4×4: 23075 (21275), 1960 (1695).
---
## Reference numbers (`num_exponentials=1`, antihermitian, sorted gate order)
| lattice | factors | Yordanov 2q / 1q / depth (optimized) | fermionic 2-mode / depth (optimized) |
|---|---|---|---|
| 3×2 | 114 | 1986 / 1824 / 1413 | 1121 / 548 |
| 4×2 | 300 | 5884 / 4800 / 3950 | 2974 / 1453 |
| 3×3 | 432 | 8976 / 6912 / 5852 | 4291 / 2111 |
| 4×3 | 1092 | 26356 / 17472 / 15685 | 10879 / 5354 |
| 4×4 | 2712 | 77336 / 43392 / 40340 | 27059 / 13310 |
Unoptimized Yordanov two-qubit counts equal $\sum(2w+5)$ from `68f6bef0eaaf5c0928a922c6/gate_count_scripts` for every lattice above. `optimize_jw_ordering` found no order with smaller $\sum w$ than spin-blocked for these gate sets; interleaved is worse at every size (4×3: mean $w$ 10.76 vs 9.57, 28964 vs 26356 two-qubit gates). Both results carry `bounds` (`depth_bounds`): `sequential`, `critical_path` (longest chain of mode-sharing factors in program order), `resource_bound` (max per-qubit/mode load), `max_concurrent` ($2N\div$ smallest block, e.g. 6 double excitations at once at 4×3).
