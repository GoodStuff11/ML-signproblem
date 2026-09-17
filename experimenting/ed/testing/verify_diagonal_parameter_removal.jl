#=
verify_diagonal_parameter_removal.jl

Regression test for dropping the density-density (diagonal) parameters from the exact
matrix-exponential ansatz when `antihermitian=true`.

Checks, for one system:
  1. antihermitian=true  -> parameter count equals the Trotter P=1 gate count, and no
     surviving operator is identically zero.
  2. antihermitian=false -> parameter count is unchanged (diagonal terms are real and
     needed for a Hermitian generator).
  3. The `precomputed_structures` branch of ensure_operator_structure! produces exactly the
     same structure as generating from scratch.
  4. Loss and gradient are unchanged: evaluating the OLD (unfiltered) structure with zeros
     in the diagonal slots gives the same loss and the same non-diagonal gradient entries
     as the NEW (filtered) structure, for both the overlap and the energy loss.

Usage (from experimenting/ed/):
  julia --project=.. testing/verify_diagonal_parameter_removal.jl ["N=(4, 3)_3x3"]
=#

using Lattices
using Combinatorics
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using HDF5
using JLD2
using Zygote
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using KrylovKit

const HERE = joinpath(@__DIR__, "..")
include(joinpath(HERE, "data_path.jl"))
include(joinpath(HERE, "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(HERE, "ed_objects.jl"))
include(joinpath(HERE, "ed_functions.jl"))
include(joinpath(HERE, "trotter.jl"))
import .Trotter
include(joinpath(HERE, "ed_optimization.jl"))

const FAILURES = String[]

function check(label::String, ok::Bool, detail="")
    println(ok ? "  PASS  $label $detail" : "  FAIL  $label $detail")
    ok || push!(FAILURES, label)
    return ok
end

function main(args)
    system = isempty(args) ? "N=(4, 3)_3x3" : args[1]
    folder = data_folder(system)
    u_index = 33

    U_values, state_vecs, indexer, precomputed_structures, N_elec, spin_conserved, use_symmetry, _ =
        load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)

    Lvec = parse_lattice_dimension(folder)
    dim = length(indexer.inv_comb_dict)
    has_prepended_ref = (state_vecs isa AbstractMatrix) && (size(state_vecs, 1) == length(U_values) + 1)
    state1 = state_vecs[1, :]
    state2 = state_vecs[has_prepended_ref ? u_index + 1 : u_index, :]

    subspace = reconstruct_subspace(indexer, spin_conserved)
    H_hop, H_int = create_hubbard_matrices(subspace; indexer=indexer, get_indexer=false,
        sign_convention=:spin_first, lattice_ordering=ColSnake())
    H = H_hop + U_values[u_index] * H_int

    println("system=$system  Lvec=$Lvec  dim=$dim  use_symmetry=$use_symmetry  U=$(U_values[u_index])")
    isempty(precomputed_structures) ||
        println("NOTE: this dataset ships precomputed_structures; test 3 exercises that branch with its own copy anyway")

    fresh(antiherm) = ensure_operator_structure!(2, Dict{Int,Dict{Symbol,Any}}(), indexer,
        spin_conserved, use_symmetry, false, :spin_first, ColSnake(), Dict(), antiherm, 0.01 + 0im)

    # ── 1. antihermitian: count matches Trotter, nothing left is a zero operator ──────────
    println("\n[1] antihermitian=true")
    sd_new = fresh(true)
    gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
        include_diagonal=false)
    check("parameter count == Trotter P=1 gate count",
        length(sd_new[:t_keys]) == length(gates),
        "($(length(sd_new[:t_keys])) vs $(length(gates)))")

    n_zero = count(eachindex(sd_new[:ops])) do i
        I, J, V = sd_new[:ops][i]
        iszero(nnz(dropzeros!(sparse(I, J, V, dim, dim))))
    end
    check("no surviving operator is identically zero", n_zero == 0, "(found $n_zero)")
    check("no surviving key is diagonal",
        !any(is_diagonal_operator_key, sd_new[:t_keys]))

    # ── 2. Hermitian generators keep the diagonal terms ──────────────────────────────────
    println("\n[2] antihermitian=false (must be unchanged)")
    sd_herm = fresh(false)
    gates_diag = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
        include_diagonal=true)
    check("parameter count == Trotter gate count with diagonals",
        length(sd_herm[:t_keys]) == length(gates_diag),
        "($(length(sd_herm[:t_keys])) vs $(length(gates_diag)))")
    check("diagonal keys are still present",
        count(is_diagonal_operator_key, sd_herm[:t_keys]) == binomial(2 * prod(Lvec), 2))

    # ── 3. precomputed_structures branch agrees with generating from scratch ─────────────
    println("\n[3] precomputed_structures branch")
    t_dict_full, t_keys_full = create_randomized_nth_order_operator(2, indexer, true;
        magnitude=1.0 + 0im, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved,
        normalize_coefficients=false, conserve_momentum=false, sign_convention=:spin_first)
    rows_f, cols_f, signs_f, ops_list_f = build_n_body_structure_from_keys(
        t_keys_full, indexer, typeof(t_dict_full[t_keys_full[1]]); sign_convention=:spin_first)
    pim_f = build_param_index_map(ops_list_f, t_keys_full)
    cache = Dict((2, use_symmetry) => Dict(
        :rows => rows_f, :cols => cols_f, :signs => signs_f,
        :ops_list => ops_list_f, :t_keys => t_keys_full, :param_index_map => pim_f))

    sd_cached = ensure_operator_structure!(2, Dict{Int,Dict{Symbol,Any}}(), indexer,
        spin_conserved, use_symmetry, false, :spin_first, ColSnake(), cache, true, 0.01 + 0im)
    check("filtered cached t_keys == freshly generated t_keys",
        sd_cached[:t_keys] == sd_new[:t_keys],
        "($(length(sd_cached[:t_keys])) vs $(length(sd_new[:t_keys])))")
    check("filtered cached rows/cols/signs match", sd_cached[:rows] == sd_new[:rows] &&
                                                   sd_cached[:cols] == sd_new[:cols] && sd_cached[:signs] == sd_new[:signs])
    check("filtered cached param_index_map matches", sd_cached[:param_index_map] == sd_new[:param_index_map])

    # ── 4. loss and gradient unchanged ───────────────────────────────────────────────────
    # Build the OLD behaviour (antihermitian structure that still carries the diagonal keys)
    # by filtering nothing, then compare against the new filtered structure with the same
    # values on the shared keys and zeros on the dropped ones.
    println("\n[4] loss and gradient equivalence (old structure with zeroed diagonals vs new)")
    t_dict_old, t_keys_old = create_randomized_nth_order_operator(2, indexer, true;
        magnitude=0.01 + 0im, omit_H_conj=!use_symmetry, conserve_spin=spin_conserved,
        normalize_coefficients=false, conserve_momentum=false, omit_diagonal=false,
        sign_convention=:spin_first)
    rows_o, cols_o, signs_o, ops_list_o = build_n_body_structure_from_keys(
        t_keys_old, indexer, typeof(t_dict_old[t_keys_old[1]]); sign_convention=:spin_first)
    pim_o = build_param_index_map(ops_list_o, t_keys_old)
    sd_old = ensure_operator_structure!(2, Dict{Int,Dict{Symbol,Any}}(), indexer,
        spin_conserved, use_symmetry, false, :spin_first, ColSnake(),
        Dict((2, use_symmetry) => Dict(:rows => rows_o, :cols => cols_o, :signs => signs_o,
            :ops_list => ops_list_o, :t_keys => t_keys_old, :param_index_map => pim_o)),
        false, 0.01 + 0im)   # antihermitian=false here ONLY to skip the new filter; the ops
    # below are rebuilt for antihermitian use via setup_loss_functions(..., antihermitian=true)

    # Rebuild sd_old's ops with the antihermitian convention, the way ensure_operator_structure!
    # does, so the only difference from sd_new is the presence of the diagonal parameters.
    sd_old_anti = build_antihermitian_ops(sd_old, dim)

    new_pos = Dict(k => i for (i, k) in enumerate(sd_new[:t_keys]))
    Random.seed!(7)
    t_new = (2 * rand(length(sd_new[:t_keys])) .- 1) * 0.01
    t_old = zeros(length(sd_old_anti[:t_keys]))
    shared = Int[]
    for (i, k) in enumerate(sd_old_anti[:t_keys])
        if haskey(new_pos, k)
            t_old[i] = t_new[new_pos[k]]
            push!(shared, i)
        end
    end
    check("every new key is present in the old key set", length(shared) == length(t_new))

    for loss_type in (:overlap, :energy)
        f_new = loss_fn(sd_new, loss_type, dim, state1, state2, H, use_symmetry)
        f_old = loss_fn(sd_old_anti, loss_type, dim, state1, state2, H, use_symmetry)

        l_new, l_old = f_new(t_new), f_old(t_old)
        check("$loss_type loss unchanged", isapprox(l_new, l_old; rtol=1e-12, atol=1e-14),
            "(new=$l_new old=$l_old)")

        g_new = Zygote.gradient(f_new, t_new)[1]
        g_old = Zygote.gradient(f_old, t_old)[1]
        check("$loss_type gradient unchanged on shared parameters",
            isapprox(g_new, g_old[shared]; rtol=1e-10, atol=1e-14),
            "(max|Δ| = $(maximum(abs.(g_new .- g_old[shared]))))")

        dropped = setdiff(1:length(t_old), shared)
        check("$loss_type gradient is zero on the dropped diagonal parameters",
            all(iszero, g_old[dropped]), "(max|g| = $(maximum(abs.(g_old[dropped]))))")
    end

    println()
    if isempty(FAILURES)
        println("ALL CHECKS PASSED")
        return 0
    end
    println("FAILED: ", join(FAILURES, "; "))
    return 1
end

"""
    build_antihermitian_ops(sd, dim) -> Dict

Rebuild `sd[:ops]` using the antihermitian convention (`s` at `(r,c)`, `-conj(s)` at `(c,r)`),
leaving the parameter set itself untouched. Reproduces what `ensure_operator_structure!` did
for antihermitian generators BEFORE the diagonal parameters were dropped, so the old and new
behaviour can be compared directly.
"""
function build_antihermitian_ops(sd, dim)
    rows, cols, signs = sd[:rows], sd[:cols], sd[:signs]
    t_keys, param_index_map = sd[:t_keys], sd[:param_index_map]
    indices_by_param = [Int[] for _ in eachindex(t_keys)]
    for k in eachindex(param_index_map)
        push!(indices_by_param[param_index_map[k]], k)
    end
    ops = []
    for i in eachindex(t_keys)
        rows_sub, cols_sub, vals_sub = Int[], Int[], ComplexF64[]
        for j in indices_by_param[i]
            push!(rows_sub, rows[j]); push!(cols_sub, cols[j]); push!(vals_sub, signs[j])
            push!(rows_sub, cols[j]); push!(cols_sub, rows[j]); push!(vals_sub, -conj(signs[j]))
        end
        push!(ops, (rows_sub, cols_sub, vals_sub))
    end
    out = copy(sd)
    out[:ops] = ops
    return out
end

"""
    loss_fn(sd, loss_type, dim, state1, state2, H, use_symmetry) -> Function

Wrap `setup_loss_functions` so the production CPU adjoint loss for `sd` can be called as
`f(t_vals)`.
"""
function loss_fn(sd, loss_type, dim, state1, state2, H, use_symmetry)
    _, _, _, f_adjoint, _ = setup_loss_functions(loss_type, 2, sd[:ops], sd[:rows], sd[:cols],
        sd[:signs], sd[:param_index_map], sd[:parameter_mapping], sd[:parity], dim,
        state1, state2, H, use_symmetry, true, false, nothing, nothing, nothing, nothing, 1)
    return t -> f_adjoint(t, nothing)
end

exit(main(ARGS))
