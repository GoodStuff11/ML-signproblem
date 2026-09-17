#=
check_param_count_discrepancy.jl

Why does benchmark_timings.jl report a different parameter count for the exact
matrix-exponential ansatz than for the P=1 Trotter ansatz on the same system?

Builds BOTH generator sets for one system and reports:
  * the Trotter gate count (with and without diagonal gates)
  * the exact-path t_keys count
  * how many of the exact-path operators are identically zero once
    make_antihermitian has been applied

Usage (from experimenting/ed/):
  julia --project=.. testing/check_param_count_discrepancy.jl "N=(4, 3)_3x3"
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

function main(args)
    system = isempty(args) ? "N=(4, 3)_3x3" : args[1]
    folder = data_folder(system)
    antihermitian = true

    U_values, state_vecs, indexer, precomputed_structures, N_elec, spin_conserved, use_symmetry, _ =
        load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=true)

    Lvec = parse_lattice_dimension(folder)
    N_sites = prod(Lvec)
    dim = length(indexer.inv_comb_dict)

    println("\nsystem=$system  Lvec=$Lvec  N_sites=$N_sites  dim=$dim  use_symmetry=$use_symmetry")

    # ── Trotter generator set, exactly as run_trotter_scan_optimization.jl builds it
    gates_nodiag = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
        include_diagonal=false)
    gates_diag = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true,
        include_diagonal=true)
    n_actually_diagonal = count(Trotter.TamFermion.is_diagonal_gate, gates_diag)

    println("\n[trotter] enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, ...)")
    println("  include_diagonal=false (what --antihermitian uses) : $(length(gates_nodiag)) gates")
    println("  include_diagonal=true                              : $(length(gates_diag)) gates")
    println("  gates satisfying is_diagonal_gate                  : $n_actually_diagonal")

    # ── Exact generator set, exactly as ensure_operator_structure! builds it
    sd = ensure_operator_structure!(2, Dict{Int,Dict{Symbol,Any}}(), indexer, spin_conserved,
        use_symmetry, false, :spin_first, ColSnake(), precomputed_structures, antihermitian, 0.01 + 0im)
    t_keys = sd[:t_keys]
    ops = sd[:ops]
    println("\n[exact] ensure_operator_structure!(2, ...) -> create_randomized_nth_order_operator")
    println("  t_keys (= parameter count)                         : $(length(t_keys))")

    # Which of those operators are identically zero after antihermitization?
    zero_ops = Int[]
    for i in eachindex(ops)
        I, J, V = ops[i]
        M = sparse(I, J, V, dim, dim)
        iszero(nnz(dropzeros!(M))) && push!(zero_ops, i)
    end
    println("  operators that are identically ZERO (antihermitian): $(length(zero_ops))")
    println("  non-zero (effective) parameters                    : $(length(t_keys) - length(zero_ops))")

    println("\n  exact t_keys - trotter(no diag) = $(length(t_keys) - length(gates_nodiag))")
    println("  C(2N,2) = number of 2-body density-density terms  = $(binomial(2N_sites, 2))")

    # Are the zero operators exactly the diagonal (density-density) keys?
    diag_keys = 0
    for (i, k) in enumerate(t_keys)
        # a key is diagonal when its creation multiset equals its annihilation multiset
        if is_diagonal_key(k)
            diag_keys += 1
        end
    end
    println("  t_keys that are diagonal (cre multiset == ann)     : $diag_keys")

    return 0
end

"""
    is_diagonal_key(k) -> Bool

A 2-body operator key is diagonal (a density-density term) when the modes it
creates are exactly the modes it annihilates. `t_keys` entries carry the
`(site, spin, :create/:annihilate)` triples produced by
`create_randomized_nth_order_operator`.
"""
function is_diagonal_key(k)
    cre = Any[]
    ann = Any[]
    for op in k
        site, spin, kind = op
        kind == :create ? push!(cre, (site, spin)) : push!(ann, (site, spin))
    end
    return sort(string.(cre)) == sort(string.(ann))
end

main(ARGS)
