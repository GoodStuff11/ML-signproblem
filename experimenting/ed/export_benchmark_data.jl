using Lattices
using Combinatorics
using SparseArrays
using LinearAlgebra
using Statistics
using ExponentialUtilities
using Zygote
using JLD2
using Plots
import Graphs
using JSON
using ChainRulesCore
using Optimization
using OptimizationOptimJL

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

function run_benchmark()
    # Attempt to load matrices and vectors produced by ed_optimization cache.
    folder = "data/N=(4, 4)_4x2"
    file_path = joinpath(folder, "meta_data_and_E.jld2")
    dic = load_saved_dict(file_path)

    meta_data = dic["meta_data"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    indexer = dic["indexer"]

    k_min = 1
    target_vecs = all_full_eig_vecs[k_min]
    if indexer isa Vector
        indexer = indexer[k_min]
    end

    if target_vecs isa AbstractMatrix
        state1 = target_vecs[1, :]
        state2 = target_vecs[end, :]
    else
        vec1 = target_vecs[1]
        vec2 = target_vecs[end]
        state1 = vec1 isa AbstractVector ? vec1 : vec1[:, 1]
        state2 = vec2 isa AbstractVector ? vec2 : vec2[:, 1]
    end

    if typeof(indexer) <: JLD2.ReconstructedMutable
        # Handle JLD2 version discrepancies in CombinationIndexer fields manually
        a = indexer.a
        comb_dict = indexer.comb_dict
        inv_comb_dict = indexer.inv_comb_dict
        # We assume the lattice shape from the data folder or default
        indexer = CombinationIndexer(a, comb_dict, inv_comb_dict, (4, 2), nothing)
    end

    # Get a real structured operator using order 1
    # To keep it standard, force a seed
    t_dict = create_randomized_nth_order_operator(1, indexer; magnitude=1.0, omit_H_conj=true, conserve_spin=true, normalize_coefficients=false, conserve_momentum=false)
    rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
    t_keys = collect(keys(t_dict))
    param_index_map = build_param_index_map(ops_list, t_keys)

    dim = length(state1)

    ops = []

    for k in collect(keys(t_dict))
        _rows, _cols, _signs, _ = build_n_body_structure(Dict(k => 1.0), indexer)
        push!(ops, make_hermitian(sparse(_rows, _cols, _signs, dim, dim)))
    end
    parameter_mapping = nothing
    parity = nothing

    t_vals = real(collect(values(t_dict)))

    # Warmup
    f_adjoint(t) = adjoint_loss(t, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, true, false)
    Zygote.withgradient(f_adjoint, t_vals)

    # Benchmark
    println("Starting Benchmark (Julia)...")
    times = []
    for _ in 1:10
        t = @elapsed begin
            res = Zygote.withgradient(f_adjoint, t_vals)
        end
        push!(times, t)
    end

    println("Julia Adjoint Gradient Time (N=$(length(t_vals)), dim=$dim):")
    println("  Mean: ", mean(times), " s")
    println("  Std: ", std(times), " s")
    println("  Min: ", minimum(times), " s")

    # Save to disk for python benchmark
    save_dict = copy(t_dict)

    # Run once to get the exact value and gradient
    exact_loss, exact_grad = Zygote.withgradient(f_adjoint, t_vals)
    exact_grad = exact_grad[1] # gradient is a tuple

    jldsave("optimization_python/benchmark_data.jld2"; t_vals=t_vals, ops=ops, state1=state1, state2=state2, dim=dim, exact_loss=exact_loss, exact_grad=exact_grad)
end

run_benchmark()
