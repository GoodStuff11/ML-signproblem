using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using OptimizationOptimJL
using JLD2
using ExponentialUtilities

# Make sure we load the local code files from the parent directory
include("../utility_functions.jl")
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")
include("../logging.jl")

function run_gradient_check()
    println("--- RUNNING GRADIENT CORRECTNESS CHECK (FINITE DIFFERENCES VS ADJOINT) ---")

    # Load test data
    dic = load_saved_dict("data/N=(3, 3)_3x2/meta_data_and_E.jld2")
    meta_data = dic["meta_data"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    indexer = dic["indexer"]
    if indexer isa Vector
        indexer = indexer[1]
    end

    state1 = all_full_eig_vecs[1][1, :]
    state2 = all_full_eig_vecs[1][min(30, size(all_full_eig_vecs[1], 1)), :]
    dim = length(indexer.inv_comb_dict)

    # Generate random parameters for order 2
    # We will build operators for order 2 and check gradients
    # We use ensure_operator_structure! which we'll expose or access via cache
    order = 2
    op_cache = Dict{Int,Dict{Symbol,Any}}()

    # We'll call the operator structure loader
    struct_data = try
        # Call it if it's already top level (we will make it top level)
        ensure_operator_structure!(order, op_cache, indexer, !isa(meta_data["electron count"], Number), true, false, :coordinate_first, Dict(), false, 0.01)
    catch e
        # Fallback if we haven't refactored yet, we can create it manually
        println("Could not call top-level ensure_operator_structure! yet, skipping direct grad check.")
        rethrow(e)
    end

    rows = struct_data[:rows]
    cols = struct_data[:cols]
    signs = struct_data[:signs]
    ops = struct_data[:ops]
    param_index_map = struct_data[:param_index_map]
    parameter_mapping = struct_data[:parameter_mapping]
    parity = struct_data[:parity]

    P = length(struct_data[:sym_data][1]) # number of param groups

    for L in [1, 2]
        println("\nTesting num_exponentials = $L (total params = $(L * P))")
        t_vals = 0.05 * (2 * rand(L * P) .- 1)

        # Test 1: Overlap Loss CPU Gradient
        f_loss = t -> adjoint_loss(t, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, true, false; num_exponentials=L)

        # Pullback gradient
        loss_val, back = Zygote.withgradient(f_loss, t_vals)
        grad_adj = back[1] - 1e-3 * t_vals

        # Finite difference gradient
        h = 1e-6
        grad_fd = zeros(length(t_vals))
        for i in eachindex(t_vals)
            t_plus = copy(t_vals)
            t_plus[i] += h
            l_plus = f_loss(t_plus)

            t_minus = copy(t_vals)
            t_minus[i] -= h
            l_minus = f_loss(t_minus)

            grad_fd[i] = (l_plus - l_minus) / (2 * h)
        end

        diff = norm(grad_adj - grad_fd) / (norm(grad_adj) + 1e-9)
        println("  Overlap CPU gradient diff norm (rel): $diff")
        if diff < 1e-4
            println("  ✓ Overlap CPU gradient check PASSED!")
        else
            println("  ✗ Overlap CPU gradient check FAILED!")
        end
    end
end

function run_optimization_comparison()
    println("\n--- RUNNING OPTIMIZATION COMPARISON: L=1 vs L=2 ---")

    dic = load_saved_dict("data/N=(3, 3)_3x2/meta_data_and_E.jld2")
    meta_data = dic["meta_data"]
    all_full_eig_vecs = dic["all_full_eig_vecs"]
    indexer = dic["indexer"]
    if indexer isa Vector
        indexer = indexer[1]
    end

    # Optimize unitary mapping using test_map_to_state
    scheme = [2]
    initial_coefficients = []
    for L in [3]
        println("\nStarting optimization with scheme = $scheme, num_exponentials = $L")
        instructions = Dict(
            "starting state" => Dict("U index" => 1, "levels" => 1),
            "ending state" => Dict("U index" => 45, "levels" => 1),
            "optimization_scheme" => scheme,
            # "use symmetry" => false,
            "num_exponentials" => L,
            "initialization_samples" => 10,
            "multi_start_iters" => 30,
            "multi_start_samples" => 3
        )

        data = test_map_to_state(all_full_eig_vecs[1], instructions, indexer, !isa(meta_data["electron count"], Number);
            maxiters=200, gradient=:adjoint_gradient, initial_coefficients=initial_coefficients)
        # println(data["coefficients"])
        opt_coeffs = data["coefficients"][1][2]
        initial_coefficients = Any[nothing, [opt_coeffs; zero(opt_coeffs)]]
        final_loss = data["loss_metrics"][end][end]
        println("Finished scheme = $scheme, num_exponentials = $L. Final Loss: $final_loss")
    end
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_multi_exponential")
    with_logging(log_path) do
        run_gradient_check()
        run_optimization_comparison()
    end
end
