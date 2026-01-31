using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JLD2
using ExponentialUtilities
using BenchmarkTools
using Printf

include("ed_objects.jl")
include("ed_functions.jl")
include("ed_optimization.jl")
include("utility_functions.jl")

# --- Configuration ---
USE_LARGE_SYSTEM = false

# --- Data Loading ---
if USE_LARGE_SYSTEM
    println("Loading Data (N=6_2x3)...")
    dic = load_saved_dict("data/N=6_2x3/meta_data_and_E.jld2")
else
    println("Loading Data (N=3_2x3)...")
    dic = load_saved_dict("data/N=3_2x3/meta_data_and_E.jld2")
end

indexer = dic["indexer"]
dim = length(indexer.inv_comb_dict)
println("Dimension: $dim")

Random.seed!(42)
state1 = normalize(randn(ComplexF64, dim))
state2 = normalize(randn(ComplexF64, dim))

println("Setting up operators...")
order = 2
t_dict = create_randomized_nth_order_operator(order, indexer; magnitude=0.5, omit_H_conj=false, conserve_spin=false)
rows, cols, signs, ops_list = build_n_body_structure(t_dict, indexer)
t_keys = collect(keys(t_dict))
param_index_map = build_param_index_map(ops_list, t_keys)
num_params = length(t_dict)
println("Full Parameters: $num_params")

inv_param_map, parameter_mapping, parity = find_symmetry_groups(
    t_keys,
    maximum(indexer.a).coordinates...,
    hermitian=true,
    trans_x=true,
    trans_y=true,
    spin_symmetry=true
)
reduced_dim = length(inv_param_map)
println("Reduced Parameters: $reduced_dim")

println("Precomputing Symmetric Operators...")
ops_sym = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, reduced_dim)
full_param_to_reduced = Dict{Int,Int}()
for (k, group_indices) in enumerate(inv_param_map)
    for idx in group_indices
        full_param_to_reduced[idx] = k
    end
end
reduced_to_ops = [Int[] for _ in 1:reduced_dim]
reduced_to_signs = [ComplexF64[] for _ in 1:reduced_dim]
signs_c = ComplexF64.(signs)
for i in 1:length(ops_list)
    full_p_idx = param_index_map[i]
    if haskey(full_param_to_reduced, full_p_idx)
        red_p_idx = full_param_to_reduced[full_p_idx]
        push!(reduced_to_ops[red_p_idx], i)
        push!(reduced_to_signs[red_p_idx], signs_c[i] * parity[full_p_idx])
    end
end
for k in 1:reduced_dim
    Hk = sparse(rows[reduced_to_ops[k]], cols[reduced_to_ops[k]], reduced_to_signs[k], dim, dim)
    ops_sym[k] = make_hermitian(Hk)
end

t_vals_sym = rand(Float64, reduced_dim)

# --- 1. Zygote (Symmetric) ---
println("\n--- 1. Baseline: Zygote (Symmetric) ---")
function f_zygote_sym(t)
    t_c = ComplexF64.(t)
    # Reconstruct from symmetric parameters
    # Note: For Zygote to work efficiently here without Sparse Arrays in the AD path,
    # we usually need dense ops or a Zygote-friendly sparse construction.
    # But here we will use the 'ops_sym' and sum them. Zygote can differentiate the sum.
    # Summing sparse matrices might be slow in Zygote if not careful, but let's test.
    # Actually, let's use the explicit sparse construction which is what we used before.
    # vals = update_values(signs_c, param_index_map, t_c, parameter_mapping, parity)
    # This sparse call might be the bottleneck for AD, but it's what was used.
    # Alternatively, use the ops_sym list:
    H = sum(t_c[k] * ops_sym[k] for k in 1:length(t))
    U = exp(1im * Matrix(H))
    return 1 - abs2(state2' * U * state1)
end

# Warmup and Run
println("Calculating Zygote Gradient...")
Zygote.gradient(f_zygote_sym, t_vals_sym) # Warmup
t_zygote = @elapsed grad_zygote = Zygote.gradient(f_zygote_sym, t_vals_sym)[1]
println("Zygote Time: $t_zygote s")
println("Zygote Grad Norm: $(norm(grad_zygote))")

# --- Benchmarking ---

println("\n--- 1. Trotter Exact (Auxiliary Matrix Method) ---")
# Uses the original trotter_gradient_bench_optimized! from previous session
# (Re-implemented here for standalone benchmark)
function trotter_grad_exact!(grad, t_vals, ops_array, state1, state2)
    dim = length(state1)
    H = sum(t_vals[i] * ops_array[i] for i in 1:length(t_vals))
    psi_u = expv(1im, H, state1)
    overlap = state2' * psi_u
    # Removed Threads.@threads for debugging/safety
    for k in 1:length(t_vals)
        M = AuxiliaryMatrix(H, ops_array[k], dim)
        # Increased m=100 matching production fix
        v_out = expv(1im, M, [state1; zeros(ComplexF64, dim)]; m=100)
        d_psi = v_out[dim+1:end]
        grad[k] = -2 * real(conj(overlap) * (state2' * d_psi))
    end
    return grad
end

println("\n--- Computing Exact Reference Gradient ---")
grad_exact = zeros(Float64, reduced_dim)
trotter_grad_exact!(grad_exact, t_vals_sym, ops_sym, state1, state2) # Warmup
t_exact = @elapsed trotter_grad_exact!(grad_exact, t_vals_sym, ops_sym, state1, state2)
println("Ref Gradient Norm: $(norm(grad_exact))")
# --- 3. Trotterized Sweep ---
println("\n--- 3. Trotterized Adjoint Sweep ---")

configs = [
    (order=1, steps=[1, 5, 10, 20]),
    (order=2, steps=[1, 5, 10, 20]),
    (order=4, steps=[1, 2, 5, 10])
]

println("\n--- Comprehensive Benchmark Results ---")
@printf("%-15s | %-6s | %-6s | %-12s | %-12s\n", "Method", "Order", "Steps", "Time (s)", "Error (vs Zyg)")
println("-"^65)

# Zygote Row
@printf("%-15s | %-6s | %-6s | %-12.6f | %-12s\n", "Zygote (Sym)", "-", "-", t_zygote, "0.0")

# Trotter Exact Row
err_exact = norm(grad_zygote - grad_exact) / norm(grad_zygote)
@printf("%-15s | %-6s | %-6s | %-12.6f | %-12.4e\n", "Trotter (Exact)", "-", "-", t_exact, err_exact)

println("-"^65)

for config in configs
    ord = config.order
    for steps in config.steps
        # Accuracy
        grad_adj = trotter_gradient_adjoint(state1, state2, ops_sym, t_vals_sym, ord, steps)
        err = norm(grad_zygote - grad_adj) / norm(grad_zygote)

        # Timing
        trotter_gradient_adjoint(state1, state2, ops_sym, t_vals_sym, ord, steps) # Warmup
        t = @elapsed trotter_gradient_adjoint(state1, state2, ops_sym, t_vals_sym, ord, steps)

        @printf("%-15s | %-6d | %-6d | %-12.6f | %-12.4e\n", "Trotter (Adj)", ord, steps, t, err)
    end
    println("-"^65)
end

println("\nBenchmark Complete.")
