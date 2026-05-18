using Lattices
using LinearAlgebra
using Combinatorics
using SparseArrays
using Plots
import Graphs
using LaTeXStrings
using Statistics
using Random
using Zygote
using Optimization, OptimizationOptimisers
using JSON
using OptimizationOptimJL
using JLD2
using HDF5
using CUDA
using KrylovKit
# Ensure CUDA is available if possible
try
    using CUDA
    CUDA.set_runtime_version!(v"12.8")
    println("CUDA version: $(CUDA.versioninfo())")
catch
    println("CUDA not available, will only test CPU.")
end

# Include the source files
include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")

function test_matrix_free_gradients()
    println("Setting up small test system...")
    Random.seed!(42)
    
    # Small dimension to allow dense exponentiation in zygote_loss
    dim = 64
    num_params = 10
    
    # Random initial states
    state1 = normalize!(randn(ComplexF64, dim))
    state2 = normalize!(randn(ComplexF64, dim))
    
    # Random parameters
    t_vals = randn(Float64, num_params) .* 0.1
    
    # Mock some sparse operator structure
    # We will just randomly place some non-zeros
    num_nonzeros = 150
    rows = rand(1:dim, num_nonzeros)
    cols = rand(1:dim, num_nonzeros)
    signs = randn(ComplexF64, num_nonzeros)
    param_index_map = rand(1:num_params, num_nonzeros)
    
    # Symmetries
    use_symmetry = false
    antihermitian = false
    parameter_mapping = nothing
    parity = nothing

    # Build the ops flat arrays the way ensure_operator_structure! does
    flat_rows = Int[]
    flat_cols = Int[]
    flat_vals = ComplexF64[]
    flat_params = Int[]

    # Simulate indices_by_param
    indices_by_param = [Int[] for _ in 1:num_params]
    for k in eachindex(param_index_map)
        push!(indices_by_param[param_index_map[k]], k)
    end

    for i in 1:num_params
        idx = indices_by_param[i]
        for j in idx
            r = rows[j]
            c = cols[j]
            s = signs[j]
            
            push!(flat_rows, r)
            push!(flat_cols, c)
            push!(flat_vals, s)
            push!(flat_params, i)
            
            push!(flat_rows, c)
            push!(flat_cols, r)
            if antihermitian
                push!(flat_vals, -conj(s))
            else
                push!(flat_vals, conj(s))
            end
            push!(flat_params, i)
        end
    end

    ops = Dict(
        :flat_rows => flat_rows,
        :flat_cols => flat_cols,
        :flat_vals => flat_vals,
        :flat_params => flat_params
    )

    println("\n--- Testing Zygote Exact Gradient ---")
    loss_exact, back_exact = Zygote.pullback(t -> zygote_loss(t, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian), t_vals)
    grad_exact = back_exact(1.0)[1]
    
    println("Zygote Exact Loss: $loss_exact")
    println("Zygote Exact Grad (first 5): $(grad_exact[1:min(5, end)])")
    
    println("\n--- Testing CPU Matrix-Free Adjoint Gradient ---")
    # Note: adjoint_loss expects state2, state1 swapped
    loss_cpu, back_cpu = Zygote.pullback(t -> adjoint_loss(t, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, !use_symmetry, antihermitian), t_vals)
    grad_cpu = back_cpu(1.0)[1]
    
    println("CPU Matrix-Free Loss: $loss_cpu")
    println("CPU Matrix-Free Grad (first 5): $(grad_cpu[1:min(5, end)])")
    
    println("\n--- CPU Error ---")
    # Note: the adjoint_loss_pullback includes L2 regularization: + 1e-3 * t_vals[i]
    # zygote_loss does not have this regularization in the code.
    # So we should subtract the regularization from grad_cpu to compare
    grad_cpu_unreg = grad_cpu .- 1e-3 .* t_vals
    
    err_cpu = norm(grad_cpu_unreg .- grad_exact)
    println("L2 Error between CPU and Exact: $err_cpu")
    
    if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
        println("\n--- Testing GPU Matrix-Free Adjoint Gradient ---")
        ops_gpu = Dict(
            :flat_rows => CUDA.CuArray(flat_rows),
            :flat_cols => CUDA.CuArray(flat_cols),
            :flat_vals => CUDA.CuArray(flat_vals),
            :flat_params => CUDA.CuArray(flat_params)
        )
        state1_gpu = CUDA.CuArray(state1)
        state2_gpu = CUDA.CuArray(state2)
        
        loss_gpu, back_gpu = Zygote.pullback(t -> gpu_adjoint_loss(t, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2_gpu, state1_gpu, nothing, !use_symmetry, antihermitian), t_vals)
        grad_gpu = back_gpu(1.0)[1]
        
        println("GPU Matrix-Free Loss: $loss_gpu")
        println("GPU Matrix-Free Grad (first 5): $(grad_gpu[1:min(5, end)])")
        
        println("\n--- GPU Error ---")
        grad_gpu_unreg = grad_gpu .- 1e-3 .* t_vals
        err_gpu = norm(grad_gpu_unreg .- grad_exact)
        println("L2 Error between GPU and Exact: $err_gpu")
        
        err_gpu_cpu = norm(grad_gpu .- grad_cpu)
        println("L2 Error between GPU and CPU: $err_gpu_cpu")
    end

    println("\nDone!")
end

test_matrix_free_gradients()
