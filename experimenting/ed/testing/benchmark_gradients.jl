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

include("../ed_objects.jl")
include("../ed_functions.jl")
include("../ed_optimization.jl")

macro peak_mem(ex)
    quote
        done = Threads.Atomic{Bool}(false)
        peak_cpu = Threads.Atomic{Int64}(0)
        peak_gpu = Threads.Atomic{Int64}(0)
        
        # GC first to get a baseline
        GC.gc(true)
        if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
            CUDA.reclaim()
        end
        
        start_cpu = try Base.gc_live_bytes() catch; Sys.maxrss() end
        start_gpu = 0
        if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
            try start_gpu = CUDA.alloc_bytes() catch end
        end
        
        t = Threads.@spawn begin
            while !done[]
                cpu_mem = try Base.gc_live_bytes() catch; Sys.maxrss() end
                Threads.atomic_max!(peak_cpu, Int64(cpu_mem))
                
                if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
                    try
                        gpu_mem = CUDA.alloc_bytes()
                        Threads.atomic_max!(peak_gpu, Int64(gpu_mem))
                    catch
                    end
                end
                sleep(0.001)
            end
        end
        
        local val = $(esc(ex))
        done[] = true
        wait(t)
        
        cpu_diff = max(0, peak_cpu[] - start_cpu)
        gpu_diff = max(0, peak_gpu[] - start_gpu)
        println("  -> Peak CPU Memory Increase: ", Base.format_bytes(cpu_diff))
        if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
            println("  -> Peak GPU Memory Increase: ", Base.format_bytes(gpu_diff))
        end
        val
    end
end

function benchmark_loss_gradients()
    println("Setting up test system for gradient benchmarking...")
    Random.seed!(42)
    
    dim = 1000000
    num_params = 1000
    
    state1 = normalize!(randn(ComplexF64, dim))
    state2 = normalize!(randn(ComplexF64, dim))
    
    t_vals = randn(Float64, num_params) .* 0.1
    
    num_nonzeros = 150
    rows = rand(1:dim, num_nonzeros)
    cols = rand(1:dim, num_nonzeros)
    signs = randn(ComplexF64, num_nonzeros)
    param_index_map = rand(1:num_params, num_nonzeros)
    
    use_symmetry = false
    antihermitian = false
    parameter_mapping = nothing
    parity = nothing

    # Simulate indices_by_param
    indices_by_param = [Int[] for _ in 1:num_params]
    for k in eachindex(param_index_map)
        push!(indices_by_param[param_index_map[k]], k)
    end

    # Build ops array as done in ensure_operator_structure!
    ops = []
    for i in 1:num_params
        idx = indices_by_param[i]
        _rows = rows[idx]
        _cols = cols[idx]
        _signs = signs[idx]
        
        if antihermitian
            S = make_antihermitian(sparse(_rows, _cols, _signs, dim, dim))
        else
            S = make_hermitian(sparse(_rows, _cols, _signs, dim, dim))
        end
        I, J, V = findnz(S)
        push!(ops, (I, J, V))
    end

    println("\n--- Warming up functions for accurate benchmarking ---")
    # loss_zygote, back_zygote = Zygote.pullback(t -> zygote_loss(t, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian), t_vals)
    # _ = back_zygote(1.0)[1]
    
    loss_cpu, back_cpu = Zygote.pullback(t -> adjoint_loss(t, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, !use_symmetry, antihermitian), t_vals)
    _ = back_cpu(1.0)[1]
    
    if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
        ops_gpu = ops
        state1_gpu = CUDA.CuArray(state1)
        state2_gpu = CUDA.CuArray(state2)
        loss_gpu, back_gpu = Zygote.pullback(t -> gpu_adjoint_loss(t, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2_gpu, state1_gpu, nothing, !use_symmetry, antihermitian), t_vals)
        _ = back_gpu(1.0)[1]
    else
        println("CUDA not available, skipping GPU adjoint benchmark.")
    end
    println("Warmup complete.")

    # println("\n--- 1. Zygote Exact Gradient ---")
    # local loss_zygote, grad_zygote
    # @time begin
    #     @peak_mem begin
    #         loss_zygote, back_zygote = Zygote.pullback(t -> zygote_loss(t, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state1, state2, use_symmetry, antihermitian), t_vals)
    #         grad_zygote = back_zygote(1.0)[1]
    #     end
    # end
    
    # println("Zygote Loss: $loss_zygote")
    # println("Zygote Grad (first 5): $(grad_zygote[1:min(5, end)])")
    
    println("\n--- 2. CPU Adjoint Gradient (Quadrature) ---")
    local loss_cpu, grad_cpu
    @time begin
        @peak_mem begin
            loss_cpu, back_cpu = Zygote.pullback(t -> adjoint_loss(t, ops, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2, state1, nothing, !use_symmetry, antihermitian), t_vals)
            grad_cpu = back_cpu(1.0)[1]
        end
    end
    
    # Remove the 1e-3 regularization added in adjoint_loss_pullback for direct comparison
    grad_cpu_unreg = grad_cpu .- 1e-3 .* t_vals
    
    println("CPU Adjoint Loss: $loss_cpu")
    println("CPU Adjoint Grad (first 5): $(grad_cpu_unreg[1:min(5, end)])")
    # println("Diff (L2 Error vs Zygote): $(norm(grad_cpu_unreg .- grad_zygote))")
    
    if isdefined(Main, :CUDA) && CUDA.has_cuda_gpu()
        println("\n--- 3. GPU Adjoint Gradient (Quadrature) ---")
        ops_gpu = ops
        state1_gpu = CUDA.CuArray(state1)
        state2_gpu = CUDA.CuArray(state2)
        
        local loss_gpu, grad_gpu
        @time begin
            @peak_mem begin
                loss_gpu, back_gpu = Zygote.pullback(t -> gpu_adjoint_loss(t, ops_gpu, rows, cols, signs, param_index_map, parameter_mapping, parity, dim, state2_gpu, state1_gpu, nothing, !use_symmetry, antihermitian), t_vals)
                grad_gpu = back_gpu(1.0)[1]
            end
        end
        
        grad_gpu_unreg = grad_gpu .- 1e-3 .* t_vals
        
        println("GPU Adjoint Loss: $loss_gpu")
        println("GPU Adjoint Grad (first 5): $(grad_gpu_unreg[1:min(5, end)])")
        # println("Diff (L2 Error vs Zygote): $(norm(grad_gpu_unreg .- grad_zygote))")
        println("Diff (L2 Error vs CPU Adjoint): $(norm(grad_gpu .- grad_cpu))")
    end

    println("\nBenchmarking complete!")
end

benchmark_loss_gradients()
