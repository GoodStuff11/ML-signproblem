#=
test_checkpoint_offload_benchmark_4x3.jl

Benchmark script comparing Trotter optimization performance on N=(5, 4)_4x3 across:
1. Pure Single-Core CPU (use_gpu=false, 1 thread)
2. Full GPU (All state checkpoints on GPU VRAM)
3. GPU with Host RAM Checkpoints (State checkpoints streamed to CPU RAM)

This script does NOT write or modify any data files.
=#

ENV["JULIA_CUDA_USE_COMPAT"] = "true"
using CUDA
using HDF5
using Lattices
using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Random

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function measure_vram_peak()
    CUDA.reclaim()
    GC.gc(true)
    free_mem, total_mem = CUDA.memory_info()
    return (total_mem - free_mem) / (1024^2) # in MiB
end

# 1. Pure CPU (Single Core)
function run_pure_cpu(A, gates, tau_terms, v_ref, basis_sector, N_sites; antihermitian=true, datatype=Float64)
    P = 1
    num_gates = length(gates)
    M = P * num_gates
    
    t_fwd = @elapsed begin
        phis = Trotter.TrotterOptimization.apply_unitary_checkpoints(
            A, gates, v_ref, basis_sector, N_sites, P;
            antihermitian=antihermitian, use_gpu=false, datatype=datatype
        )
    end

    grad_A = Vector{Float64}(undef, M)
    t_bwd = @elapsed begin
        init_adj = copy(phis[end])
        grad_A = Trotter.TrotterOptimization.backward_adjoint_propagation(
            A, gates, tau_terms, phis, init_adj, basis_sector, N_sites, P;
            antihermitian=antihermitian, use_gpu=false, datatype=datatype
        )
    end

    return t_fwd, t_bwd, grad_A
end

# 2. Standard Full-GPU Forward and Backward
function run_full_gpu(A, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=Float32)
    P = 1
    num_gates = length(gates)
    M = P * num_gates
    
    # Forward pass: keep all checkpoints on GPU
    t_fwd = @elapsed begin
        ref_dev = CUDA.CuArray(v_ref)
        phis = Vector{typeof(ref_dev)}(undef, M + 1)
        phis[1] = copy(ref_dev)
        curr = 1
        for l in 1:P
            for param_idx in 1:num_gates
                a = Float64(A[curr])
                phis[curr+1] = similar(ref_dev)
                Trotter.TrotterOptimization.gpu_apply_gate_exp!(phis[curr+1], phis[curr], gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                curr += 1
            end
        end
        CUDA.synchronize()
    end

    # Backward pass: read directly from GPU checkpoints
    grad_A = Vector{Float64}(undef, M)
    t_bwd = @elapsed begin
        adj_curr = copy(phis[end])
        ref_sample = phis[1]
        adj_next = similar(ref_sample)

        curr = M
        for l in P:-1:1
            for param_idx in num_gates:-1:1
                a = Float64(A[curr])
                tau_mat = Trotter.TrotterOptimization._get_gpu_tau_mat(gpu_ops, param_idx)
                mul!(gpu_ops.w1, tau_mat, phis[curr+1])
                dot_val = dot(adj_curr, gpu_ops.w1)
                grad_A[curr] = antihermitian ? -real(dot_val) : imag(dot_val)

                Trotter.TrotterOptimization.gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true)
                adj_curr, adj_next = adj_next, adj_curr
                curr -= 1
            end
        end
        CUDA.synchronize()
    end

    return t_fwd, t_bwd, grad_A
end

# 3. Checkpoints offloaded to Host RAM (CPU memory)
function run_host_checkpoints(A, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=Float32)
    P = 1
    num_gates = length(gates)
    M = P * num_gates
    d = length(v_ref)

    # Forward pass: GPU computes evolution, copies intermediate state vector to CPU host memory
    phis_host = Vector{Vector{datatype}}(undef, M + 1)
    t_fwd = @elapsed begin
        v_curr = CUDA.CuArray(v_ref)
        v_next = CUDA.similar(v_curr)
        phis_host[1] = Array(v_curr)
        
        curr = 1
        for l in 1:P
            for param_idx in 1:num_gates
                a = Float64(A[curr])
                Trotter.TrotterOptimization.gpu_apply_gate_exp!(v_next, v_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=false)
                v_curr, v_next = v_next, v_curr
                phis_host[curr+1] = Array(v_curr) # Transfer to CPU RAM
                curr += 1
            end
        end
        CUDA.synchronize()
    end

    # Backward pass: Stream only 1 checkpoint from Host RAM to GPU at a time into reusable buffer
    grad_A = Vector{Float64}(undef, M)
    t_bwd = @elapsed begin
        adj_curr = copy(v_curr)
        adj_next = CUDA.similar(adj_curr)
        phi_step_gpu = CUDA.similar(v_curr)

        curr = M
        for l in P:-1:1
            for param_idx in num_gates:-1:1
                a = Float64(A[curr])
                copyto!(phi_step_gpu, phis_host[curr+1]) # Host -> GPU single-vector transfer
                
                tau_mat = Trotter.TrotterOptimization._get_gpu_tau_mat(gpu_ops, param_idx)
                mul!(gpu_ops.w1, tau_mat, phi_step_gpu)
                dot_val = dot(adj_curr, gpu_ops.w1)
                grad_A[curr] = antihermitian ? -real(dot_val) : imag(dot_val)

                Trotter.TrotterOptimization.gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, param_idx, a; antihermitian=antihermitian, inverse=true)
                adj_curr, adj_next = adj_next, adj_curr
                curr -= 1
            end
        end
        CUDA.synchronize()
    end

    return t_fwd, t_bwd, grad_A
end

function benchmark_precision(dtype::Type, label::String, folder, gates, tau_terms, basis_sector, N_sites, state_vecs, A_rand)
    println("\n" * "="^85)
    println(">>> BENCHMARKING PRECISION: $label ($dtype)")
    println("="^85)

    d = length(basis_sector)
    num_gates = length(gates)

    ref_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[1, :])
    v_ref = dtype.(ref_real)

    # Initialize GPU operators
    CUDA.reclaim()
    GC.gc(true)
    gpu_ops = Trotter.TrotterOptimization.get_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=dtype)

    # Warmup all modes
    println("Performing warmups...")
    run_pure_cpu(A_rand, gates, tau_terms, v_ref, basis_sector, N_sites; antihermitian=true, datatype=dtype)
    run_full_gpu(A_rand, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=dtype)
    run_host_checkpoints(A_rand, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=dtype)

    num_iters = 5

    # 1. Benchmark Pure Single-Core CPU
    println("\n1. Running Pure CPU (1 core, use_gpu=false)...")
    fwd_times_cpu = Float64[]
    bwd_times_cpu = Float64[]
    grad_cpu = nothing
    for _ in 1:num_iters
        tf, tb, g = run_pure_cpu(A_rand, gates, tau_terms, v_ref, basis_sector, N_sites; antihermitian=true, datatype=dtype)
        push!(fwd_times_cpu, tf)
        push!(bwd_times_cpu, tb)
        grad_cpu = g
    end

    # 2. Benchmark Full GPU
    println("2. Running Full GPU (all $num_gates checkpoints on VRAM)...")
    CUDA.reclaim()
    fwd_times_full = Float64[]
    bwd_times_full = Float64[]
    grad_full = nothing
    for _ in 1:num_iters
        tf, tb, g = run_full_gpu(A_rand, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=dtype)
        push!(fwd_times_full, tf)
        push!(bwd_times_full, tb)
        grad_full = g
    end

    # 3. Benchmark Host RAM Checkpoints
    println("3. Running Host RAM Checkpoints (checkpoints streamed to CPU RAM)...")
    CUDA.reclaim()
    fwd_times_host = Float64[]
    bwd_times_host = Float64[]
    grad_host = nothing
    for _ in 1:num_iters
        tf, tb, g = run_host_checkpoints(A_rand, gates, v_ref, basis_sector, N_sites, gpu_ops; antihermitian=true, datatype=dtype)
        push!(fwd_times_host, tf)
        push!(bwd_times_host, tb)
        grad_host = g
    end

    # Verification of gradient equivalence
    diff_gpu_vs_cpu = norm(grad_full .- grad_cpu) / norm(grad_cpu)
    diff_host_vs_gpu = norm(grad_host .- grad_full) / norm(grad_full)
    println("\nGradient relative diff (Full GPU vs CPU):   ", @sprintf("%.2e", diff_gpu_vs_cpu))
    println("Gradient relative diff (Host Offload vs GPU): ", @sprintf("%.2e", diff_host_vs_gpu))

    avg_fwd_cpu = mean(fwd_times_cpu) * 1000
    avg_bwd_cpu = mean(bwd_times_cpu) * 1000
    avg_tot_cpu = avg_fwd_cpu + avg_bwd_cpu

    avg_fwd_full = mean(fwd_times_full) * 1000
    avg_bwd_full = mean(bwd_times_full) * 1000
    avg_tot_full = avg_fwd_full + avg_bwd_full

    avg_fwd_host = mean(fwd_times_host) * 1000
    avg_bwd_host = mean(bwd_times_host) * 1000
    avg_tot_host = avg_fwd_host + avg_bwd_host

    speedup_full = avg_tot_cpu / avg_tot_full
    speedup_host = avg_tot_cpu / avg_tot_host

    vector_size_mb = (d * sizeof(dtype)) / (1024^2)
    total_checkpoint_vram_mb = (num_gates + 1) * vector_size_mb

    println("\n--- Summary for $label ---")
    @printf("Hilbert dimension (d):               %d\n", d)
    @printf("Number of gates (M):                 %d\n", num_gates)
    @printf("Single state vector size:            %.3f MiB\n", vector_size_mb)
    @printf("Total Checkpoint VRAM (Full GPU):    %.2f MiB\n", total_checkpoint_vram_mb)
    @printf("Total Checkpoint VRAM (Host RAM):    %.3f MiB\n", vector_size_mb)
    println("-"^95)
    @printf("%-26s | %-12s | %-12s | %-14s | %-12s\n", "Mode", "Forward (ms)", "Backward (ms)", "Total Step (ms)", "Speedup vs CPU")
    println("-"^95)
    @printf("%-26s | %10.2f ms | %10.2f ms | %12.2f ms | %12.2fx (baseline)\n", "1. Pure CPU (1 core)", avg_fwd_cpu, avg_bwd_cpu, avg_tot_cpu, 1.0)
    @printf("%-26s | %10.2f ms | %10.2f ms | %12.2f ms | %12.2fx\n", "2. Full GPU (All VRAM)", avg_fwd_full, avg_bwd_full, avg_tot_full, speedup_full)
    @printf("%-26s | %10.2f ms | %10.2f ms | %12.2f ms | %12.2fx\n", "3. Host RAM Offload", avg_fwd_host, avg_bwd_host, avg_tot_host, speedup_host)
    println("-"^95)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_checkpoint_offload_benchmark_4x3")
    with_logging(log_path) do
        println("================================================================================")
        println("CHECKPOINT OFFLOADING BENCHMARK (4x3 Lattice)")
        println("Comparing Pure Single-Core CPU vs Full GPU vs Host RAM Checkpoint Offload")
        println("================================================================================")
        println("CPU Threads: $(Threads.nthreads())")
        println("GPU Device: $(CUDA.name(CUDA.device()))")
        println("Total GPU VRAM: $(round(CUDA.total_memory() / (1024^3), digits=2)) GiB")

        folder = data_folder("N=(5, 4)_4x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)

        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)
        num_gates = length(gates)

        Random.seed!(1234)
        A_rand = randn(num_gates)

        benchmark_precision(Float32, "Float32", folder, gates, tau_terms, basis_sector, N_sites, state_vecs, A_rand)
        benchmark_precision(Float64, "Float64", folder, gates, tau_terms, basis_sector, N_sites, state_vecs, A_rand)

        println("\nBenchmark completed successfully without saving or modifying any files.")
        return 0
    end
end
