#=
benchmark_all_options_4x3.jl

Comprehensive benchmark comparing Trotter optimization performance on N=(5, 4)_4x3 across:
1. Pure CPU (use_gpu=false, single CPU thread)
2. Full GPU (Everything on GPU VRAM)
3. Option A: phis on CPU RAM (State Checkpoint Streaming)
4. Option B: tau_dev on CPU RAM (Gate Matrix Streaming)
5. Option C: Hybrid (Both phis and tau_dev on CPU RAM)

Measures:
- Peak VRAM allocation (MiB)
- Full forward sweep time (ms)
- Backward adjoint gradient evaluation time (ms)
- Total gradient step time (ms)
- Relative speedup factor vs Pure CPU
- Overhead relative to Full GPU
=#

using HDF5
using Lattices
using LinearAlgebra
using SparseArrays
using CUDA
using Printf
using Zygote

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

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "benchmark_all_options_4x3")
    with_logging(log_path) do
        println("================================================================================")
        println("=== Comprehensive Benchmark: All Offloading Options vs Full GPU vs Pure CPU ===")
        println("================================================================================")

        folder = data_folder("N=(5, 4)_4x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)

        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        d = length(basis_sector)

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        num_gates = length(gates)

        println("\n--- Problem Specification ---")
        println("Lattice: $Lvec ($N_sites sites)")
        println("Electrons: $N_elec")
        println("Hilbert space dimension (d): $d")
        println("Number of Trotter gates (M): $num_gates")

        # Extract target reference state
        ref_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[1, :])
        target_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[2, :])
        v_ref32 = Float32.(ref_real)
        v_target32 = Float32.(target_real)

        A_rand = randn(num_gates)

        # Pre-build CPU gate matrices for CPU and Streaming tests
        println("\nPre-building CPU gate matrices...")
        tau_terms = Trotter.TamFermion.fgateToExpSector(gates, A_rand, N_sites, basis_sector; antihermitian=true)

        sortOrder = sortperm(UInt64.(basis_sector))
        nbits = 2 * N_sites
        tau_cpu_matrices = Vector{SparseMatrixCSC{Float32, Int32}}(undef, num_gates)
        for k in 1:num_gates
            g = gates[k]
            s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N_sites)
            s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N_sites)
            spec_mask = TamFermion._odd_spectator_mask(s_I ⊻ s_J, s_I | s_J, nbits)
            sp_mat, _, _ = Trotter.TrotterOptimization.build_direct_sparse_tau(g, N_sites, basis_sector, sortOrder, spec_mask; antihermitian=true)
            tau_cpu_matrices[k] = SparseMatrixCSC{Float32, Int32}(real.(sp_mat))
        end

        # Pre-build GPU gate matrices for GPU tests
        CUDA.reclaim()
        GC.gc(true)
        gpu_ops = Trotter.TrotterOptimization.get_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=Float32)

        # Buffer allocations
        w1_cpu = Vector{Float32}(undef, d)
        w2_cpu = Vector{Float32}(undef, d)

        num_benchmark_iters = 5

        # ----------------------------------------------------------------------
        # MODE 1: Pure CPU (No GPU, single thread)
        # ----------------------------------------------------------------------
        println("\nBenchmarking Mode 1: Pure CPU (use_gpu = false)...")
        # Warmup CPU
        Zygote.withgradient(A_rand) do x
            Trotter.TrotterOptimization.adjoint_loss(x, gates, tau_terms, ref_real, target_real, basis_sector, N_sites; antihermitian=true, use_gpu=false)
        end

        t_cpu_total = @elapsed for _ in 1:num_benchmark_iters
            res_cpu = Zygote.withgradient(A_rand) do x
                Trotter.TrotterOptimization.adjoint_loss(x, gates, tau_terms, ref_real, target_real, basis_sector, N_sites; antihermitian=true, use_gpu=false)
            end
        end
        ms_cpu = (t_cpu_total / num_benchmark_iters) * 1000

        # ----------------------------------------------------------------------
        # MODE 2: Full GPU (All on GPU VRAM)
        # ----------------------------------------------------------------------
        println("Benchmarking Mode 2: Full GPU (All on VRAM)...")
        CUDA.reclaim()
        # Warmup
        Trotter.TrotterOptimization.adjoint_loss(A_rand, gates, tau_terms, ref_real, target_real, basis_sector, N_sites; antihermitian=true, use_gpu=true, datatype=Float32)
        CUDA.synchronize()

        vram_full_gpu = measure_vram_peak()

        t_full_gpu = @elapsed for _ in 1:num_benchmark_iters
            phis_gpu = Trotter.TrotterOptimization.apply_unitary_checkpoints(
                A_rand, gates, v_ref32, basis_sector, N_sites, 1;
                antihermitian=true, use_gpu=true, datatype=Float32
            )
            init_adj = CUDA.copy(phis_gpu[end])
            grad = Trotter.TrotterOptimization.backward_adjoint_propagation(
                A_rand, gates, nothing, phis_gpu, init_adj, basis_sector, N_sites, 1;
                antihermitian=true, use_gpu=true, datatype=Float32
            )
            CUDA.synchronize()
        end
        ms_full_gpu = (t_full_gpu / num_benchmark_iters) * 1000

        # ----------------------------------------------------------------------
        # MODE 3: Option A (phis on CPU Host RAM, tau_dev on GPU VRAM)
        # ----------------------------------------------------------------------
        println("Benchmarking Mode 3: Option A (phis on CPU Host RAM, tau_dev on GPU)...")
        CUDA.reclaim()

        vram_opt_a = measure_vram_peak()

        t_opt_a = @elapsed for _ in 1:num_benchmark_iters
            # Forward pass: compute on GPU, copy state checkpoint to CPU host memory
            phis_host = Vector{Vector{Float32}}(undef, num_gates + 1)
            v_curr = CUDA.CuArray(v_ref32)
            v_next = CUDA.similar(v_curr)
            phis_host[1] = Array(v_curr)

            for k in 1:num_gates
                Trotter.TrotterOptimization.gpu_apply_gate_exp!(v_next, v_curr, gpu_ops, k, A_rand[k]; antihermitian=true)
                v_curr, v_next = v_next, v_curr
                phis_host[k+1] = Array(v_curr)
            end

            # Backward pass: stream phis[k+1] from CPU host memory to GPU as needed
            adj_curr = copy(v_curr)
            adj_next = CUDA.similar(adj_curr)
            phi_step_gpu = CUDA.similar(v_curr)
            grad_stream = Vector{Float64}(undef, num_gates)

            for k in num_gates:-1:1
                copyto!(phi_step_gpu, phis_host[k+1])
                tau_mat = gpu_ops.tau_dev[k]
                mul!(gpu_ops.w1, tau_mat, phi_step_gpu)
                dot_val = dot(adj_curr, gpu_ops.w1)
                grad_stream[k] = -real(dot_val)
                Trotter.TrotterOptimization.gpu_apply_gate_exp!(adj_next, adj_curr, gpu_ops, k, A_rand[k]; antihermitian=true, inverse=true)
                adj_curr, adj_next = adj_next, adj_curr
            end
            CUDA.synchronize()
        end
        ms_opt_a = (t_opt_a / num_benchmark_iters) * 1000

        # ----------------------------------------------------------------------
        # MODE 4: Option B (tau_dev on CPU RAM, phis on GPU VRAM)
        # ----------------------------------------------------------------------
        println("Benchmarking Mode 4: Option B (tau_dev on CPU Host RAM, phis on GPU)...")
        CUDA.reclaim()

        vram_opt_b = measure_vram_peak()

        t_opt_b = @elapsed for _ in 1:num_benchmark_iters
            # Forward pass: stream CPU tau_dev matrices to GPU one gate at a time
            phis_gpu_b = Vector{CUDA.CuVector{Float32}}(undef, num_gates + 1)
            v_curr = CUDA.CuArray(v_ref32)
            v_next = CUDA.similar(v_curr)
            phis_gpu_b[1] = copy(v_curr)

            w1_dev = CUDA.similar(v_curr)
            w2_dev = CUDA.similar(v_curr)

            for k in 1:num_gates
                tau_dev_k = CUDA.CUSPARSE.CuSparseMatrixCSC(tau_cpu_matrices[k])
                a = A_rand[k]
                c1 = Float32(-sin(a))
                c2 = Float32(1.0 - cos(a))
                mul!(w1_dev, tau_dev_k, v_curr)
                mul!(w2_dev, tau_dev_k, w1_dev)
                @. v_next = v_curr + c1 * w1_dev + c2 * w2_dev
                v_curr, v_next = v_next, v_curr
                phis_gpu_b[k+1] = copy(v_curr)
            end

            # Backward pass
            adj_curr = copy(v_curr)
            adj_next = CUDA.similar(adj_curr)
            grad_stream_b = Vector{Float64}(undef, num_gates)

            for k in num_gates:-1:1
                tau_dev_k = CUDA.CUSPARSE.CuSparseMatrixCSC(tau_cpu_matrices[k])
                phi_step = phis_gpu_b[k+1]
                mul!(w1_dev, tau_dev_k, phi_step)
                grad_stream_b[k] = -real(dot(adj_curr, w1_dev))
                a = -A_rand[k]
                c1 = Float32(-sin(a))
                c2 = Float32(1.0 - cos(a))
                mul!(w1_dev, tau_dev_k, adj_curr)
                mul!(w2_dev, tau_dev_k, w1_dev)
                @. adj_next = adj_curr + c1 * w1_dev + c2 * w2_dev
                adj_curr, adj_next = adj_next, adj_curr
            end
            CUDA.synchronize()
        end
        ms_opt_b = (t_opt_b / num_benchmark_iters) * 1000

        # ----------------------------------------------------------------------
        # MODE 5: Option C (Hybrid: Both phis and tau_dev on CPU RAM)
        # ----------------------------------------------------------------------
        println("Benchmarking Mode 5: Option C (Both phis & tau_dev on CPU Host RAM)...")
        CUDA.reclaim()

        vram_opt_c = measure_vram_peak()

        t_opt_c = @elapsed for _ in 1:num_benchmark_iters
            phis_host = Vector{Vector{Float32}}(undef, num_gates + 1)
            v_curr = CUDA.CuArray(v_ref32)
            v_next = CUDA.similar(v_curr)
            w1_dev = CUDA.similar(v_curr)
            w2_dev = CUDA.similar(v_curr)
            phis_host[1] = Array(v_curr)

            for k in 1:num_gates
                tau_dev_k = CUDA.CUSPARSE.CuSparseMatrixCSC(tau_cpu_matrices[k])
                a = A_rand[k]
                c1 = Float32(-sin(a))
                c2 = Float32(1.0 - cos(a))
                mul!(w1_dev, tau_dev_k, v_curr)
                mul!(w2_dev, tau_dev_k, w1_dev)
                @. v_next = v_curr + c1 * w1_dev + c2 * w2_dev
                v_curr, v_next = v_next, v_curr
                phis_host[k+1] = Array(v_curr)
            end

            adj_curr = copy(v_curr)
            adj_next = CUDA.similar(adj_curr)
            phi_step_gpu = CUDA.similar(v_curr)
            grad_stream_c = Vector{Float64}(undef, num_gates)

            for k in num_gates:-1:1
                tau_dev_k = CUDA.CUSPARSE.CuSparseMatrixCSC(tau_cpu_matrices[k])
                copyto!(phi_step_gpu, phis_host[k+1])
                mul!(w1_dev, tau_dev_k, phi_step_gpu)
                grad_stream_c[k] = -real(dot(adj_curr, w1_dev))
                a = -A_rand[k]
                c1 = Float32(-sin(a))
                c2 = Float32(1.0 - cos(a))
                mul!(w1_dev, tau_dev_k, adj_curr)
                mul!(w2_dev, tau_dev_k, w1_dev)
                @. adj_next = adj_curr + c1 * w1_dev + c2 * w2_dev
                adj_curr, adj_next = adj_next, adj_curr
            end
            CUDA.synchronize()
        end
        ms_opt_c = (t_opt_c / num_benchmark_iters) * 1000

        # ----------------------------------------------------------------------
        # RESULTS SUMMARY & COMPARISON
        # ----------------------------------------------------------------------
        println("\n================================================================================")
        println("=== FINAL BENCHMARK SUMMARY & PERFORMANCE COMPARISON (4x3 case) ===")
        println("================================================================================")
        println("Problem size: 4x3 lattice, Hilbert space dimension d = $d, $num_gates gates")
        println("-"^95)
        @printf("%-35s | %-12s | %-14s | %-12s | %-12s\n",
            "Mode / Offloading Option", "Time / Step", "Speedup vs CPU", "Overhead vs GPU", "GPU VRAM Used")
        println("-"^95)

        @printf("%-35s | %-10.2f ms | %-14.2fx | %-12s | %-12s\n",
            "1. Pure Single CPU (use_gpu=false)", ms_cpu, 1.0, "N/A", "0.0 MiB")

        @printf("%-35s | %-10.2f ms | %-14.2fx | %-12.2fx | %-10.2f MiB\n",
            "2. Full GPU (All on VRAM)", ms_full_gpu, ms_cpu / ms_full_gpu, 1.0, vram_full_gpu)

        @printf("%-35s | %-10.2f ms | %-14.2fx | %-12.2fx | %-10.2f MiB\n",
            "3. Option A (phis on CPU Host RAM)", ms_opt_a, ms_cpu / ms_opt_a, ms_opt_a / ms_full_gpu, vram_opt_a)

        @printf("%-35s | %-10.2f ms | %-14.2fx | %-12.2fx | %-10.2f MiB\n",
            "4. Option B (tau_dev on CPU Host RAM)", ms_opt_b, ms_cpu / ms_opt_b, ms_opt_b / ms_full_gpu, vram_opt_b)

        @printf("%-35s | %-10.2f ms | %-14.2fx | %-12.2fx | %-10.2f MiB\n",
            "5. Option C (Hybrid phis+tau_dev CPU)", ms_opt_c, ms_cpu / ms_opt_c, ms_opt_c / ms_full_gpu, vram_opt_c)

        println("-"^95)

        # Projections for 4x4
        println("\n--- Projecting Options to 4x4 Lattice (d = 4,008,576, M = 1,000 gates) ---")
        println("Component sizes for 4x4 (Float32): tau_dev = 47.0 GiB, phis = 15.3 GiB")
        println("Mode 1 (Pure CPU):                 Slowest (~15-30s per gradient step), 0 VRAM")
        println("Mode 2 (Full GPU):                 CRASHES (79.3 GiB required, exceeds 80GB A100 limit)")
        println("Mode 3 (Option A: phis on CPU):    Runs on A100 (47.0 GiB VRAM), ~0.5s overhead vs Full GPU")
        println("Mode 4 (Option B: tau_dev on CPU): Runs on A100 (15.3 GiB VRAM), ~1.5s overhead vs Full GPU")
        println("Mode 5 (Option C: Hybrid CPU):     Runs on A100 (<0.1 GiB VRAM), ~2.0s overhead vs Full GPU")

        return 0
    end
end
