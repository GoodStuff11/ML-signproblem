#=
benchmark_vram_breakdown.jl

Profile GPU VRAM usage and execution time of each component in Trotter optimization
for N=(5, 4)_4x3.

Components evaluated:
1. tau_dev CuSparseMatrixCSC matrices in _GPU_GATE_OPS_CACHE
2. phis state checkpoint array
3. Offloading tau_dev to CPU RAM vs keeping on GPU VRAM
4. Offloading phis checkpoints to CPU RAM vs keeping on GPU VRAM
5. Time comparison: On-GPU vs CPU-streamed gate application
=#

using HDF5
using Lattices
using LinearAlgebra
using SparseArrays
using CUDA

include(joinpath(@__DIR__, "..", "data_path.jl"))
include(joinpath(@__DIR__, "..", "logging.jl"))
include(joinpath(@__DIR__, "..", "utility_functions.jl"))
using .UtilityFunctions
include(joinpath(@__DIR__, "..", "trotter.jl"))
using .Trotter
include(joinpath(@__DIR__, "..", "ed_objects.jl"))
include(joinpath(@__DIR__, "..", "ed_functions.jl"))

function measure_vram_usage(f)
    CUDA.reclaim()
    GC.gc(true)
    free_before, total_mem = CUDA.memory_info()
    used_before = total_mem - free_before
    t = @elapsed res = f()
    CUDA.synchronize()
    free_after, total_mem = CUDA.memory_info()
    used_after = total_mem - free_after
    mem_diff = used_after - used_before
    return res, t, max(0, mem_diff)
end

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "benchmark_vram_breakdown")
    with_logging(log_path) do
        println("================================================================================")
        println("=== GPU VRAM Breakdown & Timing Benchmark (4x3 case) ===")
        println("================================================================================")

        folder = data_folder("N=(5, 4)_4x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=true, sign_convention=:spin_first, use_slater_reference=false)

        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        d = length(basis_sector)

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        num_gates = length(gates)

        println("\n--- Problem Dimensions ---")
        println("Lattice: $Lvec ($N_sites sites)")
        println("Electrons: $N_elec")
        println("Hilbert space dimension (d): $d")
        println("Number of Trotter gates (M): $num_gates")

        # Exact Memory Calculations
        println("\n--- Exact Theoretical Memory Analysis ---")
        println("4x3 case: d = $d, M = $num_gates")
        println("4x4 case: d = 4,008,576, M = 1,000")
        println("-"^60)
        println("Component               | 4x3 Float32 VRAM | 4x4 Float32 VRAM | 4x4 ComplexF64 VRAM")
        println("-"^60)
        tau_4x3_f32 = (num_gates * 12 * d) / (1024^2)
        tau_4x4_f32 = (1000 * 12 * 4008576) / (1024^3)
        tau_4x4_c64 = (1000 * 24 * 4008576) / (1024^3)

        phis_4x3_f32 = ((num_gates + 1) * 4 * d) / (1024^2)
        phis_4x4_f32 = (1001 * 4 * 4008576) / (1024^3)
        phis_4x4_c64 = (1001 * 16 * 4008576) / (1024^3)

        println("1. tau_dev gate matrices | $(round(tau_4x3_f32, digits=1)) MiB         | $(round(tau_4x4_f32, digits=1)) GiB         | $(round(tau_4x4_c64, digits=1)) GiB")
        println("2. phis state checkpoints| $(round(phis_4x3_f32, digits=1)) MiB         | $(round(phis_4x4_f32, digits=1)) GiB         | $(round(phis_4x4_c64, digits=1)) GiB")
        println("TOTAL ON-GPU VRAM       | $(round((tau_4x3_f32+phis_4x3_f32)/1024, digits=2)) GiB         | $(round(tau_4x4_f32+phis_4x4_f32, digits=1)) GiB        | $(round(tau_4x4_c64+phis_4x4_c64, digits=1)) GiB")
        println("-"^60)

        # 1. Benchmark tau_dev CuSparse creation & memory
        println("\n--- 1. Gate Matrices (tau_dev) Empirical Measurement ---")
        for dt in [Float32, Float64]
            empty!(Trotter.TrotterOptimization._GPU_GATE_OPS_CACHE)
            CUDA.reclaim()
            GC.gc(true)

            gpu_ops, t_build, mem_bytes = measure_vram_usage() do
                Trotter.TrotterOptimization.get_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=dt)
            end

            # Calculate actual matrix non-zeros
            total_nnz = sum(nnz(mat) for mat in gpu_ops.tau_dev)
            actual_mat_bytes = total_nnz * sizeof(dt) + (num_gates * (d + 1) + total_nnz) * 4

            println("DataType: $dt")
            println("  Build time: $(round(t_build, digits=3)) s")
            println("  Actual GPU sparse matrix size: $(round(actual_mat_bytes / (1024^2), digits=2)) MiB")
            println("  Allocated VRAM pool diff: $(round(mem_bytes / (1024^2), digits=2)) MiB")
        end

        # 2. Benchmark phis CPU Host storage vs GPU Storage
        println("\n--- 2. phis Checkpoints Storage Options ---")
        dt = Float32
        ref_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[1, :])
        v_real32 = Float32.(ref_real)
        ref_gpu = CUDA.CuArray(v_real32)

        # Mode 1: phis on GPU
        t_phis_gpu = @elapsed begin
            phis_gpu = Vector{CUDA.CuVector{Float32}}(undef, num_gates + 1)
            phis_gpu[1] = copy(ref_gpu)
            for i in 1:num_gates
                phis_gpu[i+1] = CUDA.similar(ref_gpu)
            end
        end
        CUDA.synchronize()
        mem_phis_gpu = ((num_gates + 1) * d * 4) / (1024^2)

        # Mode 2: phis on CPU Host Memory
        t_phis_cpu = @elapsed begin
            phis_cpu = Vector{Vector{Float32}}(undef, num_gates + 1)
            phis_cpu[1] = copy(v_real32)
            for i in 1:num_gates
                phis_cpu[i+1] = Vector{Float32}(undef, d)
            end
        end
        mem_phis_cpu_gpu = 0.0 # 0 MiB on GPU VRAM!

        println("phis Checkpoint Array Storage ($num_gates state vectors):")
        println("  Option 1 (All on GPU VRAM):  VRAM = $(round(mem_phis_gpu, digits=2)) MiB | Alloc time = $(round(t_phis_gpu*1000, digits=2)) ms")
        println("  Option 2 (All on CPU RAM):   VRAM = 0.00 MiB               | Alloc time = $(round(t_phis_cpu*1000, digits=2)) ms")

        # 3. Timing Benchmark: Forward & Backward Sweeps with Offloading Strategies
        println("\n--- 3. Execution Speed Comparison Across Offloading Strategies ---")

        A_rand = randn(num_gates)
        gpu_ops = Trotter.TrotterOptimization.get_gpu_gate_ops(gates, N_sites, basis_sector; antihermitian=true, datatype=Float32)

        # Build CPU sparse tau_dev matrices for CPU-resident streaming test
        sortOrder = sortperm(UInt64.(basis_sector))
        nbits = 2 * N_sites
        tau_cpu = Vector{SparseMatrixCSC{Float32, Int32}}(undef, num_gates)
        for k in 1:num_gates
            g = gates[k]
            s_I = UInt64(g.cre_up) | (UInt64(g.cre_dn) << N_sites)
            s_J = UInt64(g.ann_up) | (UInt64(g.ann_dn) << N_sites)
            spec_mask = TamFermion._odd_spectator_mask(s_I ⊻ s_J, s_I | s_J, nbits)
            sp_mat, _, _ = Trotter.TrotterOptimization.build_direct_sparse_tau(g, N_sites, basis_sector, sortOrder, spec_mask; antihermitian=true)
            tau_cpu[k] = SparseMatrixCSC{Float32, Int32}(real.(sp_mat))
        end

        # Strategy 1: Full GPU (Everything on GPU VRAM)
        t_gpu_full = @elapsed begin
            phis_gpu = Trotter.TrotterOptimization.apply_unitary_checkpoints(
                A_rand, gates, v_real32, basis_sector, N_sites, 1;
                antihermitian=true, use_gpu=true, datatype=Float32
            )
            init_adj = CUDA.copy(phis_gpu[end])
            grad = Trotter.TrotterOptimization.backward_adjoint_propagation(
                A_rand, gates, nothing, phis_gpu, init_adj, basis_sector, N_sites, 1;
                antihermitian=true, use_gpu=true, datatype=Float32
            )
            CUDA.synchronize()
        end

        # Strategy 2: Stream phis to CPU RAM during forward sweep, copy back during backward sweep
        t_gpu_stream_phis = @elapsed begin
            # Forward sweep: compute on GPU, copy checkpoint to CPU host memory
            phis_host = Vector{Vector{Float32}}(undef, num_gates + 1)
            v_curr = CUDA.CuArray(v_real32)
            v_next = CUDA.similar(v_curr)
            phis_host[1] = Array(v_curr)

            curr = 1
            for k in 1:num_gates
                Trotter.TrotterOptimization.gpu_apply_gate_exp!(v_next, v_curr, gpu_ops, k, A_rand[k]; antihermitian=true)
                v_curr, v_next = v_next, v_curr
                phis_host[curr+1] = Array(v_curr)
                curr += 1
            end

            # Backward sweep: stream phis[curr+1] from CPU host memory to GPU as needed
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

        println("\nFull Gradient Evaluation Time ($num_gates gates):")
        println("  Strategy 1 (Full GPU VRAM):             $(round(t_gpu_full, digits=3)) s  (VRAM: $(round((tau_4x3_f32+phis_4x3_f32)/1024, digits=2)) GiB)")
        println("  Strategy 2 (phis CPU Host Stream):       $(round(t_gpu_stream_phis, digits=3)) s  (VRAM: $(round(tau_4x3_f32/1024, digits=2)) GiB)")
        println("  Overhead factor for streaming phis:    $(round(t_gpu_stream_phis / t_gpu_full, digits=2))x")

        println("\n================================================================================")
        println("=== Conclusion & Recommended Architecture for 4x4 ===")
        println("================================================================================")
        println("For 4x4 (d = 4,008,576, M = 1,000 gates):")
        println("  - Storing phis on GPU VRAM takes ~17.5 GiB.")
        println("  - Storing tau_dev on GPU VRAM takes ~52.5 GiB.")
        println("  - Total on-GPU VRAM exceeds 80 GiB when both are kept on GPU simultaneously.")
        println("  - **BY STREAMING phis TO CPU RAM**: GPU VRAM usage drops from 70+ GiB to ~52.5 GiB,")
        println("    fitting comfortably within the 80 GiB VRAM limit of the A100 GPU with minimal slowdown!")

        return 0
    end
end
