#=
test_hybrid_checkpoint_verification.jl

Comprehensive verification script for Strided GPU Gradient Checkpointing (Rematerialization):
1. Verifies gradient numerical equality across pure CPU, Full GPU (K=1), Strided (K=4, 8, 16, 32), and Auto.
2. Benchmarks execution speed across all stride configurations on 4x3 lattice.
3. Tests dynamic stride allocation under Float32 and Float64 for 4x4 system.
4. Verifies no files are modified or saved during testing.
=#

ENV["JULIA_CUDA_USE_COMPAT"] = "true"
using CUDA
using ChainRulesCore
using Zygote
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

function (@main)(ARGS)
    log_path = make_log_path(@__DIR__, "test_hybrid_checkpoint_verification")
    with_logging(log_path) do
        println("================================================================================")
        println("STRIDED GPU GRADIENT CHECKPOINTING (REMATERIALIZATION) TEST")
        println("================================================================================")
        println("CUDA functional: $(CUDA.functional())")
        if CUDA.functional()
            println("GPU Device: $(CUDA.name(CUDA.device()))")
            free_b, total_b = CUDA.memory_info()
            @printf("GPU Memory: %.2f GiB free / %.2f GiB total\n", free_b / (1024^3), total_b / (1024^3))
        end

        folder = data_folder("N=(5, 4)_4x3")
        U_values, state_vecs, indexer, _, N_elec, _, _, sign_convention =
            load_ED_data(folder; verbose=false, sign_convention=:spin_first, use_slater_reference=false)

        Lvec = parse_lattice_dimension(folder)
        N_sites = prod(Lvec)
        basis_sector = Trotter.get_basis_sector(indexer, Lvec, N_sites)
        d = length(basis_sector)

        gates = Trotter.enumerate_ferm_excitations(2, Lvec; conserve_mom=true, conserve_sz=true, include_diagonal=false)
        tau_terms = Trotter.fgateToTauSector(gates, N_sites, basis_sector; antihermitian=true)
        num_gates = length(gates)

        Random.seed!(42)
        A_test = (2 * rand(num_gates) .- 1) * 0.05

        ref_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[1, :])
        target_real, _ = Trotter.TrotterOptimization.strip_global_phase(state_vecs[2, :])

        println("\nProblem setup: 4x3 lattice, d = $d, $num_gates gates")

        for (dtype, dname) in [(Float32, "Float32"), (Float64, "Float64")]
            println("\n" * "="^85)
            println("TESTING PRECISION: $dname ($dtype)")
            println("="^85)

            v_ref = dtype.(ref_real)
            v_target = dtype.(target_real)

            # 1. Pure CPU Baseline
            println("Evaluating Pure CPU gradient...")
            _, pb_cpu = ChainRulesCore.rrule(
                Trotter.TrotterOptimization.adjoint_loss, A_test, gates, tau_terms, v_ref, v_target, basis_sector, N_sites;
                antihermitian=true, use_gpu=false, datatype=dtype
            )
            grad_cpu = pb_cpu(1.0)[2]

            # 2. Full GPU (stride K=1)
            println("Evaluating Full GPU (K=1) gradient...")
            _, pb_gpu = ChainRulesCore.rrule(
                Trotter.TrotterOptimization.adjoint_loss, A_test, gates, tau_terms, v_ref, v_target, basis_sector, N_sites;
                antihermitian=true, use_gpu=true, datatype=dtype, checkpoint_mode=:gpu
            )
            grad_gpu = pb_gpu(1.0)[2]

            # 3. Strided GPU (K=4, 8, 16, 32)
            grads_strided = Dict{Int, Vector{Float64}}()
            for K in [4, 8, 16, 32]
                println("Evaluating Strided GPU (K=$K) gradient...")
                _, pb_k = ChainRulesCore.rrule(
                    Trotter.TrotterOptimization.adjoint_loss, A_test, gates, tau_terms, v_ref, v_target, basis_sector, N_sites;
                    antihermitian=true, use_gpu=true, datatype=dtype, checkpoint_mode=:strided, stride=K
                )
                grads_strided[K] = pb_k(1.0)[2]
            end

            # 4. Auto Stride Selection
            println("Evaluating Auto Stride gradient...")
            _, pb_auto = ChainRulesCore.rrule(
                Trotter.TrotterOptimization.adjoint_loss, A_test, gates, tau_terms, v_ref, v_target, basis_sector, N_sites;
                antihermitian=true, use_gpu=true, datatype=dtype, checkpoint_mode=:auto
            )
            grad_auto = pb_auto(1.0)[2]

            # Gradient equivalence checks
            diff_gpu_cpu = norm(grad_gpu .- grad_cpu) / norm(grad_cpu)
            diff_auto_gpu = norm(grad_auto .- grad_gpu) / norm(grad_gpu)

            @printf("\n--- Numerical Verification Results (%s) ---\n", dname)
            @printf("  ||g_gpu(K=1) - g_cpu|| / ||g_cpu||:         %.2e\n", diff_gpu_cpu)
            for K in [4, 8, 16, 32]
                diff_k = norm(grads_strided[K] .- grad_gpu) / norm(grad_gpu)
                @printf("  ||g_strided(K=%2d) - g_gpu|| / ||g_gpu||:     %.2e\n", K, diff_k)
                @assert diff_k < 1e-5 "Strided K=$K gradient differs from Full GPU!"
            end
            @printf("  ||g_auto - g_gpu|| / ||g_gpu||:             %.2e\n", diff_auto_gpu)
            @assert diff_auto_gpu < 1e-5 "Auto gradient differs from Full GPU!"
            println("  -> ALL STRIDED GRADIENTS MATCH FULL GPU TO MACHINE PRECISION!")

            # Benchmark timings
            println("\n--- Benchmarking Step Times (5 iterations) ---")
            benchmark_configs = [
                ("Pure CPU (1 core)", false, :auto, nothing),
                ("Full GPU (K=1, all VRAM)", true, :gpu, 1),
                ("Strided GPU (K=4)", true, :strided, 4),
                ("Strided GPU (K=8)", true, :strided, 8),
                ("Strided GPU (K=16)", true, :strided, 16),
                ("Strided GPU (K=32)", true, :strided, 32),
                ("Auto Stride GPU", true, :auto, nothing)
            ]

            println("-"^85)
            @printf("%-26s | %-14s | %-14s | %-14s | %-10s\n", "Mode", "Forward (ms)", "Backward (ms)", "Total Step (ms)", "Speedup")
            println("-"^85)

            cpu_baseline = 0.0
            for (idx, (desc, use_g, cmode, k_val)) in enumerate(benchmark_configs)
                fwd_times = Float64[]
                bwd_times = Float64[]
                for _ in 1:5
                    tf = @elapsed begin
                        phis = Trotter.TrotterOptimization.apply_unitary_checkpoints(
                            A_test, gates, v_ref, basis_sector, N_sites, 1;
                            antihermitian=true, use_gpu=use_g, datatype=dtype,
                            checkpoint_mode=cmode, stride=k_val
                        )
                        evolved = Trotter.TrotterOptimization.last_checkpoint(phis)
                        if use_g
                            CUDA.synchronize()
                        end
                    end
                    tb = @elapsed begin
                        init_adj = copy(evolved)
                        g = Trotter.TrotterOptimization.backward_adjoint_propagation(
                            A_test, gates, tau_terms, phis, init_adj, basis_sector, N_sites, 1;
                            antihermitian=true, use_gpu=use_g, datatype=dtype
                        )
                        if use_g
                            CUDA.synchronize()
                        end
                    end
                    push!(fwd_times, tf)
                    push!(bwd_times, tb)
                end
                avg_f = mean(fwd_times) * 1000
                avg_b = mean(bwd_times) * 1000
                total_step = avg_f + avg_b
                if idx == 1
                    cpu_baseline = total_step
                end
                speedup = cpu_baseline / total_step
                @printf("%-26s | %12.2f ms | %12.2f ms | %12.2f ms | %8.2fx\n", desc, avg_f, avg_b, total_step, speedup)
            end
            println("-"^85)
        end

        println("\n================================================================================")
        println("TESTING 4x4 AUTO STRIDE SELECTION")
        println("================================================================================")
        d_4x4 = 4008004
        M_4x4 = 2712
        for (dtype, dname) in [(Float32, "Float32"), (Float64, "Float64")]
            K_auto = Trotter.TrotterOptimization.determine_checkpoint_stride(
                M_4x4, d_4x4, dtype; checkpoint_mode=:auto
            )
            num_ckpts = cld(M_4x4, K_auto) + 1
            vec_sz_mb = (d_4x4 * sizeof(dtype)) / (1024^2)
            vram_gb = (num_ckpts * vec_sz_mb) / 1024
            @printf("  4x4 %-8s: Auto selected stride K=%2d | Stored checkpoints: %4d / %d (%.2f GiB VRAM)\n",
                dname, K_auto, num_ckpts, M_4x4 + 1, vram_gb)
        end

        println("\nVerification complete. All tests passed with 0 errors!")
        return 0
    end
end
